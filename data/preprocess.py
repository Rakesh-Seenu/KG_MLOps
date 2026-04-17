"""
data/preprocess.py
──────────────────────────────────────────────────────────────────────────────
GPU-accelerated preprocessing of the DRKG knowledge graph using NVIDIA RAPIDS.

🎓 WHAT YOU'LL LEARN:
  1. cuDF — GPU DataFrame (like pandas, but on GPU)
     - cudf.read_csv() → loads data directly into GPU memory
     - Operations like .value_counts(), .merge(), .groupby() all run on GPU
  
  2. cuGraph — GPU Graph Analytics (like NetworkX, but on GPU)
     - Builds a graph from triples and runs algorithms like PageRank
     - PageRank of disease nodes tells us which diseases are most connected

  3. WHY GPU PREPROCESSING MATTERS:
     - DRKG has 5.8M rows
     - pandas on CPU: ~30-60 seconds to load and process
     - cuDF on GPU: ~2-5 seconds (10-15x faster)
     - This is EXACTLY what Gurdeep meant by "large datasets on GPU clusters"

📊 OUTPUT: 
  - data/processed/train.tsv  (80% of triples)
  - data/processed/valid.tsv  (10%)
  - data/processed/test.tsv   (10%)
  - data/processed/entity2id.json
  - data/processed/relation2id.json
  - data/processed/stats.json
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd  # fallback if no GPU
from loguru import logger
from rich.console import Console
from rich.table import Table

console = Console()

DATA_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"

# ── Try to use RAPIDS, fall back to pandas if not available ───────────────────
try:
    import cudf
    import cugraph
    USE_GPU = True
    logger.success("⚡ NVIDIA RAPIDS detected! Using GPU-accelerated processing.")
except ImportError:
    USE_GPU = False
    logger.warning("⚠️  RAPIDS not installed. Falling back to CPU pandas.")
    logger.warning("   In Colab: !pip install cudf-cu12 --extra-index-url=https://pypi.nvidia.com")


def load_drkg() -> pd.DataFrame:
    """
    Load the DRKG TSV file into a DataFrame (GPU or CPU).
    
    Each row is a TRIPLE: (head_entity, relation, tail_entity)
    Example: ("Gene::BRCA1", "DRUGBANK::treats", "Disease::Breast_Cancer")
    """
    drkg_path = DATA_DIR / "drkg.tsv.gz"

    if not drkg_path.exists():
        logger.error(f"DRKG not found at {drkg_path}. Run: python data/download_drkg.py")
        raise FileNotFoundError(drkg_path)

    logger.info(f"📂 Loading DRKG from {drkg_path}...")
    start = time.time()

    if USE_GPU:
        # cuDF: reads directly into GPU VRAM — no CPU bottleneck
        df = cudf.read_csv(
            str(drkg_path),
            sep="\t",
            header=None,
            names=["head", "relation", "tail"],
            compression="gzip",
        )
    else:
        df = pd.read_csv(
            drkg_path,
            sep="\t",
            header=None,
            names=["head", "relation", "tail"],
            compression="gzip",
        )

    elapsed = time.time() - start
    mode = "GPU (cuDF)" if USE_GPU else "CPU (pandas)"
    logger.success(f"✅ Loaded {len(df):,} triples in {elapsed:.2f}s using {mode}")
    return df


def analyze_entity_types(df) -> dict:
    """
    Analyze the entity type distribution.

    DRKG entity format: "TypeName::EntityID"  e.g. "Gene::9606/23210"
    We extract the type prefix to understand the graph structure.
    """
    logger.info("🔍 Analyzing entity type distribution...")

    if USE_GPU:
        # GPU string operations using cuDF
        head_types = df["head"].str.split("::").list.get(0)
        tail_types = df["tail"].str.split("::").list.get(0)
        
        head_counts = head_types.value_counts().to_pandas()
        tail_counts = tail_types.value_counts().to_pandas()
    else:
        head_types = df["head"].str.split("::").str[0]
        tail_types = df["tail"].str.split("::").str[0]
        
        head_counts = head_types.value_counts()
        tail_counts = tail_types.value_counts()

    # Merge head and tail counts
    all_type_counts = head_counts.add(tail_counts, fill_value=0).sort_values(ascending=False)

    # Pretty print with rich table
    table = Table(title="Entity Type Distribution in DRKG")
    table.add_column("Entity Type", style="cyan")
    table.add_column("Count", style="magenta", justify="right")
    table.add_column("% of entities", style="green", justify="right")
    total = all_type_counts.sum()
    for entity_type, count in all_type_counts.head(15).items():
        table.add_row(entity_type, f"{int(count):,}", f"{100*count/total:.1f}%")
    console.print(table)

    return all_type_counts.to_dict()


def build_entity_relation_mappings(df) -> tuple[dict, dict]:
    """
    Create integer ID mappings for all entities and relations.
    
    KGE models work with integer IDs, not strings. This maps:
      "Gene::BRCA1"  →  42
      "Disease::Cancer"  →  107
    """
    logger.info("🗺️  Building entity and relation ID mappings...")

    if USE_GPU:
        all_entities = cudf.concat([df["head"], df["tail"]]).unique().to_pandas()
        all_relations = df["relation"].unique().to_pandas()
    else:
        all_entities = pd.concat([df["head"], df["tail"]]).unique()
        all_relations = df["relation"].unique()

    entity2id = {entity: idx for idx, entity in enumerate(sorted(all_entities))}
    relation2id = {rel: idx for idx, rel in enumerate(sorted(all_relations))}

    logger.success(f"✅ {len(entity2id):,} unique entities, {len(relation2id):,} unique relations")
    return entity2id, relation2id


def split_triples(df, train=0.8, valid=0.1, test=0.1, seed=42):
    """
    Split triples into train/validation/test sets.
    
    IMPORTANT: We use random splitting here for simplicity.
    In research, you'd use time-based or entity-based splitting
    to avoid data leakage — good thing to mention in your LinkedIn post!
    """
    assert abs(train + valid + test - 1.0) < 1e-6, "Splits must sum to 1.0"
    logger.info(f"✂️  Splitting {len(df):,} triples: {train:.0%}/{valid:.0%}/{test:.0%}")

    if USE_GPU:
        # cuDF doesn't have train_test_split, so we do it manually
        df_pandas = df.to_pandas()
    else:
        df_pandas = df

    # Shuffle
    df_shuffled = df_pandas.sample(frac=1, random_state=seed).reset_index(drop=True)

    n = len(df_shuffled)
    n_train = int(n * train)
    n_valid = int(n * valid)

    df_train = df_shuffled.iloc[:n_train]
    df_valid = df_shuffled.iloc[n_train:n_train + n_valid]
    df_test = df_shuffled.iloc[n_train + n_valid:]

    logger.success(f"   Train: {len(df_train):,} | Valid: {len(df_valid):,} | Test: {len(df_test):,}")
    return df_train, df_valid, df_test


def compute_graph_stats_with_cugraph(df) -> dict:
    """
    Use cuGraph (GPU graph library) to compute graph statistics.
    
    🎓 CONCEPT: PageRank
    PageRank assigns a score to each node based on how many important nodes 
    link to it. A high PageRank disease node = many genes/drugs connect to it.
    This is what Google originally used for web page ranking!
    """
    if not USE_GPU:
        logger.warning("Skipping cuGraph stats (GPU not available)")
        return {}

    logger.info("📊 Computing graph statistics with cuGraph (GPU)...")
    start = time.time()

    # Build cuGraph from triples
    # We need numeric IDs for cuGraph edges
    all_entities = cudf.concat([df["head"], df["tail"]]).unique().reset_index(drop=True)
    entity2id = cudf.Series(range(len(all_entities)), index=all_entities)

    edges = cudf.DataFrame({
        "src": df["head"].map(entity2id),
        "dst": df["tail"].map(entity2id),
    }).dropna()

    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(edges, source="src", destination="dst")

    # PageRank — GPU-accelerated
    pagerank_df = cugraph.pagerank(G)
    top_nodes = pagerank_df.sort_values("pagerank", ascending=False).head(10)

    elapsed = time.time() - start
    logger.success(f"✅ cuGraph stats computed in {elapsed:.2f}s")

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "pagerank_computed": True,
        "computation_time_seconds": elapsed,
    }


def save_outputs(df_train, df_valid, df_test, entity2id, relation2id, stats):
    """Save all processed outputs to disk."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"💾 Saving processed files to {PROCESSED_DIR}/...")

    # Save triple splits as TSV
    df_train.to_csv(PROCESSED_DIR / "train.tsv", sep="\t", index=False, header=False)
    df_valid.to_csv(PROCESSED_DIR / "valid.tsv", sep="\t", index=False, header=False)
    df_test.to_csv(PROCESSED_DIR / "test.tsv", sep="\t", index=False, header=False)

    # Save mappings as JSON
    with open(PROCESSED_DIR / "entity2id.json", "w") as f:
        json.dump(entity2id, f, indent=2)
    with open(PROCESSED_DIR / "relation2id.json", "w") as f:
        json.dump(relation2id, f, indent=2)

    # Save stats
    stats["n_entities"] = len(entity2id)
    stats["n_relations"] = len(relation2id)
    stats["n_train"] = len(df_train)
    stats["n_valid"] = len(df_valid)
    stats["n_test"] = len(df_test)

    with open(PROCESSED_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.success("✅ All files saved!")
    for f in PROCESSED_DIR.iterdir():
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"   {f.name}: {size_mb:.1f} MB")


def main():
    console.rule("[bold blue]🧬 DRKG GPU Preprocessing Pipeline")
    logger.info(f"Mode: {'⚡ GPU (RAPIDS)' if USE_GPU else '🐢 CPU (pandas)'}")

    # Step 1: Load data
    df = load_drkg()

    # Step 2: Analyze entity types (helps us understand the data)
    type_stats = analyze_entity_types(df)

    # Step 3: Build entity/relation mappings
    entity2id, relation2id = build_entity_relation_mappings(df)

    # Step 4: Split into train/valid/test
    df_train, df_valid, df_test = split_triples(df)

    # Step 5: GPU graph statistics (cuGraph)
    graph_stats = compute_graph_stats_with_cugraph(df)

    # Step 6: Save everything
    all_stats = {**type_stats, **graph_stats}
    save_outputs(df_train, df_valid, df_test, entity2id, relation2id, all_stats)

    console.rule("[bold green]✅ Preprocessing Complete!")


if __name__ == "__main__":
    main()
