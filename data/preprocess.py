"""
data/preprocess.py
──────────────────────────────────────────────────────────────────────────────
GPU-accelerated preprocessing of the PrimeKG knowledge graph using NVIDIA RAPIDS.

🎓 WHAT YOU'LL LEARN:
  1. cuDF — GPU DataFrame (like pandas, but on GPU)
     - cudf.read_csv() → loads PrimeKG's 8.1M edges directly into GPU memory
     - Operations like .value_counts(), .merge(), .groupby() all run on GPU
  
  2. BioBridge Modalities:
     - Filtering for specific semantic relationship types 
     - Aligning PrimeKG entities to BioBridge embeddings

📊 OUTPUT: 
  - data/processed/primekg_edges.parquet
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

DATA_DIR = Path(__file__).parent / "raw" / "primekg"
PROCESSED_DIR = Path(__file__).parent / "processed"

# Try to use RAPIDS, fall back to pandas if not available
try:
    import cudf
    USE_GPU = True
    logger.success("⚡ NVIDIA RAPIDS detected! Using GPU-accelerated processing.")
except ImportError:
    USE_GPU = False
    logger.warning("⚠️  RAPIDS not installed. Falling back to CPU pandas.")

def load_primekg() -> pd.DataFrame:
    """
    Load the PrimeKG TSV file into a DataFrame (GPU or CPU).
    
    Structure: head_index, head_id, head_type, tail_index, tail_id, tail_type, relation
    """
    primekg_path = DATA_DIR / "kg.csv"

    if not primekg_path.exists():
        logger.error(f"PrimeKG not found at {primekg_path}. Run: python data/download_biobridge.py")
        raise FileNotFoundError(primekg_path)

    logger.info(f"📂 Loading PrimeKG from {primekg_path}...")
    start = time.time()

    if USE_GPU:
        df = cudf.read_csv(
            str(primekg_path)
        )
    else:
        df = pd.read_csv(
            primekg_path, low_memory=False
        )

    # PrimeKG uses x and y inherently, rename them to standard KGE nomenclature
    df = df.rename(columns={'x_index': 'head_index', 'x_type': 'head_type', 'y_index': 'tail_index', 'y_type': 'tail_type'})

    elapsed = time.time() - start
    mode = "GPU (cuDF)" if USE_GPU else "CPU (pandas)"
    logger.success(f"✅ Loaded {len(df):,} PrimeKG triples in {elapsed:.2f}s using {mode}")
    return df

def analyze_entity_types(df) -> dict:
    """Analyze the distribution of node types (e.g. disease, drug, protein)."""
    logger.info("🔍 Analyzing PrimeKG entity type distribution...")

    if USE_GPU:
        head_counts = df["head_type"].value_counts().to_pandas()
        tail_counts = df["tail_type"].value_counts().to_pandas()
    else:
        head_counts = df["head_type"].value_counts()
        tail_counts = df["tail_type"].value_counts()

    all_type_counts = head_counts.add(tail_counts, fill_value=0).sort_values(ascending=False)

    table = Table(title="Entity Type Distribution in PrimeKG")
    table.add_column("Entity Type", style="cyan")
    table.add_column("Count (Edges)", style="magenta", justify="right")
    table.add_column("% of edges", style="green", justify="right")
    
    total = all_type_counts.sum()
    for entity_type, count in all_type_counts.head(15).items():
        table.add_row(str(entity_type), f"{int(count):,}", f"{100*count/total:.1f}%")
    console.print(table)

    return all_type_counts.to_dict()

def build_entity_relation_mappings(df) -> tuple[dict, dict]:
    """
    Create mappings based on PrimeKG's native indexing.
    PrimeKG provides `head_index` and `node_index` natively!
    We map `relation` text to contiguous IDs.
    """
    logger.info("🗺️  Building relation ID mappings...")

    if USE_GPU:
        all_relations = df["relation"].unique().to_pandas()
        max_node_idx = max(df["head_index"].max(), df["tail_index"].max())
    else:
        all_relations = df["relation"].unique()
        max_node_idx = max(df["head_index"].max(), df["tail_index"].max())

    relation2id = {rel: idx for idx, rel in enumerate(sorted(all_relations))}
    
    logger.success(f"✅ Max node index found: {max_node_idx:,}, {len(relation2id):,} unique relations")
    return {"max_node_index": int(max_node_idx)}, relation2id

def save_outputs(df, max_node_idx, relation2id, stats):
    """Save the clean processed output to Parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"💾 Saving processed PrimeKG to {PROCESSED_DIR}/...")

    if USE_GPU:
        df.to_parquet(PROCESSED_DIR / "primekg_edges.parquet")
    else:
        df.to_parquet(PROCESSED_DIR / "primekg_edges.parquet", engine='pyarrow')

    with open(PROCESSED_DIR / "relation2id.json", "w") as f:
        json.dump(relation2id, f, indent=2)

    stats["n_relations"] = len(relation2id)
    stats["max_node_index"] = max_node_idx
    stats["n_edges"] = len(df)

    with open(PROCESSED_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.success("✅ All files saved!")

def main():
    console.rule("[bold blue]🧬 BioBridge-PrimeKG Preprocessing Pipeline")
    
    df = load_primekg()
    type_stats = analyze_entity_types(df)
    node_stats, relation2id = build_entity_relation_mappings(df)
    
    # Store relation ID in the dataframe
    if USE_GPU:
        # map is slow in cudf, use merge or just CPU fallback for mapping config
        logger.info("Applying relation IDs...")
        mapping_df = cudf.DataFrame({'relation': list(relation2id.keys()), 'relation_id': list(relation2id.values())})
        df = df.merge(mapping_df, on='relation', how='left')
    else:
        df['relation_id'] = df['relation'].map(relation2id)

    all_stats = {**type_stats, **node_stats}
    save_outputs(df, node_stats['max_node_index'], relation2id, all_stats)

    console.rule("[bold green]✅ Preprocessing Complete!")

if __name__ == "__main__":
    main()
