"""
data/download_drkg.py
──────────────────────────────────────────────────────────────────────────────
Downloads the DRKG (Drug Repurposing Knowledge Graph) dataset from the
official Microsoft Research GitHub repository.

DRKG is a comprehensive biological knowledge graph relating genes, compounds,
diseases, biological processes, side effects and symptoms.

Stats:
  • 97,238 entities
  • 5,874,261 triples
  • 107 relation types

What you'll learn here:
  - How to work with large TSV graph files
  - Entity type distribution in biomedical KGs
  - What a "triple" (head, relation, tail) means
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "raw"
DRKG_URL = "https://raw.githubusercontent.com/gnn4dr/DRKG/master/drkg.tsv.gz"
ENTITY_URL = "https://raw.githubusercontent.com/gnn4dr/DRKG/master/embed/entities.tsv"
RELATION_URL = "https://raw.githubusercontent.com/gnn4dr/DRKG/master/embed/relations.tsv"


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> None:
    """
    Downloads a file with a progress bar.
    
    We use streaming=True so we don't load the whole file into memory —
    important for the 2GB DRKG file!
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        logger.info(f"⏭️  {dest_path.name} already exists, skipping download.")
        return

    logger.info(f"📥 Downloading {desc} from {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_bytes = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        desc=desc,
        total=total_bytes,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            progress_bar.update(size)

    logger.success(f"✅ Saved to {dest_path}")


def download_all() -> None:
    """Downloads all DRKG components."""
    logger.info("🧬 Starting DRKG Dataset Download")
    logger.info("=" * 60)

    download_file(DRKG_URL, DATA_DIR / "drkg.tsv.gz", desc="DRKG triples (2GB compressed)")
    download_file(ENTITY_URL, DATA_DIR / "entities.tsv", desc="Entity list")
    download_file(RELATION_URL, DATA_DIR / "relations.tsv", desc="Relation list")

    logger.success(f"\n🎉 All files downloaded to: {DATA_DIR.resolve()}")
    logger.info("\n📂 Files:")
    for f in DATA_DIR.iterdir():
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"   {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    download_all()
