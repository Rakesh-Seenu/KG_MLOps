"""
BioBridge & PrimeKG Dataset Downloader
This script downloads the specific files required to recreate the BioBridge multimodal 
environment on top of the PrimeKG dataset.
"""

import os
import requests
from pathlib import Path
from loguru import logger
from tqdm import tqdm

DATA_DIR = Path('data')
RAW_DIR = DATA_DIR / 'raw'
BIOBRIDGE_DIR = RAW_DIR / 'biobridge'
PRIMEKG_DIR = RAW_DIR / 'primekg'

BIOBRIDGE_DIR.mkdir(parents=True, exist_ok=True)
PRIMEKG_DIR.mkdir(parents=True, exist_ok=True)

(BIOBRIDGE_DIR / 'embeddings').mkdir(exist_ok=True)
(BIOBRIDGE_DIR / 'processed').mkdir(exist_ok=True)

def download_file(url: str, dest_path: Path, desc: str = "Downloading"):
    if dest_path.exists():
        logger.info(f"⏭️  {dest_path.name} already exists, skipping.")
        return
        
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f, tqdm(desc=desc, total=total, unit='iB', unit_scale=True) as pb:
            for chunk in response.iter_content(8192):
                pb.update(f.write(chunk))
        logger.info(f"✅ Saved: {dest_path.name}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()  # remove partial file

def main():
    logger.info("📥 Starting BioBridge + PrimeKG download pipeline...")
    
    # 1. Download PrimeKG files (Harvard Dataverse)
    logger.info("Fetching PrimeKG Graph...")
    PRIMEKG_GRAPH_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"
    
    download_file(PRIMEKG_GRAPH_URL, PRIMEKG_DIR / 'kg.csv', 'PrimeKG Graph')
    
    # 2. Download BioBridge Data Config
    logger.info("Fetching BioBridge Data Config...")
    CONFIG_URL = "https://raw.githubusercontent.com/RyanWangZf/BioBridge/refs/heads/main/data/BindData/data_config.json"
    download_file(CONFIG_URL, BIOBRIDGE_DIR / 'data_config.json', 'Data Config')
    
    # 3. Download Precomputed Embeddings (Huge files)
    logger.info("Fetching BioBridge Node Embeddings (.pkl)...")
    base_emb_url = "https://media.githubusercontent.com/media/RyanWangZf/BioBridge/refs/heads/main/data/embeddings/esm2b_unimo_pubmedbert"
    embeddings = ['protein.pkl', 'mf.pkl', 'cc.pkl', 'bp.pkl', 'drug.pkl', 'disease.pkl']
    
    for emb in embeddings:
        download_file(f"{base_emb_url}/{emb}", BIOBRIDGE_DIR / 'embeddings' / emb, f"Emb: {emb}")

    # 4. Download Processed Node Files
    logger.info("Fetching BioBridge Processed Nodes (.csv)...")
    base_proc_url = "https://media.githubusercontent.com/media/RyanWangZf/BioBridge/refs/heads/main/data/Processed"
    processed_files = {
        'protein.csv': 'protein.csv',
        'molecular.csv': 'mf.csv',
        'cellular.csv': 'cc.csv',
        'biological.csv': 'bp.csv',
        'drug.csv': 'drug.csv',
        'disease.csv': 'disease.csv'
    }
    
    for remote, local in processed_files.items():
        download_file(f"{base_proc_url}/{remote}", BIOBRIDGE_DIR / 'processed' / local, f"Process: {local}")

    logger.info("🎉 All downloads complete!")

if __name__ == "__main__":
    main()
