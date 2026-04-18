import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from loguru import logger


class TripleDataset(Dataset):
    """
    Highly optimized PyTorch Dataset for knowledge graph triples.
    
    Returns (head_id, relation_id, tail_id) tensors.
    Uses LongTensor for compatibility with nn.Embedding lookups.
    """

    def __init__(self, triples: np.ndarray):
        """
        Args:
            triples: A numpy array of shape (N, 3) containing [head, relation, tail] indices.
        """
        self.triples = torch.LongTensor(triples)

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        triple = self.triples[idx]
        return triple[0], triple[1], triple[2]


class PrimeKGDataModule(L.LightningDataModule):
    """
    Industrial-strength LightningDataModule for the PrimeKG dataset.
    
    Handles:
    - Data verification (prepare_data).
    - Reproducible splitting (setup).
    - Multi-worker data loading (train/val/test dataloaders).
    """

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 1024,
        num_workers: int = 4,
        train_fraction: float = 1.0,
    ):
        """
        Args:
            data_dir: Path to the directory containing preprocessed parquet/json files.
            batch_size: Number of triples per GPU batch.
            num_workers: Number of CPU cores to use for data loading.
            train_fraction: Fraction of training data to use (useful for lite mode/debugging).
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_fraction = train_fraction

        self.train_dataset: Optional[TripleDataset] = None
        self.val_dataset: Optional[TripleDataset] = None
        self.test_dataset: Optional[TripleDataset] = None
        
        self.max_node_index: int = 0
        self.n_relations: int = 0

    def prepare_data(self) -> None:
        """
        Verifies that the required preprocessed files exist on disk.
        This runs once per machine in a distributed setup.
        """
        required_files = ["primekg_edges.parquet", "relation2id.json", "stats.json"]
        for fname in required_files:
            target_path = self.data_dir / fname
            if not target_path.exists():
                logger.error(f"❌ Critical data file missing: {target_path}")
                logger.info("Please run: `python data/preprocess.py` to generate processed data.")
                raise FileNotFoundError(f"Missing required artifact: {fname}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Loads the data from Parquet and splits it into training, validation, and test sets.
        Stage can be 'fit', 'validate', 'test', or 'predict'.
        """
        # 1. Load Statistics
        stats_path = self.data_dir / "stats.json"
        with open(stats_path, "r") as f:
            stats = json.load(f)
        
        self.max_node_index = stats["max_node_index"]
        self.n_relations = stats["n_relations"]
        
        logger.info(f"🧬 Dataset Intelligence: {self.max_node_index+1:,} nodes | {self.n_relations:,} relations")

        # 2. Load Edge Data
        parquet_path = self.data_dir / "primekg_edges.parquet"
        logger.info(f"⏳ Loading edge index from {parquet_path.name}...")
        df = pd.read_parquet(parquet_path)
        
        # Ensure we have the integer index columns
        triples = df[['head_index', 'relation_id', 'tail_index']].to_numpy(dtype=np.int64)
        
        # 3. Deterministic Shuffling & Splitting
        np.random.seed(42)  # For reproducible scientific results
        np.random.shuffle(triples)
        
        n_total = len(triples)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)

        train_raw = triples[:n_train]
        val_raw = triples[n_train:n_train + n_val]
        test_raw = triples[n_train + n_val:]

        # Apply Lite Mode if requested
        if self.train_fraction < 1.0:
            n_keep = int(len(train_raw) * self.train_fraction)
            train_raw = train_raw[:n_keep]
            logger.warning(f"⚠️ Lite Mode: Using {self.train_fraction:.1%} of training data ({len(train_raw):,} triples)")

        # 4. Assign to split-specific datasets
        if stage in ("fit", None):
            self.train_dataset = TripleDataset(train_raw)
            self.val_dataset = TripleDataset(val_raw)
            logger.info(f"   Splits created -> Train: {len(self.train_dataset):,}, Val: {len(self.val_dataset):,}")

        if stage in ("test", "predict", None):
            self.test_dataset = TripleDataset(test_raw)
            logger.info(f"   Splits created -> Test: {len(self.test_dataset):,}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )
