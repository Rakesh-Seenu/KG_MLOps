"""
data/datamodule.py
──────────────────────────────────────────────────────────────────────────────
PyTorch Lightning DataModule for the PrimeKG knowledge graph.

🎓 WHAT IS A LightningDataModule?
  It's a class that organizes all your data logic in one place:
  - setup(): load and split data
  - train_dataloader(): return training batches
  - val_dataloader(): return validation batches
  - test_dataloader(): return test batches
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from loguru import logger


DATA_DIR = Path(__file__).parent / "processed"


class TripleDataset(Dataset):
    """
    PyTorch Dataset for knowledge graph triples.
    Returns (head_id, relation_id, tail_id) tensors.
    """

    def __init__(self, triples: np.ndarray):
        self.triples = torch.LongTensor(triples)

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        triple = self.triples[idx]
        return triple[0], triple[1], triple[2]


class PrimeKGDataModule(L.LightningDataModule):
    """
    LightningDataModule for the PrimeKG dataset.
    Loads processed Parquet files and creates 80/10/10 splits dynamically
    if specific splits aren't provided.
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        batch_size: int = 1024,
        num_workers: int = 4,
        train_fraction: float = 1.0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_fraction = train_fraction

        self.train_dataset: Optional[TripleDataset] = None
        self.val_dataset: Optional[TripleDataset] = None
        self.test_dataset: Optional[TripleDataset] = None
        self.max_node_index: int = 0
        self.n_relations: int = 0

    def prepare_data(self):
        """Check if preprocessed data exists."""
        required = ["primekg_edges.parquet", "relation2id.json", "stats.json"]
        for fname in required:
            path = self.data_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"Missing: {path}\nRun: python data/preprocess.py")

    def setup(self, stage: Optional[str] = None):
        """Load Parquet data, split it, and create datasets."""
        with open(self.data_dir / "stats.json") as f:
            stats = json.load(f)
        
        self.max_node_index = stats["max_node_index"]
        self.n_relations = stats["n_relations"]
        logger.info(f"📊 Dataset stats: {self.max_node_index+1:,} max node ID, {self.n_relations:,} relations")

        # Load edges from Parquet
        logger.info("Loading PrimeKG edges into memory...")
        df = pd.read_parquet(self.data_dir / "primekg_edges.parquet")
        
        # We need head_index, relation_id, tail_index
        triples = df[['head_index', 'relation_id', 'tail_index']].to_numpy(dtype=np.int64)
        
        # Shuffle explicitly using a fixed seed for reproducible splits
        np.random.seed(42)
        np.random.shuffle(triples)
        
        n = len(triples)
        n_train = int(n * 0.8)
        n_valid = int(n * 0.1)

        train_triples = triples[:n_train]
        val_triples = triples[n_train:n_train + n_valid]
        test_triples = triples[n_train + n_valid:]

        if stage in ("fit", None):
            if self.train_fraction < 1.0:
                n_keep = int(len(train_triples) * self.train_fraction)
                train_triples = train_triples[:n_keep]
                logger.warning(f"⚡ Using {self.train_fraction:.0%} of training data: {len(train_triples):,} triples")

            self.train_dataset = TripleDataset(train_triples)
            self.val_dataset = TripleDataset(val_triples)
            logger.info(f"   Train: {len(self.train_dataset):,} | Val: {len(self.val_dataset):,}")

        if stage in ("test", None):
            self.test_dataset = TripleDataset(test_triples)
            logger.info(f"   Test:  {len(self.test_dataset):,}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
