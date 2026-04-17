"""
data/datamodule.py
──────────────────────────────────────────────────────────────────────────────
PyTorch Lightning DataModule for the DRKG knowledge graph.

🎓 WHAT IS A LightningDataModule?
  It's a class that organizes all your data logic in one place:
  - setup(): load and split data
  - train_dataloader(): return training batches
  - val_dataloader(): return validation batches
  - test_dataloader(): return test batches

  WHY? Because in distributed training (multiple GPUs), Lightning
  calls setup() on each GPU worker. Having a DataModule ensures
  every GPU gets the right data without duplicating code.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from loguru import logger


DATA_DIR = Path(__file__).parent / "processed"


class TripleDataset(Dataset):
    """
    PyTorch Dataset for knowledge graph triples.
    
    🎓 PyTorch Dataset Protocol:
    Any Dataset must implement:
      __len__():      return total number of samples
      __getitem__(i): return the i-th sample as tensors
    
    DataLoader will call these to build batches.
    """

    def __init__(self, triples: np.ndarray):
        """
        Args:
            triples: (N, 3) array of [head_id, relation_id, tail_id]
        """
        self.triples = torch.LongTensor(triples)  # LongTensor = int64, required for embedding lookup

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        triple = self.triples[idx]
        return triple[0], triple[1], triple[2]  # head, relation, tail


class DRKGDataModule(L.LightningDataModule):
    """
    LightningDataModule for the DRKG dataset.
    
    Handles loading integer-encoded triples and creating DataLoaders.
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        batch_size: int = 1024,
        num_workers: int = 4,
        # For fast testing, use only a fraction of the data
        train_fraction: float = 1.0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_fraction = train_fraction

        # These are set in setup()
        self.train_dataset: Optional[TripleDataset] = None
        self.val_dataset: Optional[TripleDataset] = None
        self.test_dataset: Optional[TripleDataset] = None
        self.n_entities: int = 0
        self.n_relations: int = 0

    def prepare_data(self):
        """
        Called once to check data exists.
        Do NOT assign state here (this runs only on rank 0 in distributed training).
        """
        required = ["train.tsv", "valid.tsv", "test.tsv", "entity2id.json", "relation2id.json"]
        for fname in required:
            path = self.data_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing: {path}\n"
                    f"Run: python data/preprocess.py"
                )

    def setup(self, stage: Optional[str] = None):
        """
        Load data and create datasets. Called on ALL GPU workers.
        
        🎓 STAGES:
        - "fit"    → called before training (loads train + val)
        - "test"   → called before testing (loads test)
        - "predict"→ called before prediction
        - None     → load everything
        """
        # Load entity and relation mappings
        with open(self.data_dir / "entity2id.json") as f:
            entity2id = json.load(f)
        with open(self.data_dir / "relation2id.json") as f:
            relation2id = json.load(f)

        self.n_entities = len(entity2id)
        self.n_relations = len(relation2id)

        logger.info(f"📊 Dataset stats: {self.n_entities:,} entities, {self.n_relations:,} relations")

        def load_split(filename: str) -> np.ndarray:
            """Load a TSV split and convert entity/relation names to integer IDs."""
            rows = []
            with open(self.data_dir / filename) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != 3:
                        continue
                    head, relation, tail = parts
                    # Skip if any entity/relation is unknown (shouldn't happen after preprocessing)
                    if head not in entity2id or tail not in entity2id or relation not in relation2id:
                        continue
                    rows.append([entity2id[head], relation2id[relation], entity2id[tail]])
            return np.array(rows, dtype=np.int64)

        if stage in ("fit", None):
            train_triples = load_split("train.tsv")
            val_triples = load_split("valid.tsv")

            # Optionally subsample training data for fast prototyping
            if self.train_fraction < 1.0:
                n_keep = int(len(train_triples) * self.train_fraction)
                idx = np.random.choice(len(train_triples), n_keep, replace=False)
                train_triples = train_triples[idx]
                logger.warning(f"⚡ Using {self.train_fraction:.0%} of training data: {len(train_triples):,} triples")

            self.train_dataset = TripleDataset(train_triples)
            self.val_dataset = TripleDataset(val_triples)
            logger.info(f"   Train: {len(self.train_dataset):,} | Val: {len(self.val_dataset):,}")

        if stage in ("test", None):
            test_triples = load_split("test.tsv")
            self.test_dataset = TripleDataset(test_triples)
            logger.info(f"   Test: {len(self.test_dataset):,}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,           # Shuffle training data every epoch
            num_workers=self.num_workers,
            pin_memory=True,        # Faster GPU data transfer
            drop_last=True,         # Drop incomplete last batch for stable training
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,  # Larger batch = faster validation
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
