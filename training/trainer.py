"""
training/trainer.py
──────────────────────────────────────────────────────────────────────────────
Main training script using PyTorch Lightning + MLflow.

🎓 WHAT YOU'LL LEARN:

1. PYTORCH LIGHTNING TRAINER:
   All the boilerplate you'd write manually:
   ┌─────────────────────────────────────────────────────┐
   │  for epoch in range(max_epochs):                    │
   │      for batch in train_dataloader:                 │
   │          optimizer.zero_grad()                      │
   │          loss = model.training_step(batch, i)       │
   │          loss.backward()                            │
   │          optimizer.step()                           │
   │      for batch in val_dataloader:                   │
   │          model.validation_step(batch, i)            │
   └─────────────────────────────────────────────────────┘
   Lightning writes ALL of this. You just call: trainer.fit(model, datamodule)

2. CALLBACKS:
   Callbacks are hooks that run at specific training events.
   - ModelCheckpoint: saves model when val_loss improves
   - EarlyStopping: stops training when val_loss stops improving
   - LearningRateMonitor: logs LR to MLflow every step
   - RichProgressBar: beautiful terminal progress bar

3. MLflow LOGGING:
   - Every hyperparameter is logged → you can compare runs
   - Every metric (loss, MRR, Hits@10) is logged per epoch
   - Best model checkpoint is saved and registered
   - You can access the MLflow UI at localhost:5000 to visualize

4. PRECISION:
   - Default: float32 (32 bits per number)
   - "16-mixed": brain float16 (BF16) — uses 2x less GPU memory!
   - This lets you train with larger batches or bigger models
   - Modern GPUs (RTX 3050+) support BF16 natively
"""

import json
from pathlib import Path
from typing import Optional

import mlflow
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import MLFlowLogger
from loguru import logger

from data.datamodule import DRKGDataModule
from models.kge_model import RotatEModel


# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

# ── Hyperparameters ────────────────────────────────────────────────────────────
CONFIG = {
    # Data
    "batch_size": 1024,
    "num_workers": 4,
    "train_fraction": 0.1,  # Use 10% for fast first run (increase for final training)

    # Model
    "embed_dim": 256,        # Higher = more capacity, more memory
    "gamma": 12.0,           # Margin hyperparameter
    "n_negative_samples": 128,

    # Training
    "lr": 3e-4,
    "max_epochs": 50,
    "precision": "16-mixed",  # Use BF16 for faster training on modern GPUs
    "gradient_clip_val": 1.0,  # Prevents gradient explosion

    # MLflow
    "experiment_name": "BioKG-Disease-LinkPrediction",
    "run_name": "rotate_biobridge_drkg_v1",
    
    # Use BioBridge embeddings for initialization?
    "use_biobridge_init": False,  # Set True after running models/biobridge_encoder.py
}


def train(config: dict = CONFIG, checkpoint_path: Optional[str] = None):
    """
    Main training function.
    
    Args:
        config: Training configuration dictionary
        checkpoint_path: Path to resume training from a checkpoint
    """
    logger.info("🚀 Starting BioKG Training Pipeline")
    logger.info("=" * 60)
    logger.info(json.dumps(config, indent=2))

    # ── 1. Data Module ─────────────────────────────────────────────────────────
    logger.info("\n📊 Setting up DRKG DataModule...")
    datamodule = DRKGDataModule(
        data_dir=DATA_DIR,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        train_fraction=config["train_fraction"],
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    # ── 2. Load BioBridge embeddings (optional) ────────────────────────────────
    pretrained_embeddings = None
    if config["use_biobridge_init"]:
        emb_path = DATA_DIR / "entity_embeddings.npy"
        if emb_path.exists():
            import numpy as np
            logger.info(f"🧬 Loading BioBridge embeddings from {emb_path}")
            pretrained_embeddings = np.load(emb_path)
        else:
            logger.warning("BioBridge embeddings not found. Run models/biobridge_encoder.py first.")

    # ── 3. Model ───────────────────────────────────────────────────────────────
    logger.info("\n🧠 Building RotatE model...")
    model = RotatEModel(
        n_entities=datamodule.n_entities,
        n_relations=datamodule.n_relations,
        embed_dim=config["embed_dim"],
        gamma=config["gamma"],
        lr=config["lr"],
        n_negative_samples=config["n_negative_samples"],
        pretrained_embeddings=pretrained_embeddings,
    )

    logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 4. Callbacks ───────────────────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # Save the best model based on validation loss
        ModelCheckpoint(
            dirpath=CHECKPOINT_DIR,
            filename="biokg-{epoch:02d}-{val/loss:.3f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,      # Keep top 3 checkpoints
            verbose=True,
        ),
        # Stop early if validation loss doesn't improve
        EarlyStopping(
            monitor="val/loss",
            patience=10,        # Wait 10 epochs before stopping
            mode="min",
            verbose=True,
        ),
        # Log learning rate to MLflow
        LearningRateMonitor(logging_interval="epoch"),
        # Beautiful progress bar
        RichProgressBar(),
    ]

    # ── 5. MLflow Logger ───────────────────────────────────────────────────────
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow_logger = MLFlowLogger(
        experiment_name=config["experiment_name"],
        run_name=config["run_name"],
        tracking_uri=f"file://{MLFLOW_DIR.resolve()}",
        log_model=True,  # Save model to MLflow artifact store
    )

    # Log all config hyperparameters
    mlflow_logger.log_hyperparams(config)

    # ── 6. Trainer ─────────────────────────────────────────────────────────────
    logger.info("\n⚡ Configuring PyTorch Lightning Trainer...")
    trainer = L.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu" if __import__("torch").cuda.is_available() else "cpu",
        devices=1,
        precision=config["precision"],          # BF16 mixed precision
        gradient_clip_val=config["gradient_clip_val"],
        callbacks=callbacks,
        logger=mlflow_logger,
        log_every_n_steps=10,
        val_check_interval=0.25,               # Validate 4x per epoch
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # ── 7. Train! ──────────────────────────────────────────────────────────────
    logger.info("\n🏋️  Starting training...")
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=checkpoint_path,
    )

    logger.success(f"\n🎉 Training complete!")
    logger.info(f"   Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    logger.info(f"   Best val loss: {trainer.checkpoint_callback.best_model_score:.4f}")
    logger.info(f"\n   View MLflow UI: mlflow ui --backend-store-uri {MLFLOW_DIR.resolve()}")

    return trainer.checkpoint_callback.best_model_path


if __name__ == "__main__":
    best_model_path = train()
    logger.info(f"\n✅ Best model saved to: {best_model_path}")
