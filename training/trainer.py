"""
training/trainer.py
──────────────────────────────────────────────────────────────────────────────
Main training script using PyTorch Lightning + MLflow for PrimeKG link prediction.
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

from data.datamodule import PrimeKGDataModule
from models.kge_model import RotatEModel

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"

# ── Hyperparameters ────────────────────────────────────────────────────────────
CONFIG = {
    "batch_size": 1024,
    "num_workers": 4,
    "train_fraction": 1.0, 

    # Model
    "embed_dim": 384,        
    "gamma": 12.0,           
    "n_negative_samples": 256,
    "use_biobridge": True, 

    # Training
    "lr": 5e-4,
    "max_epochs": 50,
    "precision": "16-mixed",  
    "gradient_clip_val": 1.0,  

    # MLflow
    "experiment_name": "BioKG-PrimeKG-LinkPrediction",
    "run_name": "rotate_biobridge_primekg_v1",
}


def train(config: dict = CONFIG, checkpoint_path: Optional[str] = None):
    logger.info("🚀 Starting PrimeKG Training Pipeline")
    logger.info("=" * 60)
    logger.info(json.dumps(config, indent=2))

    logger.info("\n📊 Setting up PrimeKG DataModule...")
    datamodule = PrimeKGDataModule(
        data_dir=DATA_DIR,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        train_fraction=config["train_fraction"],
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    logger.info("\n🧠 Building RotatE model...")
    model = RotatEModel(
        max_node_idx=datamodule.max_node_index,
        n_relations=datamodule.n_relations,
        embed_dim=config["embed_dim"],
        gamma=config["gamma"],
        lr=config["lr"],
        n_negative_samples=config["n_negative_samples"],
        use_biobridge=config["use_biobridge"]
    )

    logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=CHECKPOINT_DIR,
            filename="primekg-{epoch:02d}-{val/loss:.3f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            verbose=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=5,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow_logger = MLFlowLogger(
        experiment_name=config["experiment_name"],
        run_name=config["run_name"],
        tracking_uri=f"file://{MLFLOW_DIR.resolve()}",
        log_model=True,
    )

    mlflow_logger.log_hyperparams(config)

    logger.info("\n⚡ Configuring PyTorch Lightning Trainer...")
    trainer = L.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu" if __import__("torch").cuda.is_available() else "cpu",
        devices=1,
        precision=config["precision"],
        gradient_clip_val=config["gradient_clip_val"],
        callbacks=callbacks,
        logger=mlflow_logger,
        log_every_n_steps=50,
        val_check_interval=0.5,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    logger.info("\n🏋️  Starting training...")
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=checkpoint_path,
    )

    logger.info("\n🧪 Running PyTorch Lightning Test Loop...")
    trainer.test(
        model,
        datamodule=datamodule,
        ckpt_path="best"
    )

    logger.success(f"\n🎉 Training complete!")
    logger.info(f"   Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    logger.info(f"   Best val loss: {trainer.checkpoint_callback.best_model_score:.4f}")
    logger.info(f"\n   View MLflow UI: mlflow ui --backend-store-uri {MLFLOW_DIR.resolve()}")

    return trainer.checkpoint_callback.best_model_path


if __name__ == "__main__":
    best_model_path = train()
    logger.info(f"\n✅ Best model saved to: {best_model_path}")
