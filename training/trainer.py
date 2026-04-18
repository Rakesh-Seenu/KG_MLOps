"""
training/trainer.py
──────────────────────────────────────────────────────────────────────────────
Main training script using PyTorch Lightning + MLflow for PrimeKG link prediction.
"""

import json
import yaml
import sys
from pathlib import Path
from typing import Optional, Any, Dict

# ── Path Initialization ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

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

# ── Project Root Discovery ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"

def load_config(config_path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Load and validate the YAML configuration."""
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(config_path)
        
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_callbacks(config: Dict[str, Any], checkpoint_dir: Path) -> list:
    """Initialize standard training callbacks."""
    return [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
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

def train(config_path: Optional[str] = None):
    """
    Main entry point for the PrimeKG Training Pipeline.
    
    This script encapsulates the full MLOps workflow:
    1. Configuration loading & validation.
    2. DataModule initialization (ETL).
    3. Model architecture instantiation.
    4. Experiment tracking integration (MLflow).
    5. GPU-accelerated training via Lightning.
    """
    # 1. Configuration
    raw_config = load_config(Path(config_path) if config_path else DEFAULT_CONFIG)
    
    logger.info("🚀 Starting PrimeKG Elite Training Pipeline")
    logger.info(f"Loaded config: {json.dumps(raw_config, indent=2)}")

    # Paths from config
    paths = raw_config["paths"]
    data_dir = PROJECT_ROOT / paths["data_dir"]
    checkpoint_dir = PROJECT_ROOT / paths["checkpoint_dir"]
    mlflow_dir = PROJECT_ROOT / paths["mlflow_dir"]
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mlflow_dir.mkdir(parents=True, exist_ok=True)

    # 2. Data (ETL)
    train_cfg = raw_config["training"]
    logger.info("📊 Initializing PrimeKG DataModule...")
    datamodule = PrimeKGDataModule(
        data_dir=data_dir,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        train_fraction=train_cfg["train_fraction"],
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    # 3. Model Architecture
    model_cfg = raw_config["model"]
    logger.info("🧠 Building industry-standard RotatE model...")
    model = RotatEModel(
        max_node_idx=datamodule.max_node_index,
        n_relations=datamodule.n_relations,
        embed_dim=model_cfg["embed_dim"],
        gamma=model_cfg["gamma"],
        lr=model_cfg["lr"],
        n_negative_samples=model_cfg["n_negative_samples"],
        use_biobridge=model_cfg["use_biobridge"]
    )

    # 4. MLflow Logging
    mf_cfg = raw_config["mlflow"]
    mlflow_logger = MLFlowLogger(
        experiment_name=mf_cfg["experiment_name"],
        run_name=mf_cfg["run_name"],
        tracking_uri=mlflow_dir.resolve().as_uri(),
        log_model=True,
    )

    # 5. Trainer Orchestration
    callbacks = setup_callbacks(raw_config, checkpoint_dir)
    
    logger.info("⚡ Configuring PyTorch Lightning Trainer...")
    trainer = L.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="gpu" if __import__("torch").cuda.is_available() else "cpu",
        devices=1,
        precision=train_cfg["precision"],
        gradient_clip_val=train_cfg["gradient_clip_val"],
        callbacks=callbacks,
        logger=mlflow_logger,
        log_every_n_steps=train_cfg["log_every_n_steps"],
        val_check_interval=train_cfg["val_check_interval"],
        enable_progress_bar=True,
    )

    # 6. Execution
    logger.info("🏋️  Starting training...")
    trainer.fit(model, datamodule=datamodule)

    # 7. Post-Training Validation
    logger.info("🧪 Running final test suite...")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    logger.success(f"🎉 Training complete! Best model: {trainer.checkpoint_callback.best_model_path}")
    return trainer.checkpoint_callback.best_model_path

if __name__ == "__main__":
    best_model_path = train()
    logger.info(f"\n✅ Best model saved to: {best_model_path}")
