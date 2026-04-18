"""
scripts/train_hpc.py

This script represents Step 4: Large Scale Training on GPU Cluster.
It pairs PyTorch Lightning with WandB to train our HeteroGNN across
potentially multiple compute nodes and multiple GPUs seamlessly.
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from data.biobridge_gnn_datamodule import BioBridgeGNNDataModule
from models.hetero_gnn import BioBridgeLinkPredictor

# Optional: Speed up training if using specific GPU architectures (Ampere +)
torch.set_float32_matmul_precision('medium')

def main():
    # 1. Initialize the DataModule
    # On an HPC cluster, we often have lots of CPU cores. We use 8 workers here.
    datamodule = BioBridgeGNNDataModule(
        data_dir='data/processed/',
        batch_size=2048, # Large batch size suitable for GPUs
        num_workers=8
    )
    
    # We must explicitly call setup to initialize the mock data structure 
    # so we can extract its metadata (node types and edge types).
    datamodule.setup()
    metadata = datamodule.data.metadata()
    target_edge = datamodule.target_edge_type
    
    print(f"Graph Metadata Extracted: {metadata}")

    # 2. Initialize the Model
    model = BioBridgeLinkPredictor(
        metadata=metadata,
        hidden_channels=128, # Dimensionality of message passing embeddings
        lr=0.001,
        target_edge_type=target_edge
    )

    # 3. Setup MLOps Tracking
    # Weight & Biases is standard for HPC because it tracks system metrics (GPU Temp, Memory) 
    # and securely logs across multiple nodes.
    wandb_logger = WandbLogger(
        project="BioBridge-Link-Prediction",
        name="GraphSAGE-HPC-Run",
        log_model="all" # Automatically uploads checkpoints to the cloud
    )

    # 4. Setup Callbacks
    # Save the model automatically on cluster storage
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch:02d}-{val_auroc:.3f}",
        monitor="val_auroc",
        mode="max",
        save_top_k=3,
        save_last=True
    )
    
    # If the model stops improving, kill the job to save cluster compute hours!
    early_stop_callback = EarlyStopping(
        monitor="val_auroc",
        patience=5,
        mode="max"
    )

    # 5. Initialize the Trainer (The magic of PyTorch Lightning for HPC)
    trainer = pl.Trainer(
        # devices=-1 means "use all available GPUs on this node"
        # Since we use SLURM, Lightning automatically detects how many GPUs SLURM assigned us
        devices=-1 if torch.cuda.is_available() else 1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        
        # ddp (Distributed Data Parallel) creates a process for each GPU and synchronizes gradients
        # find_unused_parameters=True is often required when using sample-based subgraphs
        strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto",
        
        num_nodes=int(os.environ.get("SLURM_NNODES", 1)), # Auto-detect multi-node from SLURM
        
        # Mixed Precision (16-bit): Cuts VRAM usage in half, making training 2x faster
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        
        max_epochs=50,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar()],
        
        val_check_interval=0.5 # Run validation twice per epoch
    )

    # 6. TRAIN!
    print("🚀 Starting training on HPC Cluster...")
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
