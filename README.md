# 🧬 BioBridge-PrimeKG: High-Performance GNN Link Prediction

A production-grade, scalable Link Prediction pipeline for biomedical knowledge graphs (BioBridge/PrimeKG) designed for **SLURM-based HPC GPU clusters** and **Google Colab**.

This repository demonstrates how to architect a Graph Neural Network (GNN) capable of handling **8.1 Million edges** and **129,000 biological entities** (Drugs, Diseases, Proteins) using multimodal precomputed embeddings (ESM-2b, PubMedBERT, SMILES).

---

## 🚀 Key Technical Highlights

- **Heterogeneous Graph Neural Networks**: Implements a GraphSAGE architecture that treats different biological entities as unique mathematical manifolds.
- **NVIDIA RAPIDS Accelerated**: GPU-accelerated preprocessing using `cuDF` to handle massive graph ETL 100x faster than Pandas.
- **Scalable Mini-Batching**: Utilizes `LinkNeighborLoader` for memory-efficient subgraph sampling, preventing Out-of-Memory (OOM) errors on large graphs.
- **HPC Ready**: Fully configured for multi-node, multi-GPU training via PyTorch Lightning with **Distributed Data Parallel (DDP)** and **16-bit Mixed Precision**.
- **MLOps Integration**: Real-time experiment tracking and hardware telemetry (GPU Power, VRAM, TDP) via **Weights & Biases**.

---

## 🛠️ Tech Stack
- **Core**: `PyTorch`, `PyTorch Lightning`, `PyTorch Geometric (PyG)`
- **Acceleration**: `NVIDIA RAPIDS (cuDF)`, `pyg-lib`, `torch-sparse`
- **HPC**: `SLURM`, `DDPStrategy`, `NCCL` backends
- **MLOps**: `WandB`, `Loguru`

---

## 📁 Repository Structure

```tree
├── data/
│   ├── biobridge_gnn_datamodule.py  # Production Graph DataModule
│   ├── download_biobridge.py        # 8GB Dataset Downloader
│   ├── preprocess.py               # GPU-accelerated Graph Builder
├── models/
│   ├── hetero_gnn.py               # Heterogeneous GraphSAGE Architecture
│   ├── biobridge_encoder.py        # Multimodal Embedding Projector (ESM-2b/BERT)
├── notebooks/
│   ├── BioKG_HPC_GNN_Masterclass.ipynb # End-to-end Tutorial
│   ├── evaluation_and_umap.ipynb   # Post-training Latent Space Analysis
├── scripts/
│   ├── slurm_submit.sh              # HPC Multi-GPU Job Submission
│   ├── train_hpc.py                # Main Distributed Training Script
│   ├── setup_env.sh                 # HPC Environment Bootstrapper
```

---

## 🚀 Quickstart (Google Colab)

1. **Clone the Repo**:
   ```bash
   !git clone https://github.com/Rakesh-Seenu/KG_MLOps.git
   %cd KG_MLOps
   ```

2. **Run the Masterclass**:
   Open `notebooks/BioKG_HPC_GNN_Masterclass.ipynb` and follow the steps to train the GNN on real-world medical data.

---

## 🏛️ HPC Deployment (SLURM)

To deploy on a high-performance GPU cluster (e.g., A100/H100 nodes):

1. **Setup Environment**:
   ```bash
   sbatch scripts/setup_env.sh
   ```

2. **Submit Training Job**:
   ```bash
   sbatch scripts/slurm_submit.sh
   ```

---

## 📊 Evaluation & Visualization

After training, use the **UMAP visualization** in `notebooks/evaluation_and_umap.ipynb` to see how the model has clustered biological entities into functional "islands" in the latent space.

#BioAI #GraphNeuralNetworks #DrugDiscovery #HPC #NVIDIARAPIDS #PyTorchLightning
