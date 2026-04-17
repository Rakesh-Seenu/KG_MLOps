# 🧬 BioKG-LinkPredictor

> **GPU-Accelerated Biomedical Disease Link Prediction using PyTorch Lightning, BioBridge & NVIDIA RAPIDS**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch_Lightning-2.x-purple?logo=lightning)](https://lightning.ai)
[![RAPIDS](https://img.shields.io/badge/NVIDIA_RAPIDS-cuGraph-green?logo=nvidia)](https://rapids.ai)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?logo=mlflow)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What This Does

Given a **disease node** in a biomedical knowledge graph, this system predicts which **genes, drugs, and pathways** are most likely to be linked to it — enabling drug repurposing and disease understanding at scale.

Built on the **DRKG (Drug Repurposing Knowledge Graph)** — a real Microsoft Research dataset with **5.8M+ biological relationships** used in COVID-19 drug repurposing research.

---

## 🏗️ Architecture

```
DRKG Dataset (5.8M triples)
        │
        ▼
RAPIDS cuDF/cuGraph  ←── GPU-accelerated preprocessing (10x faster than pandas)
        │
        ▼
BioBridge Encoder  ←── BiomedBERT embeddings per entity (2024 NeurIPS)
        │
        ▼
PyTorch Lightning RotatE  ←── Knowledge Graph Embedding model
        │
        ▼
MLflow Experiment Tracking
        │
        ▼
FastAPI REST Server  ←── POST /predict → link probability scores
        │
        ▼
Docker + Hugging Face Spaces Demo
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Free GPU)
Open the notebook directly:
👉 [**Phase 1: Data + RAPIDS**](notebooks/phase1_data_rapids.ipynb)
👉 [**Phase 2: BioBridge Embeddings**](notebooks/phase2_biobridge.ipynb)
👉 [**Phase 3: PyTorch Lightning Training**](notebooks/phase3_training.ipynb)

### Option 2: Local (RTX GPU)
```bash
# Install Miniconda first, then:
conda env create -f environment.yml
conda activate biokg
make setup
make train
```

---

## 📊 Results

| Model | MRR | Hits@1 | Hits@10 |
|-------|-----|--------|---------|
| TransE (baseline) | - | - | - |
| RotatE + BioBridge | - | - | - |

*(Updated after training)*

---

## 🧠 Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| GPU Data Processing | NVIDIA RAPIDS cuDF/cuGraph | 10-100x faster than pandas on GPU |
| Biomedical Embeddings | BioBridge (BiomedBERT) | SOTA multi-modal biomedical encoder |
| Model Training | PyTorch Lightning 2.x | Clean, scalable DL code |
| Experiment Tracking | MLflow | Reproducibility |
| API Server | FastAPI | High-performance REST API |
| Deployment | Docker + HuggingFace Spaces | Public demo |

---

## 📁 Project Structure

```
KG_MLOps/
├── notebooks/           ← Colab-ready notebooks (start here)
├── data/                ← Data download & preprocessing scripts
├── models/              ← PyTorch Lightning model definitions
├── training/            ← Trainer + MLflow config
├── api/                 ← FastAPI inference server
└── Dockerfile           ← Production container
```

---

*Built to demonstrate GPU-scale ML engineering on real biomedical data.*