# BioKG-LinkPredictor: Disease Link Prediction with PyTorch Lightning, BioBridge & RAPIDS

## Goal

Build a production-grade, GPU-accelerated **biomedical knowledge graph disease link prediction system** that demonstrates skills in:
- Deep Learning (PyTorch Lightning)
- Biomedical NLP (BioBridge / BiomedBERT embeddings)
- GPU-accelerated graph analytics (NVIDIA RAPIDS / cuGraph)
- ML Engineering (MLflow, model serving via FastAPI)
- Cloud deployment (Docker → Hugging Face Spaces or AWS/GCP)

This is designed so you **learn every concept as you build it**, with clear explanations at each step.

---

## Why This Project Is Impressive

> "Applying DL methods on **large datasets on GPU clusters**" — Gurdeep's exact criteria for RA candidates.

This project directly addresses that requirement:
- Processes the **DRKG (Drug Repurposing Knowledge Graph)** — 5.8M+ triples, real biomedical data
- Uses **GPU-accelerated preprocessing** (RAPIDS cuGraph / cuDF) — not just "used GPU for training"
- Uses **BioBridge** (SOTA biomedical KG bridge model, 2024 NeurIPS paper)
- Deploys a live **REST API** others can query — shows MLOps awareness
- Full experiment tracking with **MLflow** — what industry expects

---

## Architecture Overview

```
Raw Biomedical Data (DRKG/PrimeKG)
        │
        ▼
[RAPIDS cuDF]  ──── GPU-accelerated graph preprocessing
        │              Entity/relation statistics, subgraph sampling
        ▼
[BioBridge Encoder] ── Pre-trained BiomedBERT embeddings per entity
        │              Bridges text ↔ graph modalities
        ▼
[PyTorch Lightning KGE Model]
        │   ┌── TransE / RotatE / ComplEx (you pick)
        │   └── Trained with negative sampling + margin ranking loss
        ▼
[MLflow Experiment Tracker] ── Log metrics, model versions
        ▼
[FastAPI Inference Server] ── REST endpoint: POST /predict
        │                       Input: entity pairs
        │                       Output: link probability scores
        ▼
[Docker Container] ──── Deployed to Hugging Face Spaces / Railway / AWS EC2
        │
        ▼
[Gradio UI (optional)] ── Interactive demo for LinkedIn post
```

---

## Phase-by-Phase Breakdown

### Phase 1: Environment Setup & Data Pipeline (Week 1)
**What you'll learn:** RAPIDS cuDF/cuGraph, biomedical data formats, graph statistics

#### Files to create:
- `environment.yml` — conda env with RAPIDS, PyTorch, Lightning
- `data/download_drkg.py` — downloads DRKG dataset (~2GB TSV file)
- `data/preprocess.py` — **RAPIDS cuDF** for fast GPU-accelerated preprocessing:
  - Entity/relation frequency analysis
  - Train/val/test splitting (80/10/10)
  - Subgraph sampling for tractable training
- `data/graph_stats.py` — **cuGraph** for graph analytics:
  - Degree distribution
  - Connected components
  - PageRank of disease nodes

#### Key RAPIDS concepts you'll learn:
- `cudf.read_csv()` — 10-100x faster than pandas on GPU
- `cugraph.Graph()` — GPU graph construction
- `cugraph.pagerank()` — GPU-accelerated graph algorithms

---

### Phase 2: BioBridge Embeddings (Week 1-2)
**What you'll learn:** Biomedical LLMs, multi-modal KG embeddings, transfer learning

#### Files to create:
- `models/biobridge_encoder.py` — Wraps the BioBridge pre-trained model:
  - Loads `QSong5/BioBridge` from HuggingFace
  - Encodes entity names → 768-dim vectors using BiomedBERT
  - Caches embeddings to disk (avoid re-computing)
- `data/entity_mapper.py` — Maps DRKG entity IDs → text names for encoding

#### Key concepts you'll learn:
- What BioBridge does: bridges **protein sequences / SMILES / text** modalities into a shared embedding space
- Why it matters: pre-trained on massive biomedical corpora → transfer learning
- How to use HuggingFace `pipeline` and `AutoModel`

---

### Phase 3: PyTorch Lightning KGE Model (Week 2)
**What you'll learn:** Knowledge Graph Embeddings, PyTorch Lightning structure, GPU training

#### Files to create:
- `models/kge_model.py` — **PyTorch Lightning Module** implementing:
  - `RotatE` scoring function: `s ◦ r ≈ t` in complex space
  - Self-supervised negative sampling (corrupt head or tail)
  - Margin ranking loss
  - Metrics: MRR (Mean Reciprocal Rank), Hits@1, Hits@10
- `training/trainer.py` — PyTorch Lightning **Trainer** config:
  - Multi-GPU support (`strategy="ddp"`)
  - Gradient checkpointing for large graphs
  - Early stopping + LR scheduling
  - MLflow logger integration

#### PyTorch Lightning concepts you'll learn:
- `LightningModule`: clean separation of model, loss, metrics
- `LightningDataModule`: encapsulates data loading
- `Trainer` flags: `accelerator="gpu"`, `precision="16-mixed"` (BF16)
- Callbacks: `ModelCheckpoint`, `EarlyStopping`, `LearningRateMonitor`

---

### Phase 4: MLflow Experiment Tracking (Week 2-3)
**What you'll learn:** MLOps fundamentals, experiment management

#### Files to create:
- `training/mlflow_config.py` — MLflow setup
- Automatic logging of:
  - Hyperparameters (embedding dim, learning rate, batch size)
  - Metrics per epoch (MRR, Hits@10, loss)
  - Best model artifact registration
- `Makefile` — `make train`, `make evaluate`, `make serve`

---

### Phase 5: FastAPI Inference Server (Week 3)
**What you'll learn:** REST API design, model serving, async Python

#### Files to create:
- `api/main.py` — FastAPI app with:
  - `POST /predict` — given entity pair, returns link probabilities
  - `GET /entities` — list all supported entity types
  - `GET /health` — liveness probe for deployment
- `api/inference.py` — Loads trained model checkpoint, runs inference
- `api/schemas.py` — Pydantic input/output schemas

---

### Phase 6: Docker & Deployment (Week 3-4)
**What you'll learn:** Containerization, deployment, making your work publicly accessible

#### Files to create:
- `Dockerfile` — Multi-stage: training image (with CUDA) + inference image (slim)
- `docker-compose.yml` — Local stack: API + MLflow server
- `app.py` — **Gradio demo UI** deployed to Hugging Face Spaces:
  - Type a disease name → get top 10 predicted drug/gene/pathway links
  - Shows prediction scores + confidence
  - **This is your LinkedIn demo link**

---

### Phase 7: LinkedIn Showcase (Week 4)
**What you'll learn:** Technical communication (critical for PhD/industry)

#### Deliverables:
- GitHub README with architecture diagram (auto-generated with matplotlib)
- Benchmark table: RAPIDS vs Pandas preprocessing speed
- Training curve plots
- Live Gradio demo link
- LinkedIn post draft (I'll write this for you)

---

## Project Structure

```
KG_MLOps/
├── README.md                    ← Showcase document
├── environment.yml              ← RAPIDS + PyTorch + Lightning env
├── Makefile                     ← make setup / train / serve
├── Dockerfile
├── docker-compose.yml
├── app.py                       ← Gradio demo (HuggingFace Spaces)
│
├── data/
│   ├── download_drkg.py         ← Download DRKG
│   ├── preprocess.py            ← RAPIDS cuDF preprocessing
│   ├── graph_stats.py           ← cuGraph analytics
│   ├── entity_mapper.py         ← Entity ID → text name
│   └── datamodule.py            ← LightningDataModule
│
├── models/
│   ├── biobridge_encoder.py     ← BioBridge/BiomedBERT encoder
│   └── kge_model.py             ← RotatE LightningModule
│
├── training/
│   ├── trainer.py               ← Lightning Trainer + MLflow
│   ├── mlflow_config.py         ← MLflow setup
│   └── evaluate.py              ← MRR / Hits@K evaluation
│
├── api/
│   ├── main.py                  ← FastAPI server
│   ├── inference.py             ← Model loading + scoring
│   └── schemas.py               ← Pydantic schemas
│
└── notebooks/
    ├── 01_data_exploration.ipynb   ← Learn DRKG structure
    ├── 02_rapids_demo.ipynb        ← RAPIDS speed comparison
    └── 03_training_analysis.ipynb  ← Training curve analysis
```

---

## Learning Curriculum (What I'll Teach You Step by Step)

| Week | Topic | Tool | Concept |
|------|-------|------|---------|
| 1 | Graph data science | RAPIDS cuDF/cuGraph | GPU DataFrames, graph algorithms |
| 1 | Biomedical NLP | BioBridge/BiomedBERT | Transfer learning, embeddings |
| 2 | KG embeddings | PyTorch + Lightning | RotatE, negative sampling, MRR |
| 2 | Experiment tracking | MLflow | Reproducibility, model registry |
| 3 | Model serving | FastAPI | REST APIs, async, Pydantic |
| 3 | Containerization | Docker | Images, volumes, compose |
| 4 | Deployment | HuggingFace Spaces | Public demo, CI/CD |
| 4 | Communication | GitHub + LinkedIn | Technical storytelling |

---

## Hardware Requirements

| Scenario | Minimum | Recommended |
|----------|---------|-------------|
| Local dev | CPU only (slower) | NVIDIA GPU (GTX 1060+) |
| RAPIDS | NVIDIA GPU (CUDA 11.2+) | RTX 3080+ |
| Full training | 8GB VRAM | 16GB+ VRAM |
| Free alternative | Google Colab Pro | Kaggle (free T4 GPU) |

> **If you don't have a GPU locally**: We'll use **Google Colab** for RAPIDS and training, and a **free Hugging Face Space** (CPU) for the inference API demo. This is perfectly fine and still impressive.

---

## Dataset: DRKG (Drug Repurposing Knowledge Graph)
- **Source**: Microsoft Research (open source)
- **Size**: 5,874,261 triples, 97,238 entities, 107 relation types
- **Entities**: Genes, Compounds, Diseases, Proteins, Pathways, Side Effects
- **Why impressive**: Real-world biomedical data used in COVID-19 drug repurposing research

---

## Verification Plan

1. ✅ RAPIDS preprocessing runs 10x+ faster than pandas (benchmarked)
2. ✅ Model trains and achieves MRR > 0.15 on DRKG test set
3. ✅ FastAPI server responds to `/predict` within 200ms
4. ✅ Docker container builds and runs locally
5. ✅ Gradio demo is publicly accessible via HuggingFace Spaces link
6. ✅ GitHub repo has README with results, architecture diagram, and demo link

---

## Open Questions for You

1. **Do you have an NVIDIA GPU locally?** (This affects RAPIDS setup — if not, we use Colab)
2. **Which week do you want to start?** I recommend we go phase by phase, one phase per session.
3. **Do you want to use `conda` or `pip` + `venv`?** (conda is strongly recommended for RAPIDS)

