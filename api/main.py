"""
api/main.py
──────────────────────────────────────────────────────────────────────────────
FastAPI inference server for the trained PrimeKG RotatE model.
"""

import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.schemas import PredictRequest, PredictResponse, PredictionItem, HealthResponse
from models.kge_model import RotatEModel

# ── Paths  ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


# ── App State  ────────────────────────────────────────────────────────────────
class AppState:
    model: Optional[RotatEModel] = None
    relation2id: Optional[dict] = None
    id2relation: Optional[dict] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_node_index: int = 0


state = AppState()


# ── Lifespan  ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting BioKG PrimeKG Inference Server...")

    with open(DATA_DIR / "relation2id.json") as f:
        state.relation2id = json.load(f)
    state.id2relation = {v: k for k, v in state.relation2id.items()}

    with open(DATA_DIR / "stats.json") as f:
        stats = json.load(f)
        state.max_node_index = stats["max_node_index"]

    checkpoints = list(CHECKPOINT_DIR.glob("*.ckpt"))
    if not checkpoints:
        logger.warning("⚠️  No checkpoint found. Server running in API-only mode without ML capability.")
    else:
        best_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
        logger.info(f"🧠 Loading RotatE model from: {best_ckpt.name}")

        state.model = RotatEModel.load_from_checkpoint(
            best_ckpt,
            map_location=state.device,
        )
        state.model.eval()
        state.model = state.model.to(state.device)
        logger.success(f"✅ PrimeKG Model loaded on {state.device.upper()}")

    logger.success("🌐 Server ready! Visit http://localhost:8000/docs for API.")
    yield
    logger.info("🛑 Shutting down server...")
    state.model = None


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="BioKG-PrimeKG Disease Link Predictor",
    description="""
## 🧬 PrimeKG Knowledge Graph Link Prediction API

Predicts interactions across 129,000 biological nodes spanning Proteins, Diseases, and Drugs.
Powered by a Multimodal BioBridge RotatE framework.
""",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=state.model is not None,
        device=state.device,
        n_nodes=state.max_node_index + 1,
        n_relations=len(state.relation2id) if state.relation2id else 0,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_links(request: PredictRequest):
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train RotatE via Trainer first."
        )

    if request.head_index < 0 or request.head_index > state.max_node_index:
        raise HTTPException(
            status_code=404,
            detail=f"Node Index {request.head_index} out of bounds (0-{state.max_node_index})"
        )

    if request.relation not in state.relation2id:
        raise HTTPException(
            status_code=404,
            detail=f"Relation '{request.relation}' not found. Check /relations for available edge types."
        )

    # Convert to integer IDs
    head_id = request.head_index
    relation_id = state.relation2id[request.relation]

    # Run inference
    with torch.no_grad():
        predictions = state.model.predict(
            head_id=head_id,
            relation_id=relation_id,
            top_k=request.top_k,
        )

    result_items = [
        PredictionItem(
            node_index=entity_id,
            score=float(score),
            rank=rank + 1,
        )
        for rank, (entity_id, score) in enumerate(predictions)
    ]

    return PredictResponse(
        head_index=request.head_index,
        relation=request.relation,
        predictions=result_items,
    )


@app.get("/relations", tags=["Discovery"])
async def list_relations():
    if state.relation2id is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    return {
        "total_relations": len(state.relation2id),
        "relations": list(state.relation2id.keys()),
    }


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
