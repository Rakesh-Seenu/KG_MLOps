"""
app.py — Gradio Demo for LinkedIn Showcase
──────────────────────────────────────────────────────────────────────────────
A beautiful interactive demo that runs on Hugging Face Spaces (FREE).
Now upgraded to PrimeKG + BioBridge!
"""

import json
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
import numpy as np
import pandas as pd

try:
    from models.kge_model import RotatEModel
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False

# ── Load Resources ────────────────────────────────────────────────────────────
DATA_DIR = Path("data/raw/primekg")
PROC_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")

model = None
relation2id = {}
id2relation = {}
node_metadata = {}  # index -> dict(type, name)

def load_data():
    global model, relation2id, id2relation, node_metadata
    
    try:
        with open(PROC_DIR / "relation2id.json") as f:
            relation2id = json.load(f)
        id2relation = {v: k for k, v in relation2id.items()}

        # Load node mappings to make the UI human readable if present
        nodes_path = DATA_DIR / "kg.csv"
        if nodes_path.exists():
            df = pd.read_csv(nodes_path, low_memory=False, usecols=['x_index', 'x_type', 'x_name'])
            df = df.rename(columns={'x_index': 'node_index', 'x_type': 'node_type', 'x_name': 'node_name'}).drop_duplicates()
            node_metadata = df.set_index('node_index').to_dict('index')
            print(f"Loaded {len(node_metadata)} nodes for UI naming.")
        
        checkpoints = list(CHECKPOINT_DIR.glob("*.ckpt"))
        if checkpoints and MODEL_AVAILABLE:
            best_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = RotatEModel.load_from_checkpoint(best_ckpt, map_location=device)
            model.eval()
            return True
    except Exception as e:
        print(f"Could not load model: {e}")
    return False


def predict(head_index: int, relation: str, top_k: int) -> str:
    """Run link prediction and format results as a nice HTML table."""
    
    if model is None:
        return "⚠️ Model not loaded yet. Please train the model first."
        
    if relation not in relation2id:
        return f"❌ Relation '{relation}' not found."
    
    relation_id = relation2id[relation]
    
    with torch.no_grad():
        predictions = model.predict(int(head_index), relation_id, top_k=top_k)
    
    rows = ["| Rank | Node Index | Name | Type | Score |", "|------|------------|------|------|-------|"]
    for rank, (entity_id, score) in enumerate(predictions, 1):
        meta = node_metadata.get(entity_id, {"node_name": "Unknown", "node_type": "Unknown"})
        name = meta['node_name']
        e_type = meta['node_type']
        rows.append(f"| {rank} | {entity_id} | {name} | {e_type} | {score:.3f} |")
    
    return "\n".join(rows)


def get_example_nodes():
    return [
        ("BRCA1 (Gene / Protein)", 9796),
        ("Alzheimer's Disease (Disease)", 12345), # Examples, the exact IDs depend on dataset
        ("Aspirin (Drug)", 500)
    ]


def get_example_relations() -> list[str]:
    return list(relation2id.keys()) if relation2id else ["indication", "synergistic_interaction"]


# ── Gradio UI ─────────────────────────────────────────────────────────────────
load_data()

with gr.Blocks(
    title="🧬 BioKG PrimeKG Link Predictor",
    theme=gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
    css="""
        .header { text-align: center; padding: 20px 0; }
        .prediction-box { font-family: monospace; }
    """,
) as demo:
    
    gr.Markdown("""
    <div class="header">
    
    # 🧬 BioKG PrimeKG Link Predictor
    
    **GPU-Accelerated Biomedical Knowledge Graph Link Prediction**
    
    Predicts biological relationships across over 129,000 multimodal nodes using **RotatE** knowledge graph embeddings. 
    Uses **BioBridge** integrations mapping ESM2, SMILES, and PubMedBERT spaces directly to RotatE complex dimensions!
    
    Built with: PyTorch Lightning • BioBridge Multimodality • NVIDIA cuDF • MLflow • FastAPI
    
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 🔍 Query the Prime Knowledge Graph")
            
            head_index = gr.Number(
                label="Head Node Index (Integer)",
                value=42
            )
            
            relation = gr.Dropdown(
                choices=get_example_relations(),
                label="Relation Type",
                value=get_example_relations()[0] if get_example_relations() else None,
            )
            
            top_k = gr.Slider(
                minimum=5,
                maximum=50,
                value=10,
                step=5,
                label="Number of predictions (Top-K)",
            )
            
            predict_btn = gr.Button("🚀 Predict Links", variant="primary", size="lg")
        
        with gr.Column(scale=3):
            gr.Markdown("### 📊 Predictions")
            output = gr.Markdown(
                value="*Results will appear here after prediction...*",
                label="Link Predictions",
            )
    
    with gr.Row():
        gr.Markdown("""
        ### 📈 Model Architecture
        | Component | Detail |
        |-----------|--------|
        | **KGE Model** | RotatE (Sun et al., ICLR 2019) |
        | **Entity Encoders** | BioBridge / ESM2 / PubMedBERT / SMILES |
        | **Training** | PyTorch Lightning 2.x Precision-16 mixed |
        | **Dataset** | PrimeKG: 129K Entities, 8.1M Triples |
        """)
        
        gr.Markdown("""
        ### 🧬 About PrimeKG
        PrimeKG is a massive precision medicine knowledge graph harmonized from 20 databases:
        - 🧬 **Proteins/Genes** — Human interactions 
        - 💊 **Drugs** — Experimental and FDA-approved compounds
        - 🦠 **Diseases/Phenotypes** — Medical presentations
        
        Unlike DRKG, it provides incredibly rich meta-data and separates perfectly into biological multimodal foundations.
        """)
    
    predict_btn.click(
        fn=predict,
        inputs=[head_index, relation, top_k],
        outputs=output,
    )
    
if __name__ == "__main__":
    demo.launch(share=True)
