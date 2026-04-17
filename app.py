"""
app.py — Gradio Demo for LinkedIn Showcase
──────────────────────────────────────────────────────────────────────────────
A beautiful interactive demo that runs on Hugging Face Spaces (FREE).
Anyone can use it → this is your LinkedIn demo link.

🎓 WHAT IS GRADIO?
  - Python library to build ML demos in minutes
  - Hugging Face Spaces hosts them for FREE with a public URL
  - Perfect for portfolio/LinkedIn showcases
"""

import json
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
import numpy as np

try:
    from models.kge_model import RotatEModel
    from api.schemas import PredictRequest
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False

# ── Load Resources ────────────────────────────────────────────────────────────
DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")

model = None
entity2id = {}
id2entity = {}
relation2id = {}

def load_model():
    global model, entity2id, id2entity, relation2id
    
    try:
        with open(DATA_DIR / "entity2id.json") as f:
            entity2id = json.load(f)
        id2entity = {v: k for k, v in entity2id.items()}
        
        with open(DATA_DIR / "relation2id.json") as f:
            relation2id = json.load(f)
        
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


def predict(head_entity: str, relation: str, top_k: int) -> str:
    """Run link prediction and format results as a nice HTML table."""
    
    if model is None:
        return "⚠️ Model not loaded yet. Please train the model first (Phase 3 notebook)."
    
    if head_entity not in entity2id:
        return f"❌ Entity '{head_entity}' not found in the knowledge graph."
    
    if relation not in relation2id:
        return f"❌ Relation '{relation}' not found."
    
    head_id = entity2id[head_entity]
    relation_id = relation2id[relation]
    
    with torch.no_grad():
        predictions = model.predict(head_id, relation_id, top_k=top_k)
    
    # Format as markdown table
    rows = ["| Rank | Entity | Type | Score |", "|------|--------|------|-------|"]
    for rank, (entity_id, score) in enumerate(predictions, 1):
        entity_name = id2entity.get(entity_id, f"ID:{entity_id}")
        entity_type = entity_name.split("::")[0] if "::" in entity_name else "Unknown"
        short_name = entity_name.split("::")[-1][:40] if "::" in entity_name else entity_name[:40]
        rows.append(f"| {rank} | {short_name} | {entity_type} | {score:.3f} |")
    
    return "\n".join(rows)


def get_example_entities() -> list[str]:
    """Return some example disease entities."""
    if not entity2id:
        return ["Disease::MESH:D003920", "Disease::MESH:D000544"]
    
    diseases = [e for e in entity2id if e.startswith("Disease::")]
    return diseases[:20] if diseases else list(entity2id.keys())[:20]


def get_example_relations() -> list[str]:
    """Return all available relations."""
    return list(relation2id.keys()) if relation2id else ["GNBR::T::Compound:Disease"]


# ── Gradio UI ─────────────────────────────────────────────────────────────────
load_model()

with gr.Blocks(
    title="🧬 BioKG Disease Link Predictor",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
    css="""
        .header { text-align: center; padding: 20px 0; }
        .prediction-box { font-family: monospace; }
    """,
) as demo:
    
    # Header
    gr.Markdown("""
    <div class="header">
    
    # 🧬 BioKG Disease Link Predictor
    
    **GPU-Accelerated Biomedical Knowledge Graph Link Prediction**
    
    Predicts biological relationships using **RotatE** knowledge graph embeddings 
    trained on the **DRKG** (Drug Repurposing Knowledge Graph) — 5.8M biological triples.
    
    Built with: PyTorch Lightning • BioBridge/BiomedBERT • NVIDIA RAPIDS • MLflow • FastAPI
    
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 🔍 Query the Knowledge Graph")
            
            head_entity = gr.Dropdown(
                choices=get_example_entities(),
                label="Head Entity (Disease)",
                info="Select a disease to find its linked compounds, genes, or pathways",
                value=get_example_entities()[0] if get_example_entities() else None,
            )
            
            relation = gr.Dropdown(
                choices=get_example_relations(),
                label="Relation Type",
                info="The type of biological relationship to predict",
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
    
    # Stats row
    with gr.Row():
        gr.Markdown("""
        ### 📈 Model Architecture
        | Component | Detail |
        |-----------|--------|
        | **KGE Model** | RotatE (Sun et al., ICLR 2019) |
        | **Entity Encoder** | BioBridge/BiomedBERT-base |
        | **Training Framework** | PyTorch Lightning 2.x |
        | **GPU Preprocessing** | NVIDIA RAPIDS cuDF/cuGraph |
        | **Dataset** | DRKG: 97K entities, 5.8M triples |
        | **Experiment Tracking** | MLflow |
        """)
        
        gr.Markdown("""
        ### 🧬 About DRKG
        The **Drug Repurposing Knowledge Graph** connects:
        - 🧬 **Genes** — Human gene-protein interactions
        - 💊 **Compounds** — FDA-approved and experimental drugs  
        - 🦠 **Diseases** — MESH-coded disease nodes
        - 🔬 **Pathways** — Biological process pathways
        - ⚗️ **Pharmacological Classes** — Drug mechanism groups
        
        Originally used in COVID-19 **drug repurposing research** by Microsoft Research.
        """)
    
    predict_btn.click(
        fn=predict,
        inputs=[head_entity, relation, top_k],
        outputs=output,
    )
    
    gr.Markdown("""
    ---
    *Built by Rakesh Hadne Sreenath | 
    [GitHub](https://github.com/rakesh-hs) | 
    [LinkedIn](https://linkedin.com/in/rakesh-hs)*
    """)


if __name__ == "__main__":
    demo.launch(share=True)
