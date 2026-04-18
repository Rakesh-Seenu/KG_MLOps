import json
import yaml
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

# ── Path Initialization ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import gradio as gr
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from models.kge_model import RotatEModel

# ── Resource Discovery ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

def load_config():
    if not CONFIG_PATH.exists():
        # Fallback for initialization
        return {
            "paths": {"raw_dir": "data/raw/primekg", "data_dir": "data/processed", "checkpoint_dir": "checkpoints"},
            "mlflow": {"experiment_name": "BioKG"},
            "app": {"title": "BioKG Elite"}
        }
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()
DATA_DIR = PROJECT_ROOT / CONFIG["paths"]["raw_dir"]
PROC_DIR = PROJECT_ROOT / CONFIG["paths"]["data_dir"]
CHECKPOINT_DIR = PROJECT_ROOT / CONFIG["paths"]["checkpoint_dir"]

# ── Persistent State & Cache ──────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.model: Optional[RotatEModel] = None
        self.relation2id: Dict[str, int] = {}
        self.id2relation: Dict[int, str] = {}
        self.node_metadata: Dict[int, Dict[str, str]] = {}
        self.is_loaded: bool = False

    def load_resources(self) -> str:
        """Centralized resource loading with robust error handling."""
        try:
            # 1. Load Mappings
            rel_path = PROC_DIR / "relation2id.json"
            if not rel_path.exists():
                return "❌ Processed data missing. Please run preprocessing (`python data/preprocess.py`) first."
            
            with open(rel_path) as f:
                self.relation2id = json.load(f)
            self.id2relation = {v: k for k, v in self.relation2id.items()}

            # 2. Load Node Metadata for UI Names
            nodes_path = DATA_DIR / "kg.csv"
            if nodes_path.exists():
                logger.info("⏳ Loading 129K node names for mapping...")
                df = pd.read_csv(nodes_path, low_memory=False, usecols=['x_index', 'x_type', 'x_name'])
                df = df.rename(columns={'x_index': 'idx', 'x_type': 'type', 'x_name': 'name'}).drop_duplicates('idx')
                self.node_metadata = df.set_index('idx').to_dict('index')
                logger.success(f"✅ Indexed {len(self.node_metadata):,} biological entities.")
            
            # 3. Load Model Checkpoint
            checkpoints = list(CHECKPOINT_DIR.glob("*.ckpt"))
            if not checkpoints:
                return "⚠️ No model checkpoints found in /checkpoints. System standby."
            
            best_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"🧠 Loading Elite RotatE Model from {best_ckpt.name}...")
            self.model = RotatEModel.load_from_checkpoint(best_ckpt, map_location=device)
            self.model.eval()
            self.is_loaded = True
            
            return f"🟢 System Online: Loaded checkpoint {best_ckpt.name}"
        except Exception as e:
            logger.error(f"Boot failed: {e}")
            return f"🔴 Boot Error: {str(e)}"

# Initial singleton
app_state = AppState()

# ── Inference & Analytics Logic ───────────────────────────────────────────────
def run_prediction(head_index: int, relation_name: str, top_k: int) -> Tuple[str, Any]:
    """Execute prediction and generate analytics visualization."""
    if not app_state.is_loaded:
        # Attempt one-time load if not loaded
        msg = app_state.load_resources()
        if not app_state.is_loaded:
            return f'<div class="error-msg">⚠️ Model offline. {msg}</div>', None
        
    if relation_name not in app_state.relation2id:
        return f'<div class="error-msg">❌ Relation \'{relation_name}\' not found in KG registry.</div>', None

    rid = app_state.relation2id[relation_name]
    
    try:
        # Run Vectorized Inference
        predictions = app_state.model.predict(int(head_index), rid, top_k=top_k)
        
        # Build Summary Table
        html = '<div class="table-container"><table class="styled-table">'
        html += '<thead><tr><th>Rank</th><th>Node</th><th>Type</th><th>Confidence Score</th></tr></thead><tbody>'
        
        plot_data = []
        for rank, (eid, score) in enumerate(predictions, 1):
            meta = app_state.node_metadata.get(eid, {"name": f"Node_{eid}", "type": "Other"})
            name, etype = meta['name'], meta['type']
            
            badge_class = "badge-default"
            if "drug" in etype.lower(): badge_class = "badge-drug"
            elif "disease" in etype.lower(): badge_class = "badge-disease"
            elif "protein" in etype.lower() or "gene" in etype.lower(): badge_class = "badge-gene"
            
            html += f'<tr><td><span class="rank-circle">#{rank}</span></td><td class="node-name">{name} <br><small class="node-idx">IDX: {eid}</small></td>'
            html += f'<td><span class="badge {badge_class}">{etype}</span></td><td><div class="score-bar-bg"><div class="score-bar-fg" style="width: {min(100, max(0, (score+14)*6))}%"></div></div> {score:.4f}</td></tr>'
            
            plot_data.append({"Node": name, "Score": float(score), "Type": etype})

        html += '</tbody></table></div>'
        
        # Build Visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        df_plot = pd.DataFrame(plot_data)
        sns.barplot(data=df_plot, x="Score", y="Node", hue="Type", palette="viridis", ax=ax)
        ax.set_title("Top-K Prediction Confidence Distribution", color="white", fontsize=14)
        
        # Style for Dark Mode
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        plt.tight_layout()
        
        return html, fig
    except Exception as e:
        logger.error(f"Inference crash: {e}")
        return f'<div class="error-msg">❌ Inference pipeline failure: {str(e)}</div>', None

# ── Gradio Theme & CSS ────────────────────────────────────────────────────────
custom_theme = gr.themes.Default(
    primary_hue="fuchsia",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"],
).set(
    body_background_fill="transparent",
    block_background_fill="rgba(20, 25, 40, 0.7)",
    block_border_width="1px",
    block_border_color="rgba(255,255,255,0.08)",
    button_primary_background_fill="linear-gradient(90deg, #6366f1, #d946ef)",
    button_primary_text_color="white",
)

custom_css = """
.gradio-container {
    background: radial-gradient(circle at top right, #1e1b4b, #0f172a, #020617) !important;
    color: #f8fafc !important;
}
.glass-panel {
    background: rgba(30, 41, 59, 0.4) !important;
    backdrop-filter: blur(16px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 25px !important;
    box-shadow: 0 20px 50px rgba(0,0,0,0.5) !important;
}
.grad-text {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
}
.styled-table { width: 100%; border-collapse: collapse; margin: 10px 0; }
.styled-table thead tr { background: rgba(255,255,255,0.05); color: #94a3b8; text-transform: uppercase; font-size: 0.75rem; }
.styled-table th, .styled-table td { padding: 14px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.05); }
.node-name { font-weight: 700; color: #fff; }
.node-idx { color: #64748b; font-size: 0.8em; }
.rank-circle { background: rgba(255,255,255,0.1); padding: 4px 10px; border-radius: 50%; border: 1px solid rgba(255,255,255,0.2); font-size: 0.8em; }
.score-bar-bg { background: rgba(255,255,255,0.05); height: 6px; border-radius: 3px; width: 100px; display: inline-block; margin-right: 10px; }
.score-bar-fg { background: linear-gradient(90deg, #4f46e5, #0ea5e9); height: 6px; border-radius: 3px; }
.badge { padding: 4px 12px; border-radius: 999px; font-size: 0.7rem; font-weight: 800; text-transform: uppercase; border: 1px solid transparent; }
.badge-drug { background: rgba(14, 165, 233, 0.15); color: #0ea5e9; border-color: rgba(14, 165, 233, 0.3); }
.badge-disease { background: rgba(244, 63, 94, 0.15); color: #f43f5e; border-color: rgba(244, 63, 94, 0.3); }
.badge-gene { background: rgba(34, 197, 94, 0.15); color: #22c55e; border-color: rgba(34, 197, 94, 0.3); }
.error-msg { background: rgba(244, 63, 94, 0.1); color: #fda4af; padding: 20px; border-radius: 12px; border-left: 5px solid #f43f5e; }
"""

# ── UI Construction ───────────────────────────────────────────────────────────
def get_ui():
    with gr.Blocks(title="BioKG Elite Dashboard") as demo:
        
        with gr.Row():
            with gr.Column(scale=4):
                gr.HTML("""
                    <div style="padding: 20px 0;">
                        <h1 class="grad-text" style="font-size: 3rem; margin-bottom: 0;">🧬 BioKG Elite Dashboard</h1>
                        <p style="color: #94a3b8; font-size: 1.2rem; margin-top: 5px;">PrimeKG Multimodal Knowledge Graph Link Prediction</p>
                    </div>
                """)
            with gr.Column(scale=1):
                status_indicator = gr.HTML(value="System initializing...")
                reboot_btn = gr.Button("🔄 Reboot Core", size="sm")

        with gr.Row():
            with gr.Column(scale=2, elem_classes=["glass-panel"]):
                gr.Markdown("### 🔍 Knowledge Graph Query")
                
                head_idx = gr.Number(label="Head Node Index", value=9796, info="PrimeKG index for the subject entity (e.g. 9796 for BRCA1)")
                
                rel_choices = list(app_state.relation2id.keys()) if app_state.relation2id else ["indication", "synergistic_interaction"]
                rel_dropdown = gr.Dropdown(
                    choices=rel_choices,
                    label="Biomedical Relation",
                    value=rel_choices[0] if rel_choices else None,
                    allow_custom_value=True
                )
                
                top_k_slider = gr.Slider(minimum=5, maximum=100, step=5, value=10, label="Prediction Depth (Top-K)")
                
                predict_btn = gr.Button("🚀 Execute Real-time Prediction", variant="primary")
                
                gr.Markdown("---")
                gr.Markdown("#### 📑 Model Status")
                gr.JSON(value={
                    "Model": "RotatE (Complex Sp.)",
                    "Encoders": "BioBridge/ESM2/MM",
                    "Backend": "PyTorch Lightning",
                    "Precision": "16-Mixed"
                })

            with gr.Column(scale=3, elem_classes=["glass-panel"]):
                gr.Markdown("### 📈 Prediction Analytics")
                
                with gr.Tabs():
                    with gr.TabItem("Ranked Results"):
                        output_html = gr.HTML(value='<div style="text-align:center; padding:100px; color:#64748b;">Waiting for query execution...</div>')
                    with gr.TabItem("Confidence Distribution"):
                        output_plot = gr.Plot()

        with gr.Row(elem_classes=["glass-panel"]):
            gr.HTML("""
                <div style="display: flex; gap: 40px; padding: 10px;">
                    <div style="flex: 1;">
                        <h4 style="color: #c084fc;">📡 PrimeKG Infrastructure</h4>
                        <p style="color: #94a3b8; font-size: 0.9rem;">Integrated with Harvard Dataverse PrimeKG dataset featuring 129,375 nodes and 8,100,225 precision medicine edges across 20 primary databases.</p>
                    </div>
                    <div style="flex: 1;">
                        <h4 style="color: #38bdf8;">🧠 BioBridge Foundation</h4>
                        <p style="color: #94a3b8; font-size: 0.9rem;">Utilizing cross-modal semantic projectors that map protein ESM2, drug SMILES, and phenotype PubMedBERT spaces into the unified RotatE complex dimensions.</p>
                    </div>
                </div>
            """)

        # ── Interactivity ─────────────────────────────────────────────────────────
        def reboot_system():
            return app_state.load_resources()

        reboot_btn.click(fn=reboot_system, outputs=status_indicator)
        
        predict_btn.click(
            fn=run_prediction, 
            inputs=[head_idx, rel_dropdown, top_k_slider], 
            outputs=[output_html, output_plot]
        )
        
        # Initial status seed
        demo.load(fn=reboot_system, outputs=status_indicator)

    return demo

if __name__ == "__main__":
    # Pre-load resources if possible
    app_state.load_resources()
    
    # Launch with Gradio 6 style
    demo = get_ui()
    demo.launch(share=True, theme=custom_theme, css=custom_css)
