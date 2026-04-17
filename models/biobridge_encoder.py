"""
models/biobridge_encoder.py
──────────────────────────────────────────────────────────────────────────────
BioBridge Multimodal Encoder for PrimeKG Link Prediction.

Unlike DRKG where we ran BiomedBERT from scratch, BioBridge gives us 
PRECOMPUTED embeddings for nodes using their best modalities:
- Proteins: ESM-2b sequence embeddings (2560-dim)
- Drugs: SMILES structured embeddings (512-dim)
- Diseases/Phenotypes: PubMedBERT text embeddings (768-dim)

This module loads these multimodal embeddings and projects them into a 
unified dimension (e.g., 768) so they can interact in our KGE model.
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from loguru import logger

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "biobridge"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

class BioBridgeProjector(nn.Module):
    """
    Combines pre-computed multimodal embeddings and unifies their dimensions.
    
    If a node is a protein (2560d), it gets projected to `target_dim`.
    If a node is a drug (512d), it gets projected up to `target_dim`.
    Missing nodes get standard trainable embeddings.
    """

    def __init__(self, max_node_index: int, target_dim: int = 768):
        super().__init__()
        self.target_dim = target_dim
        self.max_node_index = max_node_index

        logger.info(f"🧬 Initializing BioBridge Projector (Target Dim: {target_dim})")

        # 1. Base embedding for ALL nodes (trainable fallback for nodes without BioBridge vectors)
        self.base_embeddings = nn.Embedding(max_node_index + 1, target_dim)
        
        # 2. Modality specific linear projections (to align them to target_dim)
        self.protein_proj = nn.Linear(2560, target_dim)
        self.drug_proj = nn.Linear(512, target_dim)
        self.disease_proj = nn.Linear(768, target_dim) # Even if 768->768, a map helps align spaces

        # Track which nodes belong to which modality
        self.register_buffer("protein_mask", torch.zeros(max_node_index + 1, dtype=torch.bool))
        self.register_buffer("drug_mask", torch.zeros(max_node_index + 1, dtype=torch.bool))
        self.register_buffer("disease_mask", torch.zeros(max_node_index + 1, dtype=torch.bool))

        # Modality embedding stores (frozen or finely tuned)
        self.protein_embs = nn.Parameter(torch.zeros(max_node_index + 1, 2560), requires_grad=False)
        self.drug_embs = nn.Parameter(torch.zeros(max_node_index + 1, 512), requires_grad=False)
        self.disease_embs = nn.Parameter(torch.zeros(max_node_index + 1, 768), requires_grad=False)

    def load_pretrained_mappings(self):
        """
        Loads the .pkl dictionaries into memory and populates the embedding buffers.
        Requires that python data/download_biobridge.py has been executed.
        """
        logger.info("Loading BioBridge .pkl embeddings into PyTorch buffers...")

        def _load_dict(filename: str, emb_buffer: nn.Parameter, mask_buffer: torch.Tensor, expected_dim: int):
            path = EMBEDDINGS_DIR / filename
            if not path.exists():
                logger.warning(f"Embedding file {filename} not found.")
                return 0

            with open(path, 'rb') as f:
                data = pickle.load(f)

            count = 0
            for node_idx, vector in data.items():
                if node_idx <= self.max_node_index:
                    if len(vector) == expected_dim:
                        emb_buffer.data[node_idx] = torch.tensor(vector, dtype=torch.float32)
                        mask_buffer[node_idx] = True
                        count += 1
            logger.info(f"   Loaded {count:,} vectors from {filename}")
            return count

        p_count = _load_dict("protein.pkl", self.protein_embs, self.protein_mask, 2560)
        dr_count = _load_dict("drug.pkl", self.drug_embs, self.drug_mask, 512)
        di_count = _load_dict("disease.pkl", self.disease_embs, self.disease_mask, 768)

        total_loaded = p_count + dr_count + di_count
        logger.success(f"✅ Loaded {total_loaded:,} pre-computed BioBridge embeddings!")

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Returns the unified embeddings for the given nodes.
        
        Algorithm:
        1. Look up base trainable embedding.
        2. If node is a protein, add projected BioBridge protein vector.
        3. If node is a drug, add projected BioBridge drug vector.
        4. If node is a disease, add projected BioBridge disease vector.
        """
        out = self.base_embeddings(node_indices)

        # Get modality masks for the requested batch
        p_mask = self.protein_mask[node_indices]
        dr_mask = self.drug_mask[node_indices]
        di_mask = self.disease_mask[node_indices]

        # Apply projections only where masks are true (saves computation)
        if p_mask.any():
            out[p_mask] += self.protein_proj(self.protein_embs[node_indices[p_mask]])
            
        if dr_mask.any():
            out[dr_mask] += self.drug_proj(self.drug_embs[node_indices[dr_mask]])
            
        if di_mask.any():
            out[di_mask] += self.disease_proj(self.disease_embs[node_indices[di_mask]])

        return out

if __name__ == "__main__":
    # Test execution
    dummy_encoder = BioBridgeProjector(max_node_index=130000, target_dim=768)
    dummy_encoder.load_pretrained_mappings()
    
    sample_nodes = torch.tensor([0, 10, 1000])
    embeddings = dummy_encoder(sample_nodes)
    print(f"Sample Embeddings Shape: {embeddings.shape}")
