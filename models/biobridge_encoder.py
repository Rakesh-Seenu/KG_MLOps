"""
models/biobridge_encoder.py
──────────────────────────────────────────────────────────────────────────────
BioBridge Multimodal Encoder for PrimeKG Link Prediction.
Aligns precomputed ESM-2b, PubMedBERT, and SMILES embeddings with the GNN.
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from loguru import logger
from torch_geometric.data import HeteroData

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "biobridge"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

class BioBridgeProjector(nn.Module):
    """
    Projector that maps BioBridge multimodal embeddings into the GNN latent space.
    """
    def __init__(self, target_dim: int = 768):
        super().__init__()
        self.target_dim = target_dim

        # Projections to unify dimensions
        self.gene_proj = nn.Linear(2560, target_dim)    # ESM-2b
        self.drug_proj = nn.Linear(512, target_dim)     # SMILES
        self.disease_proj = nn.Linear(768, target_dim)  # PubMedBERT

        # Fallback embeddings for nodes we don't have PKL data for
        self.fallbacks = nn.ModuleDict()

        # Mapping: global_id -> local_id for each type
        self.global_to_local = {} 

        self.initialized = False

    def initialize_from_data(self, data: HeteroData):
        """
        Initializes the embedding buffers based on the real nodes in the HeteroData object.
        """
        logger.info("Initializing BioBridge buffers from HeteroData...")
        
        self.gene_embs = nn.Parameter(torch.zeros(data['gene'].num_nodes, 2560), requires_grad=False)
        self.drug_embs = nn.Parameter(torch.zeros(data['drug'].num_nodes, 512), requires_grad=False)
        self.disease_embs = nn.Parameter(torch.zeros(data['disease'].num_nodes, 768), requires_grad=False)
        
        # Initialize fallbacks
        for node_type in ['gene', 'drug', 'disease']:
            self.fallbacks[node_type] = nn.Embedding(data[node_type].num_nodes, self.target_dim)

        self.initialized = True

    def load_pretrained_mappings(self, data: HeteroData):
        """
        Loads the .pkl files and maps them to the local indices of the HeteroData object.
        """
        if not self.initialized:
            self.initialize_from_data(data)

        logger.info("Loading BioBridge .pkl embeddings and aligning with Local IDs...")

        def _align_modality(filename: str, node_type: str, emb_buffer: nn.Parameter, expected_dim: int):
            path = EMBEDDINGS_DIR / filename
            if not path.exists():
                logger.warning(f"Embedding file {filename} not found.")
                return 0

            with open(path, 'rb') as f:
                raw_data = pickle.load(f)

            # Map global_id -> embedding vector
            global_to_vec = {}
            if isinstance(raw_data, dict) and 'node_index' in raw_data and 'embedding' in raw_data:
                for gid, vec in zip(raw_data['node_index'], raw_data['embedding']):
                    global_to_vec[int(gid)] = vec
            else:
                for gid, vec in raw_data.items():
                    global_to_vec[int(gid)] = vec

            # Align with HeteroData local indices
            count = 0
            global_ids = data[node_type].global_id.numpy()
            for local_id, gid in enumerate(global_ids):
                gid = int(gid)
                if gid in global_to_vec:
                    vec = global_to_vec[gid]
                    if len(vec) == expected_dim:
                        emb_buffer.data[local_id] = torch.tensor(vec, dtype=torch.float32)
                        count += 1
            
            logger.info(f"   Aligned {count:,} {node_type} nodes with BioBridge embeddings.")
            return count

        _align_modality("protein.pkl", "gene", self.gene_embs, 2560)
        _align_modality("drug.pkl", "drug", self.drug_embs, 512)
        _align_modality("disease.pkl", "disease", self.disease_embs, 768)

        logger.success("✅ BioBridge Alignment Complete.")

    def forward_for_type(self, node_type: str, indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieves and projects embeddings for a specific batch of nodes.
        'indices' are the local indices within the HeteroData node type.
        """
        device = indices.device
        
        if node_type == 'gene':
            return self.gene_proj(self.gene_embs[indices]) + self.fallbacks['gene'](indices)
        elif node_type == 'drug':
            return self.drug_proj(self.drug_embs[indices]) + self.fallbacks['drug'](indices)
        elif node_type == 'disease':
            return self.disease_proj(self.disease_embs[indices]) + self.fallbacks['disease'](indices)
        else:
            return torch.zeros(len(indices), self.target_dim, device=device)

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Projects the modality-specific embeddings into the target dimension.
        """
        out = {}
        for node_type in x_dict.keys():
            if node_type in ['gene', 'drug', 'disease']:
                indices = torch.arange(len(x_dict[node_type]), device=x_dict[node_type].device)
                out[node_type] = self.forward_for_type(node_type, indices)
        return out
