"""
models/kge_model.py
──────────────────────────────────────────────────────────────────────────────
PyTorch Lightning module implementing the RotatE Knowledge Graph Embedding model.

🎓 INTEGRATING BIOBRIDGE:
   Instead of initializing random `nn.Embedding`s for entities, this model 
   incorporates the `BioBridgeProjector`. It dynamically maps Pre-computed 
   Proteins, Drugs, and Diseases into the RotatE Complex space (Real+Imaginary).

🎓 RotatE MODEL:
   - Represents each relation as a ROTATION in complex vector space
   - head ◦ relation ≈ tail  (◦ is element-wise complex multiplication)
"""

import json
from pathlib import Path
from typing import Optional, Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from loguru import logger

from .biobridge_encoder import BioBridgeProjector

class RotatEModel(L.LightningModule):
    """
    Advanced RotatE Knowledge Graph Embedding Model integrated with BioBridge.
    
    This implementation follows Senior Engineer patterns:
    - Explicit type hinting for all methods.
    - Modular rotation and scoring logic.
    - Integrated inference step (predict_step) for production use.
    - Leverages BioBridge Multimodal Projectors for biological node embeddings.
    """

    def __init__(
        self,
        max_node_idx: int,
        n_relations: int,
        embed_dim: int = 384,
        gamma: float = 12.0,
        lr: float = 3e-4,
        n_negative_samples: int = 128,
        use_biobridge: bool = True
    ):
        """
        Initialize the RotatE Model.

        Args:
            max_node_idx: Highest node index in the KG (PrimeKG).
            n_relations: Number of unique relationship types.
            embed_dim: The dimension of the complex embedding space (Real + Imaginary).
            gamma: Fixed margin for the distance-based score function.
            lr: Learning rate for training.
            n_negative_samples: Number of corrupt triples to sample per positive.
            use_biobridge: Whether to use pretrained multimodal projectors.
        """
        super().__init__()
        self.save_hyperparameters()

        self.max_node_idx = max_node_idx
        self.n_relations = n_relations
        self.embed_dim = embed_dim
        self.gamma = gamma
        self.lr = lr
        self.n_negative_samples = n_negative_samples
        self.use_biobridge = use_biobridge

        # RotatE works in complex space (Head_real + i*Head_imag)
        # BioBridge projects to 2 * embed_dim (e.g. 768) to fill both components.
        target_unified_dim = self.embed_dim * 2

        # ── Entity Embeddings ──────────────────────────────────────────────────
        if self.use_biobridge:
            logger.info(f"🧬 Bootstrapping RotatE with BioBridge Projector (Target Dim: {target_unified_dim})")
            self.entity_embeddings = BioBridgeProjector(
                max_node_index=max_node_idx, 
                target_dim=target_unified_dim
            )
            self.entity_embeddings.load_pretrained_mappings()
        else:
            logger.info("🎲 Initializing randomized entity embeddings (He initialization)")
            self.entity_embeddings = nn.Embedding(max_node_idx + 1, target_unified_dim)
            nn.init.uniform_(self.entity_embeddings.weight, -1.0, 1.0)

        # ── Relation Embeddings (Phase/Angle in complex space) ──────────────────
        # Relations rotate the head entity by an angle theta.
        self.relation_embeddings = nn.Embedding(n_relations, embed_dim)
        nn.init.uniform_(self.relation_embeddings.weight, -np.pi, np.pi)

    def rotate(self, head_complex: torch.Tensor, relation_angle: torch.Tensor) -> torch.Tensor:
        """
        Applies element-wise complex rotation: h_rotated = h * e^(i * r).
        
        Args:
            head_complex: [batch, embed_dim, 2] (Real and Imaginary parts).
            relation_angle: [batch, embed_dim] (Phase/Angle for rotation).
            
        Returns:
            rotated_complex: [batch, embed_dim, 2].
        """
        cos_r = torch.cos(relation_angle).unsqueeze(-1)
        sin_r = torch.sin(relation_angle).unsqueeze(-1)

        head_real = head_complex[..., 0:1]
        head_imag = head_complex[..., 1:2]

        rotated_real = head_real * cos_r - head_imag * sin_r
        rotated_imag = head_real * sin_r + head_imag * cos_r

        return torch.cat([rotated_real, rotated_imag], dim=-1)

    def score(
        self, 
        head_ids: torch.Tensor, 
        relation_ids: torch.Tensor, 
        tail_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the distance score: score = gamma - ||h * r - t||_2.
        Lower distance = higher score = higher probability of link existence.
        """
        # Lookup Embeddings
        head_emb = self.entity_embeddings(head_ids)
        tail_emb = self.entity_embeddings(tail_ids)
        rel_emb = self.relation_embeddings(relation_ids)

        # Reshape to Complex Space: h, t ∈ ℝ^(d x 2)
        head_complex = head_emb.view(-1, self.embed_dim, 2)
        tail_complex = tail_emb.view(-1, self.embed_dim, 2)

        # Apply Rotation: h_rotated = h ◦ r
        rotated = self.rotate(head_complex, rel_emb)
        
        # Calculate L2 Distance
        diff = rotated - tail_complex
        distance = diff.norm(dim=-1).sum(dim=-1)

        return self.gamma - distance

    def _sample_negatives(
        self, 
        heads: torch.Tensor, 
        relations: torch.Tensor, 
        tails: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples corrupt entities for negative contrastive loss."""
        batch_size = heads.shape[0]
        device = heads.device

        # Sample across entire space of valid node ids
        neg_entities = torch.randint(0, self.max_node_idx, (batch_size, self.n_negative_samples), device=device)
        corrupt_head = torch.rand(batch_size, self.n_negative_samples, device=device) < 0.5

        neg_heads = torch.where(corrupt_head, neg_entities, heads.unsqueeze(1).expand_as(neg_entities))
        neg_tails = torch.where(~corrupt_head, neg_entities, tails.unsqueeze(1).expand_as(neg_entities))
        neg_relations = relations.unsqueeze(1).expand_as(neg_entities)

        return neg_heads.reshape(-1), neg_relations.reshape(-1), neg_tails.reshape(-1)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        heads, relations, tails = batch
        batch_size = heads.shape[0]

        # 1. Positives
        pos_scores = self.score(heads, relations, tails)
        
        # 2. Negatives
        neg_h, neg_r, neg_t = self._sample_negatives(heads, relations, tails)
        neg_scores = self.score(neg_h, neg_r, neg_t).view(batch_size, -1)

        # 3. SELF-ADVERSARIAL LOSS (Sun et al.)
        # Adaptive weights for harder negatives
        neg_weights = F.softmax(neg_scores * 1.0, dim=-1).detach()
        pos_loss = -F.logsigmoid(pos_scores).mean()
        neg_loss = -(neg_weights * F.logsigmoid(-neg_scores)).sum(dim=-1).mean()
        
        loss = (pos_loss + neg_loss) / 2

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        heads, relations, tails = batch
        pos_scores = self.score(heads, relations, tails)
        
        # Quick MRR Check
        neg_h, neg_r, neg_t = self._sample_negatives(heads, relations, tails)
        neg_scores = self.score(neg_h, neg_r, neg_t).view(heads.shape[0], -1)

        approx_mrr = (pos_scores.unsqueeze(1) > neg_scores).float().mean()
        val_loss = -F.logsigmoid(pos_scores).mean()

        self.log("val/loss", val_loss, prog_bar=True)
        self.log("val/approx_mrr", approx_mrr, prog_bar=True)
        return val_loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Lightning built-in inference hook for large scale prediction."""
        heads, relations, tails = batch
        return self.score(heads, relations, tails)

    @torch.no_grad()
    def predict(self, head_index: int, relation_id: int, top_k: int = 10) -> list:
        """
        High-level inference API for UI consumption. 
        Efficiently scores one head+relation against ALL possible tails.
        """
        device = self.device
        all_tail_ids = torch.arange(0, self.max_node_idx + 1, device=device)
        
        # Vectorized scoring
        head_batch = torch.full((len(all_tail_ids),), head_index, device=device)
        rel_batch = torch.full((len(all_tail_ids),), relation_id, device=device)
        
        # Score in chunks to avoid GPU OOM for 129K entities
        scores = []
        chunk_size = 32768
        for i in range(0, len(all_tail_ids), chunk_size):
            s = self.score(
                head_batch[i:i+chunk_size], 
                rel_batch[i:i+chunk_size], 
                all_tail_ids[i:i+chunk_size]
            )
            scores.append(s)
            
        scores = torch.cat(scores)
        top_scores, top_indices = torch.topk(scores, k=top_k)
        
        return list(zip(top_indices.cpu().tolist(), top_scores.cpu().tolist()))

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=self.lr * 0.01
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}
