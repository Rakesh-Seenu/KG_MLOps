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
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from loguru import logger

from .biobridge_encoder import BioBridgeProjector

class RotatEModel(L.LightningModule):
    """
    RotatE paired with BioBridge Multimodal Encoders.
    
    Args:
        max_node_idx: Highest node index in PrimeKG
        n_relations:  Number of unique relation types
        embed_dim:    Complex space embedding dimension (half of BioBridge 768 target)
        gamma:        Margin hyperparameter for the loss function
        lr:           Learning rate for Adam optimizer
    """

    def __init__(
        self,
        max_node_idx: int,
        n_relations: int,
        embed_dim: int = 384, # BioBridge projects to 768 (2 * 384 for Complex Space)
        gamma: float = 12.0,
        lr: float = 3e-4,
        n_negative_samples: int = 128,
        use_biobridge: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()

        self.max_node_idx = max_node_idx
        self.n_relations = n_relations
        self.embed_dim = embed_dim
        self.gamma = gamma
        self.lr = lr
        self.n_negative_samples = n_negative_samples
        self.use_biobridge = use_biobridge

        target_unified_dim = self.embed_dim * 2

        # ── Entity Embeddings ──────────────────────────────────────────────────
        if self.use_biobridge:
            logger.info(f"🧬 Initializing RotatE with BioBridge Projector (dim={target_unified_dim})")
            self.entity_embeddings = BioBridgeProjector(max_node_idx=max_node_idx, target_dim=target_unified_dim)
            self.entity_embeddings.load_pretrained_mappings()
        else:
            logger.info("🎲 Initializing entity embeddings randomly")
            self.entity_embeddings = nn.Embedding(max_node_idx + 1, target_unified_dim)
            nn.init.uniform_(self.entity_embeddings.weight, -1.0, 1.0)

        # ── Relation Embeddings ────────────────────────────────────────────────
        self.relation_embeddings = nn.Embedding(n_relations, embed_dim)
        nn.init.uniform_(self.relation_embeddings.weight, -np.pi, np.pi)

    def rotate(self, head_complex, relation_angle):
        """Apply relation angle rotation to head entity in complex sequence."""
        cos_r = torch.cos(relation_angle).unsqueeze(-1)
        sin_r = torch.sin(relation_angle).unsqueeze(-1)

        head_real = head_complex[..., 0:1]
        head_imag = head_complex[..., 1:2]

        rotated_real = head_real * cos_r - head_imag * sin_r
        rotated_imag = head_real * sin_r + head_imag * cos_r

        return torch.cat([rotated_real, rotated_imag], dim=-1)

    def score(self, head_ids: torch.Tensor, relation_ids: torch.Tensor, tail_ids: torch.Tensor) -> torch.Tensor:
        """Compute RotatE scores for triples: gamma - ||h ◦ r - t||"""
        
        # Determine how to lookup entities
        if self.use_biobridge:
            head_emb = self.entity_embeddings(head_ids)
            tail_emb = self.entity_embeddings(tail_ids)
        else:
            head_emb = self.entity_embeddings(head_ids)
            tail_emb = self.entity_embeddings(tail_ids)
            
        rel_emb = self.relation_embeddings(relation_ids)

        head_complex = head_emb.view(-1, self.embed_dim, 2)
        tail_complex = tail_emb.view(-1, self.embed_dim, 2)

        rotated = self.rotate(head_complex, rel_emb)
        diff = rotated - tail_complex
        distance = diff.norm(dim=-1).sum(dim=-1)

        return self.gamma - distance

    def _sample_negatives(self, heads, relations, tails):
        batch_size = heads.shape[0]
        device = heads.device

        # Sample across entire space of valid node ids
        neg_entities = torch.randint(0, self.max_node_idx, (batch_size, self.n_negative_samples), device=device)
        corrupt_head = torch.rand(batch_size, self.n_negative_samples, device=device) < 0.5

        neg_heads = torch.where(corrupt_head, neg_entities, heads.unsqueeze(1).expand_as(neg_entities))
        neg_tails = torch.where(~corrupt_head, neg_entities, tails.unsqueeze(1).expand_as(neg_entities))
        neg_relations = relations.unsqueeze(1).expand_as(neg_entities)

        return neg_heads, neg_relations, neg_tails

    def training_step(self, batch, batch_idx):
        heads, relations, tails = batch

        pos_scores = self.score(heads, relations, tails)
        
        neg_h, neg_r, neg_t = self._sample_negatives(heads, relations, tails)
        batch_size, n_neg = neg_h.shape

        neg_scores = self.score(
            neg_h.view(-1), neg_r.view(-1), neg_t.view(-1)
        ).view(batch_size, n_neg)

        neg_weights = F.softmax(neg_scores * 1.0, dim=-1).detach()
        pos_loss = -F.logsigmoid(pos_scores).mean()
        neg_loss = -(neg_weights * F.logsigmoid(-neg_scores)).sum(dim=-1).mean()
        loss = (pos_loss + neg_loss) / 2

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        heads, relations, tails = batch

        pos_scores = self.score(heads, relations, tails)
        neg_h, neg_r, neg_t = self._sample_negatives(heads, relations, tails)
        
        neg_scores = self.score(
            neg_h.view(-1), neg_r.view(-1), neg_t.view(-1)
        ).view(heads.shape[0], -1)

        pos_expanded = pos_scores.unsqueeze(1)
        approx_mrr = (pos_expanded > neg_scores).float().mean()

        pos_loss = -F.logsigmoid(pos_scores).mean()
        neg_loss = -F.logsigmoid(-neg_scores).mean()
        val_loss = (pos_loss + neg_loss) / 2

        self.log("val/loss", val_loss, prog_bar=True)
        self.log("val/approx_mrr", approx_mrr, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        heads, relations, tails = batch

        pos_scores = self.score(heads, relations, tails)
        neg_h, neg_r, neg_t = self._sample_negatives(heads, relations, tails)
        
        neg_scores = self.score(
            neg_h.view(-1), neg_r.view(-1), neg_t.view(-1)
        ).view(heads.shape[0], -1)

        pos_expanded = pos_scores.unsqueeze(1)
        approx_mrr = (pos_expanded > neg_scores).float().mean()

        pos_loss = -F.logsigmoid(pos_scores).mean()
        neg_loss = -F.logsigmoid(-neg_scores).mean()
        test_loss = (pos_loss + neg_loss) / 2

        self.log("test/loss", test_loss, prog_bar=True)
        self.log("test/approx_mrr", approx_mrr, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=self.lr * 0.01
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}
