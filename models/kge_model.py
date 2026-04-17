"""
models/kge_model.py
──────────────────────────────────────────────────────────────────────────────
PyTorch Lightning module implementing the RotatE Knowledge Graph Embedding model.

🎓 WHAT YOU'LL LEARN:

1. KNOWLEDGE GRAPH EMBEDDINGS (KGE) — Core Concept:
   Goal: Learn vectors (embeddings) for entities and relations such that
   for each TRUE triple (head, relation, tail):
     score(head, relation, tail) >> score(fake_triple)
   
   TRUE:  (BRCA1, associated_with, Breast_Cancer) → score: 9.8
   FALSE: (BRCA1, associated_with, Diabetes)      → score: 0.3

2. RotatE MODEL:
   - Represents each relation as a ROTATION in complex vector space
   - head ◦ relation ≈ tail  (where ◦ is element-wise complex multiplication)
   - This means: applying the relation "rotation" to the head entity should
     give approximately the tail entity embedding
   - Better than TransE (additive) at capturing symmetric/antisymmetric relations

3. PYTORCH LIGHTNING STRUCTURE:
   - LightningModule: wraps your model + training logic cleanly
   - configure_optimizers(): tells Lightning how to optimize
   - training_step(): one batch of training
   - validation_step(): one batch of validation
   - Lightning handles: GPU placement, gradient zeroing, batch loops, logging

4. NEGATIVE SAMPLING:
   - We only have POSITIVE triples in DRKG (true relationships)
   - We need NEGATIVE examples too (false relationships) — we CREATE them
   - Corruption: randomly replace head or tail with a random entity
   - (BRCA1, associated_with, Breast_Cancer) → (RANDOM_GENE, associated_with, Breast_Cancer)
   - The model learns to score positives high and negatives low

5. METRICS:
   - MRR (Mean Reciprocal Rank): average of 1/rank of correct answer
     If the correct tail is ranked #1 → contribution = 1.0
     If ranked #5 → contribution = 0.2
   - Hits@K: how often the correct answer is in top K predictions
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


# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


class RotatEModel(L.LightningModule):
    """
    RotatE Knowledge Graph Embedding Model.
    
    Paper: "RotatE: Knowledge Graph Embedding by Relational Rotation in 
           Complex Space" (Sun et al., ICLR 2019)
    
    Args:
        n_entities:   Number of unique entities in the KG
        n_relations:  Number of unique relation types
        embed_dim:    Embedding dimension (higher = more expressive, more memory)
        gamma:        Margin hyperparameter for the loss function
        lr:           Learning rate for Adam optimizer
        pretrained_embeddings: Optional BioBridge embeddings to initialize from
    """

    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        embed_dim: int = 256,
        gamma: float = 12.0,
        lr: float = 3e-4,
        n_negative_samples: int = 128,
        pretrained_embeddings: Optional[np.ndarray] = None,
    ):
        super().__init__()
        
        # Save ALL args as hyperparameters — MLflow will log these automatically
        self.save_hyperparameters(ignore=["pretrained_embeddings"])

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embed_dim = embed_dim
        self.gamma = gamma
        self.lr = lr
        self.n_negative_samples = n_negative_samples

        # ── Entity Embeddings ──────────────────────────────────────────────────
        # Each entity gets a complex vector: real part + imaginary part
        # We store them as 2 * embed_dim reals = [real_0...real_d, imag_0...imag_d]
        # Shape: (n_entities, 2 * embed_dim)
        self.entity_embeddings = nn.Embedding(n_entities, 2 * embed_dim)

        # ── Relation Embeddings ────────────────────────────────────────────────
        # In RotatE, each relation is a ROTATION ANGLE in complex space
        # So relation embeddings are angles in [0, 2π]
        # Shape: (n_relations, embed_dim)
        self.relation_embeddings = nn.Embedding(n_relations, embed_dim)

        # ── Initialize Weights ─────────────────────────────────────────────────
        self._init_weights(pretrained_embeddings)

        # Track training metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def _init_weights(self, pretrained_embeddings: Optional[np.ndarray] = None):
        """
        Initialize embeddings. If we have BioBridge embeddings, use them!
        
        🎓 WHY INITIALIZATION MATTERS:
        - Random init: model starts knowing NOTHING about biology
        - BioBridge init: model starts with biomedical knowledge from 21M papers
        - This is the key insight of TRANSFER LEARNING
        """
        if pretrained_embeddings is not None:
            logger.info("🧬 Initializing entity embeddings with BioBridge pre-trained vectors!")
            
            n_pretrained, pretrained_dim = pretrained_embeddings.shape
            
            if pretrained_dim != 2 * self.embed_dim:
                # Project BioBridge 768-dim → 2 * embed_dim using linear projection
                logger.info(f"   Projecting {pretrained_dim}-dim BioBridge → {2*self.embed_dim}-dim")
                projection = nn.Linear(pretrained_dim, 2 * self.embed_dim, bias=False)
                with torch.no_grad():
                    projected = projection(
                        torch.FloatTensor(pretrained_embeddings[:self.n_entities])
                    )
                self.entity_embeddings.weight.data[:n_pretrained] = projected
            else:
                self.entity_embeddings.weight.data = torch.FloatTensor(pretrained_embeddings)
        else:
            logger.info("🎲 Initializing entity embeddings randomly (no BioBridge)")
            nn.init.uniform_(self.entity_embeddings.weight, -1.0, 1.0)

        # Relation embeddings: initialize as uniform angles [0, 2π]
        nn.init.uniform_(self.relation_embeddings.weight, -np.pi, np.pi)

    def rotate(self, head_complex, relation_angle):
        """
        Apply a rotation to a complex-valued entity embedding.
        
        🎓 COMPLEX NUMBER ROTATION:
        A complex number z = a + bi can be written as r * e^(iθ)
        Multiplication by e^(iθ) rotates z by angle θ:
          (a + bi) * (cos θ + i sin θ) = (a cos θ - b sin θ) + i(a sin θ + b cos θ)
        
        RotatE interprets relations as these rotation angles θ.
        
        Args:
            head_complex: (batch, embed_dim, 2)  — [real, imag] pairs
            relation_angle: (batch, embed_dim)    — rotation angles
        
        Returns:
            rotated: (batch, embed_dim, 2)
        """
        # Convert angles to cos/sin components
        cos_r = torch.cos(relation_angle).unsqueeze(-1)  # (batch, dim, 1)
        sin_r = torch.sin(relation_angle).unsqueeze(-1)  # (batch, dim, 1)

        # head_complex shape: (batch, dim, 2) where [:,:,0]=real, [:,:,1]=imag
        head_real = head_complex[..., 0:1]  # (batch, dim, 1)
        head_imag = head_complex[..., 1:2]  # (batch, dim, 1)

        # Complex multiplication: (a + bi)(cos + i sin) = (a cos - b sin) + i(a sin + b cos)
        rotated_real = head_real * cos_r - head_imag * sin_r
        rotated_imag = head_real * sin_r + head_imag * cos_r

        return torch.cat([rotated_real, rotated_imag], dim=-1)  # (batch, dim, 2)

    def score(self, head_ids: torch.Tensor, relation_ids: torch.Tensor, tail_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute RotatE scores for a batch of triples.
        
        Higher score = more likely to be a TRUE triple.
        
        Args:
            head_ids:     (batch,)
            relation_ids: (batch,)
            tail_ids:     (batch,)
        
        Returns:
            scores: (batch,) — the gamma - ||h ◦ r - t|| distance
        """
        # Look up embeddings
        head_emb = self.entity_embeddings(head_ids)       # (batch, 2*dim)
        tail_emb = self.entity_embeddings(tail_ids)       # (batch, 2*dim)
        rel_emb = self.relation_embeddings(relation_ids)  # (batch, dim)

        # Reshape to complex pairs: (batch, dim, 2)
        head_complex = head_emb.view(-1, self.embed_dim, 2)
        tail_complex = tail_emb.view(-1, self.embed_dim, 2)

        # Apply rotation: h ◦ r
        rotated = self.rotate(head_complex, rel_emb)  # (batch, dim, 2)

        # Distance: ||h ◦ r - t||
        # Small distance = head and tail are related by this relation
        diff = rotated - tail_complex   # (batch, dim, 2)
        distance = diff.norm(dim=-1).sum(dim=-1)  # (batch,)

        # Score = gamma - distance (so higher score = smaller distance = better)
        return self.gamma - distance

    def _sample_negatives(self, heads, relations, tails):
        """
        Generate negative triples by randomly corrupting head or tail.
        
        🎓 NEGATIVE SAMPLING STRATEGY:
        For each positive triple (h, r, t), we create n_negative_samples fakes:
        - 50% of time: corrupt the HEAD → (random_entity, r, t)
        - 50% of time: corrupt the TAIL → (h, r, random_entity)
        
        The model must learn to score these LOWER than the positive triple.
        """
        batch_size = heads.shape[0]
        device = heads.device

        # Random entity IDs for corruption
        neg_entities = torch.randint(0, self.n_entities, (batch_size, self.n_negative_samples), device=device)

        # Randomly decide to corrupt head (0) or tail (1)
        corrupt_head = torch.rand(batch_size, self.n_negative_samples, device=device) < 0.5

        # Build negative heads and tails
        neg_heads = torch.where(corrupt_head, neg_entities, heads.unsqueeze(1).expand_as(neg_entities))
        neg_tails = torch.where(~corrupt_head, neg_entities, tails.unsqueeze(1).expand_as(neg_entities))
        neg_relations = relations.unsqueeze(1).expand_as(neg_entities)

        return neg_heads, neg_relations, neg_tails

    def training_step(self, batch, batch_idx):
        """
        One step of gradient descent.
        
        🎓 PYTORCH LIGHTNING MAGIC:
        You DON'T need to write:
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        Lightning handles all of this for you!
        
        You just need to: compute loss, return it.
        """
        heads, relations, tails = batch

        # Score positive triples (should be HIGH)
        pos_scores = self.score(heads, relations, tails)  # (batch,)

        # Sample and score negatives (should be LOW)
        neg_h, neg_r, neg_t = self._sample_negatives(heads, relations, tails)
        batch_size, n_neg = neg_h.shape

        # Flatten, score, reshape
        neg_scores = self.score(
            neg_h.view(-1), neg_r.view(-1), neg_t.view(-1)
        ).view(batch_size, n_neg)  # (batch, n_neg)

        # Self-adversarial negative sampling loss (RotatE paper)
        # Weight negatives by their scores — hard negatives get higher weight
        neg_weights = F.softmax(neg_scores * 1.0, dim=-1).detach()

        # Margin ranking loss:
        # positive_loss: log-sigmoid of positive score  (want: high)
        # negative_loss: weighted log-sigmoid of negative scores  (want: low)
        pos_loss = -F.logsigmoid(pos_scores).mean()
        neg_loss = -(neg_weights * F.logsigmoid(-neg_scores)).sum(dim=-1).mean()
        loss = (pos_loss + neg_loss) / 2

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/pos_score", pos_scores.mean(), on_epoch=True)
        self.log("train/neg_score", neg_scores.mean(), on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Evaluate on validation triples.
        
        We compute RANKING METRICS:
        For each test triple (h, r, t):
          1. Replace tail with every entity in the KG
          2. Score all (h, r, candidate) pairs
          3. Find the rank of the TRUE tail
          4. Compute 1/rank (MRR) and check if rank <= K (Hits@K)
        
        Note: This is slow (n_entities forward passes per triple).
        In practice we use a subset for validation.
        """
        heads, relations, tails = batch

        # For efficiency, we only do full link prediction on small batches
        # during validation. For final evaluation, use evaluate.py
        pos_scores = self.score(heads, relations, tails)
        
        # Quick approximate metric: score vs random negatives
        neg_h, neg_r, neg_t = self._sample_negatives(heads, relations, tails)
        neg_scores = self.score(
            neg_h.view(-1), neg_r.view(-1), neg_t.view(-1)
        ).view(heads.shape[0], -1)

        # Approximate MRR: fraction of positives scoring higher than negatives
        pos_expanded = pos_scores.unsqueeze(1)  # (batch, 1)
        approx_mrr = (pos_expanded > neg_scores).float().mean()

        pos_loss = -F.logsigmoid(pos_scores).mean()
        neg_loss = -F.logsigmoid(-neg_scores).mean()
        val_loss = (pos_loss + neg_loss) / 2

        self.log("val/loss", val_loss, prog_bar=True)
        self.log("val/approx_mrr", approx_mrr, prog_bar=True)

        return val_loss

    def configure_optimizers(self):
        """
        Set up the optimizer and learning rate scheduler.
        
        🎓 ADAM OPTIMIZER:
        - Adaptive Moment Estimation
        - Most popular optimizer for deep learning
        - Adjusts learning rate per-parameter based on gradient history
        
        🎓 COSINE ANNEALING SCHEDULER:
        - Gradually reduces LR following a cosine curve
        - Good at finding sharp minima late in training
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=self.lr * 0.01
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }

    def predict(self, head_id: int, relation_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Predict the most likely tail entities given a head and relation.
        
        This is the INFERENCE function — what the API will call.
        
        Args:
            head_id:     Integer ID of the head entity
            relation_id: Integer ID of the relation
            top_k:       Return top K predictions
        
        Returns:
            List of (entity_id, score) tuples, sorted by score descending
        """
        self.eval()
        device = next(self.parameters()).device

        head_tensor = torch.tensor([head_id], device=device)
        relation_tensor = torch.tensor([relation_id], device=device)

        # Score against ALL possible tail entities
        all_tails = torch.arange(self.n_entities, device=device)
        heads_expanded = head_tensor.expand(self.n_entities)
        relations_expanded = relation_tensor.expand(self.n_entities)

        with torch.no_grad():
            scores = self.score(heads_expanded, relations_expanded, all_tails)

        # Get top K
        top_scores, top_indices = torch.topk(scores, k=min(top_k, self.n_entities))

        return list(zip(
            top_indices.cpu().tolist(),
            top_scores.cpu().tolist()
        ))
