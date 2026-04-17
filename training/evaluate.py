"""
training/evaluate.py
──────────────────────────────────────────────────────────────────────────────
Evaluation script for the PrimeKG RotatE model.
Computes standard Knowledge Graph Embedding metrics: Hits@K and MRR.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from data.datamodule import PrimeKGDataModule
from models.kge_model import RotatEModel

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

def evaluate_model(checkpoint_path: str, data_dir: Path = DATA_DIR, batch_size: int = 128, top_k: List[int] = [1, 3, 10]):
    logger.info(f"🔍 Loading model from {checkpoint_path}")
    
    # Load model and set to eval mode
    model = RotatEModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Load DataModule to get the test dataloader
    datamodule = PrimeKGDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=4)
    datamodule.prepare_data()
    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()
    
    n_entities = model.max_node_idx + 1
    device = model.device

    hits_counts = {k: 0 for k in top_k}
    mrr_sum = 0.0
    total_samples = 0

    logger.info("🧪 Starting Evaluation on Test Set...")
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Scoring Test Triples"):
            heads, relations, tails = [x.to(device) for x in batch]
            
            # The standard protocol: compute rank of true tail vs ALL possible tails
            # Because n_entities is ~130,000, we do this iteratively for each triple
            # or in small batches of targets
            
            for i in range(len(heads)):
                h = heads[i].unsqueeze(0).expand(n_entities)
                r = relations[i].unsqueeze(0).expand(n_entities)
                all_t = torch.arange(n_entities, device=device)
                
                # Score all possible tails
                scores = model.score(h, r, all_t)
                
                # Find rank of the true tail
                true_tail = tails[i].item()
                true_score = scores[true_tail].item()
                
                # Rank is the number of entities with score > true_score
                # +1 because best rank is 1
                rank = (scores > true_score).sum().item() + 1
                
                # Update metrics
                mrr_sum += 1.0 / rank
                for k in top_k:
                    if rank <= k:
                        hits_counts[k] += 1
                        
                total_samples += 1

    mrr = mrr_sum / total_samples
    hits_pct = {f"Hits@{k}": count / total_samples for k, count in hits_counts.items()}
    
    logger.success("\n🎉 Evaluation Complete!")
    logger.info(f"   Total Test Triples: {total_samples:,}")
    logger.info(f"   MRR: {mrr:.4f}")
    for k, val in hits_pct.items():
        logger.info(f"   {k}: {val:.4f}")

    return mrr, hits_pct

if __name__ == "__main__":
    parser = argparse.ArgumentParser("BioKG Evaluator")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .ckpt model file")
    args = parser.parse_args()
    
    evaluate_model(args.ckpt)
