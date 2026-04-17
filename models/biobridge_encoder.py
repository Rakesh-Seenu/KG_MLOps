"""
models/biobridge_encoder.py
──────────────────────────────────────────────────────────────────────────────
Wraps the BioBridge pre-trained model to generate rich biomedical embeddings.

🎓 WHAT YOU'LL LEARN:

1. WHAT IS BioBridge?
   - A 2024 NeurIPS paper from Stanford
   - It "bridges" different biomedical data modalities into ONE embedding space:
     • Text descriptions    →  768-dim vector
     • Protein sequences    →  768-dim vector  
     • Drug SMILES strings  →  768-dim vector
   - This means a gene described in text and a protein sequence can be
     compared directly! This is multi-modal learning.

2. WHAT IS BiomedBERT?
   - Microsoft's BERT model pre-trained on 21 million biomedical abstracts
   - Much better than regular BERT for bio/pharma text
   - BioBridge uses it as its text encoder

3. WHY USE EMBEDDINGS FOR KG LINK PREDICTION?
   - Raw entity IDs like "Gene::9606/23210" carry NO semantic information
   - But the text "BRCA1 is a tumor suppressor gene involved in DNA repair"
     is rich with meaning
   - BioBridge embeddings give our model a HEAD START from pre-trained knowledge

4. TRANSFER LEARNING:
   - We DON'T train BioBridge from scratch (that needs 21M papers + months)
   - We use the pre-trained weights and just FINE-TUNE for our task
   - This is why large pre-trained models are so powerful in practice
"""

import json
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ── Constants ─────────────────────────────────────────────────────────────────
BIOBRIDGE_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
# Note: The full BioBridge model is at "QSong5/BioBridge" on HuggingFace
# We use BiomedBERT directly as the text encoder — same underlying model.

EMBEDDING_DIM = 768  # BiomedBERT outputs 768-dimensional vectors
CACHE_DIR = Path(__file__).parent.parent / "data" / "embeddings_cache"
BATCH_SIZE = 64      # Process 64 entities at a time to fit in GPU memory


class BioBridgeEncoder:
    """
    Encodes biomedical entities (genes, diseases, drugs) into dense vectors.
    
    Usage:
        encoder = BioBridgeEncoder()
        entity_names = ["BRCA1 breast cancer gene", "Ibuprofen anti-inflammatory"]
        embeddings = encoder.encode(entity_names)  # shape: (2, 768)
    """

    def __init__(
        self,
        model_name: str = BIOBRIDGE_MODEL,
        device: Optional[str] = None,
        cache_embeddings: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_embeddings = cache_embeddings
        self.model_name = model_name

        logger.info(f"🧬 Loading BioBridge encoder: {model_name}")
        logger.info(f"   Device: {self.device}")

        # Load tokenizer and model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()  # IMPORTANT: eval mode disables dropout for inference

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.success("✅ BioBridge encoder loaded!")

    def _get_cache_path(self, text: str) -> Path:
        """Generate a unique cache filename based on the text hash."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return CACHE_DIR / f"{text_hash}.npy"

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text string into a 768-dim embedding.
        
        🎓 HOW BERT ENCODING WORKS:
        1. Tokenizer splits text into subword tokens
           "BRCA1" → ["BR", "##CA", "##1"]  (subword tokenization)
        2. Model processes tokens through 12 transformer layers
        3. We take the [CLS] token embedding as the sentence representation
           [CLS] is the special "summary" token at position 0
        """
        cache_path = self._get_cache_path(text)

        # Check cache first — avoids re-computing for same entities
        if self.cache_embeddings and cache_path.exists():
            return np.load(cache_path)

        # Tokenize: convert text → input IDs + attention mask
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=128,         # Truncate very long texts
            padding="max_length",   # Pad short texts
            truncation=True,
        ).to(self.device)

        # Forward pass through BiomedBERT
        with torch.no_grad():  # no_grad: don't compute gradients during inference
            outputs = self.model(**inputs)

        # Extract [CLS] token embedding: shape (1, 768) → (768,)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        # Cache for future use
        if self.cache_embeddings:
            np.save(cache_path, cls_embedding)

        return cls_embedding

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of texts in batches.
        
        Batching is crucial for GPU efficiency:
        - GPU excels at parallel matrix operations
        - Processing one text at a time wastes 99% of GPU capacity
        - With batch_size=64, we process 64 texts simultaneously
        
        Returns:
            np.ndarray of shape (len(texts), 768)
        """
        all_embeddings = []
        
        iterator = range(0, len(texts), BATCH_SIZE)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding entities", unit="batch")

        for batch_start in iterator:
            batch_texts = texts[batch_start:batch_start + BATCH_SIZE]

            # Check which texts need encoding (rest are cached)
            uncached_texts = []
            uncached_indices = []
            batch_embeddings = [None] * len(batch_texts)

            for i, text in enumerate(batch_texts):
                cache_path = self._get_cache_path(text)
                if self.cache_embeddings and cache_path.exists():
                    batch_embeddings[i] = np.load(cache_path)
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Batch encode uncached texts
            if uncached_texts:
                inputs = self.tokenizer(
                    uncached_texts,
                    return_tensors="pt",
                    max_length=128,
                    padding=True,
                    truncation=True,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Shape: (batch_size, sequence_length, 768) → (batch_size, 768)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                for i, (idx, emb) in enumerate(zip(uncached_indices, cls_embeddings)):
                    batch_embeddings[idx] = emb
                    if self.cache_embeddings:
                        np.save(self._get_cache_path(uncached_texts[i]), emb)

            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)  # Shape: (N, 768)


def encode_drkg_entities(data_dir: Path = None) -> np.ndarray:
    """
    Encodes all DRKG entities using BioBridge.
    
    DRKG entity format: "TypeName::EntityID"
    We use the entity ID as the text input, cleaned up for readability.
    
    Example:
        "Gene::9606/23210"  →  tokenize as "Gene 23210"
        "Disease::MESH:D003920"  →  tokenize as "Disease MESH D003920"
    
    For a production system, you'd look up the actual names from a biomedical
    database like UniProt (proteins) or PubChem (compounds). That's a 
    great extension for your PhD application portfolio!
    """
    data_dir = data_dir or Path(__file__).parent.parent / "data" / "processed"
    entity2id_path = data_dir / "entity2id.json"

    if not entity2id_path.exists():
        logger.error("entity2id.json not found. Run data/preprocess.py first.")
        raise FileNotFoundError(entity2id_path)

    with open(entity2id_path) as f:
        entity2id = json.load(f)

    # Sort by ID to get consistent ordering
    entities_sorted = sorted(entity2id.items(), key=lambda x: x[1])
    entity_names = [e[0] for e in entities_sorted]

    # Clean entity names for better tokenization
    def clean_entity_name(name: str) -> str:
        """Convert "Gene::9606/23210" → "Gene 23210" for BiomedBERT."""
        parts = name.split("::", 1)
        if len(parts) == 2:
            entity_type, entity_id = parts
            # Clean up the ID
            entity_id = entity_id.replace("/", " ").replace("_", " ").replace(":", " ")
            return f"{entity_type} {entity_id}"
        return name

    cleaned_names = [clean_entity_name(e) for e in entity_names]
    logger.info(f"📊 Encoding {len(cleaned_names):,} DRKG entities with BioBridge...")
    logger.info(f"   Examples: {cleaned_names[:3]}")

    encoder = BioBridgeEncoder()
    embeddings = encoder.encode(cleaned_names)

    # Save embeddings
    output_path = data_dir / "entity_embeddings.npy"
    np.save(output_path, embeddings)
    logger.success(f"✅ Saved {embeddings.shape} embedding matrix to {output_path}")
    logger.info(f"   Size on disk: {output_path.stat().st_size / (1024**2):.1f} MB")

    return embeddings


if __name__ == "__main__":
    embeddings = encode_drkg_entities()
    logger.info(f"\n🎉 Entity embeddings shape: {embeddings.shape}")
    logger.info(f"   Each entity is a {embeddings.shape[1]}-dimensional vector")
    logger.info(f"   These are the STARTING POINT for our KGE model!")
