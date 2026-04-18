import os
import torch
import pandas as pd
import pytorch_lightning as pl
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit
from loguru import logger
from pathlib import Path

# Try to use RAPIDS for lightning-fast graph construction in Colab
try:
    import cudf
    USE_GPU = True
except ImportError:
    USE_GPU = False

class BioBridgeGNNDataModule(pl.LightningDataModule):
    """
    Production-grade DataModule for BioBridge/PrimeKG.
    Handles real-world graph construction from 8.1M triples and multimodal 
    embedding alignment using NVIDIA RAPIDS.
    """
    def __init__(self, data_dir: str = 'data/processed/', batch_size: int = 1024, num_workers: int = 4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.target_edge_type = ("drug", "treats", "disease")
        self.num_neighbors = [15, 10]
        self.graph_path = self.data_dir / 'biobridge_hetero_graph.pt'

    def setup(self, stage: str = None):
        """
        Loads the pre-processed graph or builds it from raw Parquet files.
        """
        if self.graph_path.exists():
            logger.info(f"🚀 Loading Real BioBridge Graph from {self.graph_path}...")
            # weights_only=False is required for PyTorch 2.6+ to load complex PyG objects
            self.data = torch.load(self.graph_path, weights_only=False)
        else:
            logger.info("🛠️  Graph .pt file not found. Building real HeteroData from Parquet...")
            self.data = self._build_from_processed()
            torch.save(self.data, self.graph_path)
            logger.success(f"💾 Saved processed graph to {self.graph_path}")

        # Splitting for Link Prediction
        transform = RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            disjoint_train_ratio=0.3,
            edge_types=[self.target_edge_type],
            rev_edge_types=[("disease", "rev_treats", "drug")],
            add_negative_train_samples=False,
            neg_sampling_ratio=1.0
        )
        
        self.train_data, self.val_data, self.test_data = transform(self.data)
        logger.info(f"✅ Real graph setup complete: {self.data.num_nodes} nodes, {self.data.num_edges} edges.")

    def _build_from_processed(self) -> HeteroData:
        """
        Parses 8.1M triples and constructs the PyG HeteroData object.
        """
        parquet_path = self.data_dir / "primekg_edges.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file {parquet_path} missing. Run preprocess.py first.")

        logger.info(f"Reading {parquet_path} (Using GPU: {USE_GPU})...")
        if USE_GPU:
            df = cudf.read_parquet(parquet_path)
        else:
            df = pd.read_parquet(parquet_path)

        data = HeteroData()

        # --- 1. Map Global IDs to Type-Specific Local IDs ---
        # PrimeKG uses global indices (0 to 129k). PyG HeteroData requires 
        # local indices (0 to N) for each node type.
        
        def _build_nodes(df_sub, node_type):
            logger.info(f"   Extracting unique nodes for: {node_type}")
            if USE_GPU:
                # Get unique IDs across head and tail if they match the type
                h = df_sub[df_sub['head_type'] == node_type]['head_index']
                t = df_sub[df_sub['tail_type'] == node_type]['tail_index']
                unique_ids = cudf.concat([h, t]).unique().sort_values()
                id_map = cudf.DataFrame({'global_id': unique_ids, 'local_id': cudf.Series(range(len(unique_ids)))})
                return unique_ids.to_pandas(), id_map
            else:
                h = df_sub[df_sub['head_type'] == node_type]['head_index']
                t = df_sub[df_sub['tail_type'] == node_type]['tail_index']
                unique_ids = pd.concat([h, t]).unique()
                unique_ids.sort()
                id_map = pd.DataFrame({'global_id': unique_ids, 'local_id': range(len(unique_ids))})
                return unique_ids, id_map

        # Map 'gene/protein' from PrimeKG to 'gene' in our GNN
        gene_ids, gene_map = _build_nodes(df, 'gene/protein')
        drug_ids, drug_map = _build_nodes(df, 'drug')
        dis_ids, dis_map = _build_nodes(df, 'disease')

        # Store Global IDs for alignment with BioBridge embeddings
        data['gene'].global_id = torch.tensor(gene_ids.values if hasattr(gene_ids, 'values') else gene_ids, dtype=torch.long)
        data['drug'].global_id = torch.tensor(drug_ids.values if hasattr(drug_ids, 'values') else drug_ids, dtype=torch.long)
        data['disease'].global_id = torch.tensor(dis_ids.values if hasattr(dis_ids, 'values') else dis_ids, dtype=torch.long)

        # Initialize Features (Placeholder for now, aligned in Encoder)
        data['gene'].x = torch.randn(len(gene_ids), 128)
        data['drug'].x = torch.randn(len(drug_ids), 128)
        data['disease'].x = torch.randn(len(dis_ids), 128)

        # --- 2. Build Edge Indices ---
        def _add_edge_type(src_type, rel, dst_type, rev_rel=None):
            logger.info(f"   Building edges for: ({src_type}, {rel}, {dst_type})")
            # Filter triples
            mask = (df['head_type'] == (src_type if src_type != 'gene' else 'gene/protein')) & \
                   (df['tail_type'] == (dst_type if dst_type != 'gene' else 'gene/protein'))
            # Filter by relation if needed, but for BioBridge demo we often take all links between types
            # as a start, or filter for specific relation strings.
            sub_df = df[mask]
            
            # Map to local IDs
            # This is slow in Pandas but 100x faster in cuDF merge
            src_map = drug_map if src_type == 'drug' else (gene_map if src_type == 'gene' else dis_map)
            dst_map = dis_map if dst_type == 'disease' else (drug_map if dst_type == 'drug' else gene_map)
            
            sub_df = sub_df.merge(src_map.rename(columns={'global_id':'head_index', 'local_id':'src'}), on='head_index')
            sub_df = sub_df.merge(dst_map.rename(columns={'global_id':'tail_index', 'local_id':'dst'}), on='tail_index')
            
            if USE_GPU:
                edge_index = torch.stack([
                    torch.from_dlpack(sub_df['src'].to_dlpack()), 
                    torch.from_dlpack(sub_df['dst'].to_dlpack())
                ], dim=0)
            else:
                edge_index = torch.tensor(sub_df[['src', 'dst']].values.T, dtype=torch.long)
            
            data[src_type, rel, dst_type].edge_index = edge_index
            if rev_rel:
                data[dst_type, rev_rel, src_type].edge_index = edge_index[[1, 0]]

        # Populate the trio interactions
        _add_edge_type('drug', 'treats', 'disease', 'rev_treats')
        _add_edge_type('gene', 'interacts_with', 'drug', 'rev_interacts')
        _add_edge_type('gene', 'interacts_with', 'gene', 'rev_interacts_gene')

        return data

    def train_dataloader(self):
        return LinkNeighborLoader(
            data=self.train_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=(self.target_edge_type, self.train_data[self.target_edge_type].edge_label_index),
            neg_sampling_ratio=1.0,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return LinkNeighborLoader(
            data=self.val_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=(self.target_edge_type, self.val_data[self.target_edge_type].edge_label_index),
            edge_label=self.val_data[self.target_edge_type].edge_label,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
