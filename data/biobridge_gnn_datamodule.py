import os
import torch
import pytorch_lightning as pl
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit

# Ensure loguru is installed for our logging (standard in your repo)
from loguru import logger 

class BioBridgeGNNDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the BioBridge Heterogeneous Graph using PyTorch Geometric.
    Handles downloading, preprocessing, train/val/test splitting, and 
    efficient subgraph mini-batching for Graph Neural Networks.
    """
    def __init__(self, data_dir: str = 'data/processed/', batch_size: int = 1024, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Defines how deep the network "looks" around a node. 
        # 2 hops deep: 1st hop samples 15 neighbors, 2nd hop samples 10.
        # This allows massive graphs to fit in GPU memory!
        self.num_neighbors = [15, 10] 
        self.target_edge_type = ("drug", "treats", "disease") 

    def prepare_data(self):
        """
        Step 1: Download raw data. 
        Executed only locally on the main process in an HPC environment. 
        """
        # (Assuming the preprocessing from PrimeKG or BioBridge is already run)
        pass 

    def setup(self, stage: str = None):
        """
        Step 2: Understand and format Heterogeneous Graph Structure.
        Executed on EVERY GPU. This builds the graph and splits it.
        """
        graph_path = os.path.join(self.data_dir, 'biobridge_hetero_graph.pt')
        
        if os.path.exists(graph_path):
            self.data = torch.load(graph_path)
            logger.info(f"Loaded BioBridge with {self.data.num_nodes} nodes and {self.data.num_edges} edges.")
        else:
            logger.warning("Graph not found! Creating a mock heterogeneous BioBridge structure for demonstration...")
            self.data = HeteroData()
            self.data['drug'].x = torch.randn(1000, 128)
            self.data['disease'].x = torch.randn(500, 128)
            self.data['gene'].x = torch.randn(5000, 128)
            
            src_drug = torch.randint(0, 1000, (4000,))
            dst_dis = torch.randint(0, 500, (4000,))
            self.data['drug', 'treats', 'disease'].edge_index = torch.stack([src_drug, dst_dis], dim=0)

        # Step 3: Train/val/test splitting strategies & Negative Sampling
        # RandomLinkSplit takes out edges for validation and testing to prevent data leakage.
        transform = RandomLinkSplit(
            num_val=0.1,  
            num_test=0.1,
            disjoint_train_ratio=0.3, # Edges hidden during message passing for training supervision
            edge_types=[self.target_edge_type], 
            rev_edge_types=[("disease", "rev_treats", "drug")] if ("disease", "rev_treats", "drug") in self.data.edge_types else None, 
            add_negative_train_samples=False, # We'll do dynamic negative sampling inside dataloader
            neg_sampling_ratio=1.0 # 1 negative edge for every 1 positive edge
        )
        
        self.train_data, self.val_data, self.test_data = transform(self.data)
        logger.info("Graph splits generated successfully.")

    def train_dataloader(self):
        """
        Step 4: Efficient Data Loading for Large Graphs
        LinkNeighborLoader extracts subgraphs (mini-batches) centered around specific edges.
        """
        edge_label_index = self.train_data[self.target_edge_type].edge_label_index

        return LinkNeighborLoader(
            data=self.train_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=(self.target_edge_type, edge_label_index),
            edge_label=None, # Will auto-generate negatives dynamically if neg_sampling_ratio is set
            neg_sampling_ratio=1.0, # Dynamic negative sampling per epoch!
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        edge_label_index = self.val_data[self.target_edge_type].edge_label_index
        edge_label = self.val_data[self.target_edge_type].edge_label
        
        return LinkNeighborLoader(
            data=self.val_data,
            num_neighbors=self.num_neighbors,
            edge_label_index=(self.target_edge_type, edge_label_index),
            edge_label=edge_label,
            neg_sampling_ratio=0.0, # Validation negatives are fixed from the transform split
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
