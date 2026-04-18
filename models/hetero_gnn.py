import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import SAGEConv, to_hetero
import torchmetrics
from models.biobridge_encoder import BioBridgeProjector

class HeteroGNN(torch.nn.Module):
    """
    Standard Message Passing GNN. 
    By default this handles homogeneous graphs, but we will "heterogenize" it
    dynamically using PyG's `to_hetero` function.
    """
    def __init__(self, hidden_channels: int = 64, num_layers: int = 2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relus = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        
        # Note: input channel is -1 for lazy initialization (infers dynamically from data)
        for _ in range(num_layers):
            self.convs.append(SAGEConv((-1, -1), hidden_channels))
            self.relus.append(torch.nn.ReLU())
            self.dropouts.append(torch.nn.Dropout(p=0.2))

    def forward(self, x, edge_index):
        # We loop through our layers and apply ReLU activations 
        # Using module-based ReLU/Dropout is safer for to_hetero FX tracing
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.relus[i](x)
            x = self.dropouts[i](x)
            
        # Last layer (no activation or dropout usually for latent embeddings)
        return self.convs[-1](x, edge_index)


class LinkPredictionDecoder(torch.nn.Module):
    """
    Takes the embeddings of two nodes computed by the GNN 
    and predicts if an edge exists between them (dot product).
    """
    def forward(self, z_dict, edge_label_index, target_edge_type):
        src_type, _, dst_type = target_edge_type
        
        # Get the node embeddings for source and destination types
        # z_dict is output by the GNN
        z_src = z_dict[src_type]
        z_dst = z_dict[dst_type]
        
        # Extract the specific node indices involved in the target edges
        src_indices = edge_label_index[0]
        dst_indices = edge_label_index[1]
        
        # Perform dot product between corresponding source and destination embeddings
        # This outputs a single scalar score per edge
        return (z_src[src_indices] * z_dst[dst_indices]).sum(dim=-1)


class BioBridgeLinkPredictor(pl.LightningModule):
    """
    PyTorch Lightning module tying together the GNN encoder, 
    the link prediction decoder, and the training loop dynamics.
    """
    def __init__(self, metadata: tuple, hidden_channels: int = 64, lr: float = 1e-3, target_edge_type: tuple = ("drug", "treats", "disease")):
        super().__init__()
        # Save hyperparams for wandb logging
        self.save_hyperparameters()
        self.target_edge_type = target_edge_type
        
        # 1. BioBridge Modal Encoder (Projects real ESM-2b/PubMedBERT vectors)
        self.projector = BioBridgeProjector(target_dim=hidden_channels)
        
        # 2. Instantiate the Base GNN
        base_gnn = HeteroGNN(hidden_channels=hidden_channels, num_layers=2)
        
        # 2. "Heterogenize" it: This magic function creates a separate unique
        # neural network layer for EVERY edge type and node type defined in `metadata`
        self.encoder = to_hetero(base_gnn, metadata, aggr='sum')
        
        self.decoder = LinkPredictionDecoder()
        
        # Metrics setup
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_auroc = torchmetrics.AUROC(task="binary")

    def forward(self, batch):
        # 1. Device Synchronization (CRITICAL for full-graph inference)
        # Ensure the graph/batch is on the same device as the model (GPU or CPU)
        batch = batch.to(self.device)

        # 2. Multimodal Projection (BioBridge)
        x_dict = {}
        for node_type in batch.node_types:
            if hasattr(batch[node_type], 'n_id'):
                indices = batch[node_type].n_id
            else:
                num_nodes = batch[node_type].num_nodes
                indices = torch.arange(num_nodes, device=self.device)
            
            x_dict[node_type] = self.projector.forward_for_type(node_type, indices)
            
        # 3. Pass into the HeteroGNN
        z_dict = self.encoder(x_dict, batch.edge_index_dict)
        return z_dict

    def _step(self, batch, batch_idx, mode="train"):
        # 1. Forward pass: Calculate embeddings
        z_dict = self(batch)
        
        # 2. Decode the target edges
        edge_label_index = batch[self.target_edge_type].edge_label_index
        # This gives us logit scores for every edge in the mini-batch (both real and negative samples)
        logits = self.decoder(z_dict, edge_label_index, self.target_edge_type)
        
        # 3. Calculate Loss
        labels = batch[self.target_edge_type].edge_label
        # Binary Cross Entropy with Logits Loss
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        # 4. Calculate metrics
        preds = torch.sigmoid(logits)
        
        if mode == "train":
            self.train_acc(preds, labels)
            self.log("train_loss", loss, batch_size=edge_label_index.size(1), prog_bar=True)
            self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        else:
            self.val_acc(preds, labels)
            self.val_auroc(preds, labels)
            self.log("val_loss", loss, batch_size=edge_label_index.size(1), prog_bar=True, sync_dist=True)
            self.log("val_acc", self.val_acc, sync_dist=True)
            self.log("val_auroc", self.val_auroc, sync_dist=True)
            
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="val")

    def configure_optimizers(self):
        # AdamW is robust for large-scale graphs
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        # StepLR scheduler often helps GNNs converge better
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]
