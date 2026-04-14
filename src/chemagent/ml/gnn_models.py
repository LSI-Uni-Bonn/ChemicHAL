"""Graph neural network model definitions used by chemagent.

This module contains PyTorch Geometric architectures for graph-level
classification and regression.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import (
    GATConv,
    GINConv,
    GINEConv,
    GraphConv,
    SAGEConv,
    GCNConv,
    global_add_pool,
)


def _validate_num_layers(num_layers: int) -> int:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1.")
    return num_layers


def _make_gin_mlp(input_dim: int, hidden_channels: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_channels),
        torch.nn.BatchNorm1d(hidden_channels),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_channels, hidden_channels),
        torch.nn.BatchNorm1d(hidden_channels),
        torch.nn.ReLU(),
    )

class GCN(torch.nn.Module):
    """Graph Convolutional Network with configurable depth.

    Uses `GCNConv` layers with ReLU activations, global add pooling and a
    linear head. Accepts optional `edge_weight` if present in the graph data.
    Default depth is 4 layers.
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 4,
    ) -> None:
        super().__init__()

        num_layers = _validate_num_layers(num_layers)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(node_features_dim, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin = Linear(hidden_channels, num_classes)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        x = x.float()
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if i < len(self.convs) - 1:
                x = F.relu(x)

        x = global_add_pool(x, batch)

        x = F.dropout(x, training=self.training)
        x = self.lin(x)

        return x

class GraphSAGE(torch.nn.Module):
    """GraphSAGE encoder with configurable depth and mean aggregation.

    Applies stacked SAGEConv layers, global add pooling, and a linear head.
    Default depth is 4 layers.
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 4,
    ) -> None:
        super().__init__()

        num_layers = _validate_num_layers(num_layers)
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(node_features_dim, hidden_channels, aggr="mean"))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr="mean"))
        self.lin = Linear(hidden_channels, num_classes)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        del edge_weight  # Not used by SAGEConv.

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        
        x = global_add_pool(x, batch)
        
        x = F.dropout(x, training=self.training)
        x = self.lin(x)

        return x


class GC_GNN(torch.nn.Module):
    """GraphConv-based network with max aggregation and configurable depth.

    Uses edge weights when provided, then performs graph-level pooling and a
    linear projection to produce output logits. Default depth is 4 layers.
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 4,
    ):
        super().__init__()

        num_layers = _validate_num_layers(num_layers)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConv(node_features_dim, hidden_channels, aggr="max"))
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_channels, hidden_channels, aggr="max"))
        self.lin = Linear(hidden_channels, num_classes)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        
        x = global_add_pool(x, batch)
        
        x = F.dropout(x, training=self.training)
        x = self.lin(x)

        return x


class GINE(torch.nn.Module):
    """GINE model with edge-aware message passing and configurable depth.

    Applies stacked GINEConv blocks whose internal MLPs include BatchNorm and
    ReLU. Requires edge weights to build edge attributes for convolution.
    Default depth is 4 layers.
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_channels: int,
        num_classes: int,
        edge_emb_dim: int = 1,
        num_layers: int = 4,
    ):
        super().__init__()

        num_layers = _validate_num_layers(num_layers)
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GINEConv(_make_gin_mlp(node_features_dim, hidden_channels), edge_dim=edge_emb_dim)
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                GINEConv(_make_gin_mlp(hidden_channels, hidden_channels), edge_dim=edge_emb_dim)
            )

        self.lin = Linear(hidden_channels, num_classes)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        if edge_weight is None:
            raise ValueError("GINE requires edge_weight for edge features.")

        edge_attr = torch.unsqueeze(edge_weight, dim=1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        
        x = global_add_pool(x, batch)
        
        x = F.dropout(x, training=self.training)

        x = self.lin(x)

        return x


class GIN(torch.nn.Module):
    """GIN model for graph-level prediction with configurable depth.

    Uses stacked GINConv blocks with MLP updates, followed by global add
    pooling and a final linear classifier/regressor head. Default depth is 4.
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_channels: int,
        num_classes: int,
        edge_emb_dim: int = 1,
        num_layers: int = 4,
    ):
        super().__init__()

        del edge_emb_dim  # Kept for API compatibility.

        num_layers = _validate_num_layers(num_layers)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(_make_gin_mlp(node_features_dim, hidden_channels)))
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(_make_gin_mlp(hidden_channels, hidden_channels)))

        self.lin = Linear(hidden_channels, num_classes)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        del edge_weight  # Not used by GINConv.

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
    
        x = global_add_pool(x, batch)
        
        x = F.dropout(x, training=self.training)

        x = self.lin(x)

        return x


class GAT(torch.nn.Module):
    """Graph Attention Network for graph-level prediction.

    Stacks GATConv layers that can consume scalar edge attributes, aggregates
    node embeddings with global add pooling, and predicts outputs through a
    linear head. Default depth is 4 layers.
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int = 4,
    ):
        super().__init__()

        num_layers = _validate_num_layers(num_layers)
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(node_features_dim, hidden_channels, edge_dim=1))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels, edge_dim=1))
        self.lin = Linear(hidden_channels, num_classes)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_weight)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        
        x = global_add_pool(x, batch)
        
        x = F.dropout(x, training=self.training)
        x = self.lin(x)

        return x


__all__ = ["GCN", "GraphSAGE", "GC_GNN", "GINE", "GIN", "GAT"]
