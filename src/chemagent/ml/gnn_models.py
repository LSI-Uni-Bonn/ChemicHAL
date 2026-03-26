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

class GCN(torch.nn.Module):
    """4-layer Graph Convolutional Network.

    Uses `GCNConv` layers with ReLU activations, global add pooling and a
    linear head. Accepts optional `edge_weight` if present in the graph data.
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.conv1 = GCNConv(node_features_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        x = F.relu(self.conv1(x.float(), edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight=edge_weight))
        x = self.conv4(x, edge_index, edge_weight=edge_weight)

        x = global_add_pool(x, batch)

        x = F.dropout(x, training=self.training)
        x = self.lin(x)

        return x

class GraphSAGE(torch.nn.Module):
    """GraphSAGE encoder with mean neighborhood aggregation.

    Stacks four SAGEConv layers, applies global add pooling, and maps the
    pooled graph embedding to class logits.
    """

    def __init__(
        self, node_features_dim: int, hidden_channels: int, num_classes: int
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(node_features_dim, hidden_channels, aggr="mean")
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.conv4 = SAGEConv(hidden_channels, hidden_channels, aggr="mean")
        self.lin = Linear(hidden_channels, num_classes)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        del edge_weight  # Not used by SAGEConv.

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        
        x = global_add_pool(x, batch)
        
        x = F.dropout(x, training=self.training)
        x = self.lin(x)

        return x


class GC_GNN(torch.nn.Module):
    """GraphConv-based network with max aggregation.

    Uses edge weights when provided, then performs graph-level pooling and a
    linear projection to produce output logits.
    """

    def __init__(self, node_features_dim: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = GraphConv(node_features_dim, hidden_channels, aggr="max")
        self.conv2 = GraphConv(hidden_channels, hidden_channels, aggr="max")
        self.conv3 = GraphConv(hidden_channels, hidden_channels, aggr="max")
        self.conv4 = GraphConv(hidden_channels, hidden_channels, aggr="max")
        self.lin = Linear(hidden_channels, num_classes)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:

        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight=edge_weight))
        x = self.conv4(x, edge_index, edge_weight=edge_weight)
        
        x = global_add_pool(x, batch)
        
        x = F.dropout(x, training=self.training)
        x = self.lin(x)

        return x


class GINE(torch.nn.Module):
    """GINE model with edge-aware message passing.

    Applies four GINEConv blocks whose internal MLPs include BatchNorm and
    ReLU. Requires edge weights to build edge attributes for convolution.
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_channels: int,
        num_classes: int,
        edge_emb_dim: int = 1,
    ):
        super().__init__()

        self.conv1 = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(node_features_dim, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
            ),
            edge_dim=edge_emb_dim,
        )

        self.conv2 = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
            ),
            edge_dim=edge_emb_dim,
        )

        self.conv3 = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
            ),
            edge_dim=edge_emb_dim,
        )

        self.conv4 = GINEConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
            ),
            edge_dim=edge_emb_dim,
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

        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr=edge_attr))
        x = self.conv4(x, edge_index, edge_attr=edge_attr)
        
        x = global_add_pool(x, batch)
        
        x = F.dropout(x, training=self.training)

        x = self.lin(x)

        return x


class GIN(torch.nn.Module):
    """GIN model for graph-level prediction without edge attributes.

    Uses four GINConv blocks with MLP updates, followed by global add pooling
    and a final linear classifier/regressor head.
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_channels: int,
        num_classes: int,
        edge_emb_dim: int = 1,
    ):
        super().__init__()

        del edge_emb_dim  # Kept for API compatibility.

        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(node_features_dim, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
            )
        )

        self.conv2 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
            )
        )

        self.conv3 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
            )
        )

        self.conv4 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.BatchNorm1d(hidden_channels),
                torch.nn.ReLU(),
            )
        )

        self.lin = Linear(hidden_channels, num_classes)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        del edge_weight  # Not used by GINConv.

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
    
        x = global_add_pool(x, batch)
        
        x = F.dropout(x, training=self.training)

        x = self.lin(x)

        return x


class GAT(torch.nn.Module):
    """Graph Attention Network for graph-level prediction.
    Stacks four GATConv layers that can consume scalar edge attributes,
    aggregates node embeddings with global add pooling, and predicts outputs
    through a linear head.
    """

    def __init__(self, node_features_dim: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = GATConv(node_features_dim, hidden_channels, edge_dim=1)
        self.conv2 = GATConv(hidden_channels, hidden_channels, edge_dim=1)
        self.conv3 = GATConv(hidden_channels, hidden_channels, edge_dim=1)
        self.conv4 = GATConv(hidden_channels, hidden_channels, edge_dim=1)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:

        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_attr=edge_weight))
        x = self.conv4(x, edge_index, edge_attr=edge_weight)
        
        x = global_add_pool(x, batch)
        
        x = F.dropout(x, training=self.training)
        x = self.lin(x)

        return x


__all__ = ["GCN", "GraphSAGE", "GC_GNN", "GINE", "GIN", "GAT"]
