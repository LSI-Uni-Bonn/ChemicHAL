"""GNN training utilities for compound selectivity prediction.

Converts SMILES to molecular graphs, featurizes with node/edge attributes,
and provides training loops for PyTorch Geometric GNN models.
"""

from __future__ import annotations

import pickle
from typing import Optional

import networkx as nx
import torch
from rdkit import Chem
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from chemagent.ml.gnn_models import GCN, GAT, GC_GNN, GIN, GINE, GraphSAGE



def smiles_to_nx_graph(smiles: str) -> Optional[nx.Graph]:
    """Convert SMILES string to NetworkX graph with node/edge features.

    Parameters
    ----------
    smiles :
        SMILES string.

    Returns
    -------
    NetworkX graph with atomic and bond features, or None if parsing fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    G = nx.Graph()

    # Add nodes with atomic features
    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            atomic_num=atom.GetAtomicNum(),
            formal_charge=atom.GetFormalCharge(),
            num_hs=atom.GetTotalNumHs(),
            is_aromatic=atom.GetIsAromatic(),
        )

    # Add edges
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_type=int(bond.GetBondType()),
        )

    return G


def nx_graph_to_pyg_data(nx_graph: Optional[nx.Graph], label: int) -> Optional[Data]:
    """Convert NetworkX graph to PyTorch Geometric Data object.

    Parameters
    ----------
    nx_graph :
        NetworkX graph with node/edge attributes, or None.
    label :
        Class label (0 or 1 for binary selectivity).

    Returns
    -------
    PyTorch Geometric Data object, or None if graph is invalid.
    """
    if nx_graph is None or len(nx_graph) == 0:
        return None

    # Node features: [atomic_num/100, formal_charge, num_hs/4, is_aromatic]
    node_features = []
    node_map = {node: idx for idx, node in enumerate(nx_graph.nodes())}

    for node in nx_graph.nodes():
        attrs = nx_graph.nodes[node]
        features = [
            attrs.get("atomic_num", 0) / 100.0,
            attrs.get("formal_charge", 0),
            attrs.get("num_hs", 0) / 4.0,
            float(attrs.get("is_aromatic", False)),
        ]
        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)

    # Edge indices (undirected)
    edge_list = []
    for u, v in nx_graph.edges():
        edge_list.append([node_map[u], node_map[v]])
        edge_list.append([node_map[v], node_map[u]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


class SmilesGraphDataset(InMemoryDataset):
    """PyTorch Geometric in-memory dataset from SMILES and labels.

    Attributes
    ----------
    smiles_list : list
        List of SMILES strings.
    labels : list
        List of binary class labels.
    name : str
        Dataset identifier for caching.
    """

    def __init__(
        self,
        smiles_list: list[str],
        labels: list[int],
        root: str = "./data/gnn_dataset",
        name: str = "smiles_graphs",
    ) -> None:
        self.smiles_list = smiles_list
        self.labels = labels
        self.name = name
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{self.name}_processed.pt"]

    def process(self) -> None:
        """Convert SMILES to PyG Data objects and cache."""
        data_list = []

        for smiles, label in zip(self.smiles_list, self.labels):
            nx_graph = smiles_to_nx_graph(smiles)
            pyg_data = nx_graph_to_pyg_data(nx_graph, label)

            if pyg_data is not None:
                data_list.append(pyg_data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_and_prepare_gnn_dataset(
    split_file_path: str,
    smiles_list: Optional[list[str]] = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[SmilesGraphDataset, SmilesGraphDataset, SmilesGraphDataset]:
    """Load split file and create train/val/test GNN datasets.

    Parameters
    ----------
    split_file_path :
        Path to .pkl split file with train/test indices and labels.
    smiles_list :
        Optional full list of SMILES strings (parallel to split indices).
        Required only for index-based split files.
    test_size :
        Fraction for validation split from training data (default 0.2).
    seed :
        Random seed for reproducibility (default 42).

    Returns
    -------
    Tuple of (train_dataset, val_dataset, test_dataset).
    """
    with open(split_file_path, "rb") as f:
        split = pickle.load(f)

    train_labels = split.get("train_labels")
    test_labels = split.get("test_labels")
    if train_labels is None or test_labels is None:
        raise ValueError(
            "Split file is missing train_labels/test_labels. "
            "Ensure it was produced by split_dataset()."
        )

    train_labels = list(train_labels)
    test_labels = list(test_labels)

    # Preferred schema: split file already includes split SMILES arrays.
    if "train_smiles" in split and "test_smiles" in split:
        train_smiles = list(split["train_smiles"])
        test_smiles = list(split["test_smiles"])
    else:
        # Fallback schema: split file includes indices only.
        train_idx = split.get("train_idx")
        test_idx = split.get("test_idx")
        if train_idx is None or test_idx is None:
            raise ValueError(
                "Split file has labels but no train_smiles/test_smiles or train_idx/test_idx."
            )
        if smiles_list is None:
            raise ValueError(
                "smiles_list is required for index-based split files."
            )
        train_smiles = [smiles_list[i] for i in train_idx]
        test_smiles = [smiles_list[i] for i in test_idx]

    # Split training into train/val
    train_smiles, val_smiles, train_labels, val_labels = train_test_split(
        train_smiles,
        train_labels,
        test_size=test_size,
        random_state=seed,
        stratify=train_labels,
    )

    # Create datasets
    train_dataset = SmilesGraphDataset(
        train_smiles,
        train_labels,
        name="train",
    )
    val_dataset = SmilesGraphDataset(
        val_smiles,
        val_labels,
        name="val",
    )
    test_dataset = SmilesGraphDataset(
        test_smiles,
        test_labels,
        name="test",
    )

    return train_dataset, val_dataset, test_dataset


def train_gnn_model(
    split_file_path: str,
    smiles_list: list[str],
    model_class: type = GCN,
    model_save_path: Optional[str] = None,
    hidden_channels: int = 64,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> dict:
    """Train a GNN model on graph-structured selectivity data.

    Parameters
    ----------
    split_file_path :
        Path to .pkl split file with train/test indices and labels.
    smiles_list :
        Full list of SMILES strings (parallel to split indices).
    model_class :
        GNN model class (GCN, GraphSAGE, GAT, etc., default GCN).
    model_save_path :
        Optional path to save best model state dict.
    hidden_channels :
        Hidden dimension for all GNN layers (default 64).
    epochs :
        Number of training epochs (default 100).
    lr :
        Learning rate (default 0.001).
    batch_size :
        Batch size for training and evaluation (default 32).
    device :
        torch device string (default: auto-detect cuda/cpu).

    Returns
    -------
    Dictionary with keys:
        - "best_val_acc": Best validation accuracy across epochs.
        - "test_acc": Final test accuracy.
        - "model_path": Path where model was saved (or None).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load datasets
    train_dataset, val_dataset, test_dataset = load_and_prepare_gnn_dataset(
        split_file_path,
        smiles_list,
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    model = model_class(
        node_features_dim=4,  # atomic_num, formal_charge, num_hs, is_aromatic
        hidden_channels=hidden_channels,
        num_classes=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                val_acc += (out.argmax(dim=1) == batch.y.squeeze()).sum().item()

        val_acc /= len(val_dataset) if len(val_dataset) > 0 else 1.0
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)

    # Test evaluation
    model.eval()
    test_acc = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            test_acc += (out.argmax(dim=1) == batch.y.squeeze()).sum().item()

    test_acc /= len(test_dataset) if len(test_dataset) > 0 else 1.0

    return {
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "model_path": model_save_path,
    }


__all__ = [
    "smiles_to_nx_graph",
    "nx_graph_to_pyg_data",
    "SmilesGraphDataset",
    "load_and_prepare_gnn_dataset",
    "train_gnn_model",
]