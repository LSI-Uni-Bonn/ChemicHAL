"""GNN training utilities for compound selectivity prediction.

Converts SMILES to molecular graphs, featurizes with node/edge attributes,
and provides training loops for PyTorch Geometric GNN models.
"""

from __future__ import annotations

import hashlib
import pickle
from typing import Optional

import joblib
import networkx as nx
from tqdm.auto import tqdm
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
        # PyTorch 2.6 defaults to weights_only=True, which blocks loading
        # arbitrary dataset objects saved by InMemoryDataset processing.
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

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


def _build_dataset_cache_name(prefix: str, smiles: list[str], labels: list[int]) -> str:
    """Build a stable cache key so processed graphs track split content.

    This avoids reusing stale `*_processed.pt` files when switching datasets
    (e.g., multiclass -> binary), which can otherwise produce out-of-range
    labels for the current model head and trigger CUDA device-side asserts.
    """
    hasher = hashlib.sha1()
    for smi, lbl in zip(smiles, labels):
        hasher.update(smi.encode("utf-8", errors="ignore"))
        hasher.update(b"|")
        hasher.update(str(int(lbl)).encode("ascii"))
        hasher.update(b"\n")
    digest = hasher.hexdigest()[:12]
    return f"{prefix}_{len(smiles)}_{digest}"


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
    # Split files are commonly serialized with joblib in this project.
    # Keep pickle support for backwards compatibility.
    split = None
    load_errors: list[str] = []

    try:
        with open(split_file_path, "rb") as f:
            split = pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        load_errors.append(f"pickle: {type(exc).__name__}: {exc}")

    if split is None:
        try:
            split = joblib.load(split_file_path)
        except Exception as exc:  # noqa: BLE001
            load_errors.append(f"joblib: {type(exc).__name__}: {exc}")

    if split is None:
        errors = "; ".join(load_errors)
        raise ValueError(f"Could not load split file '{split_file_path}'. {errors}")

    train_labels = split.get("train_labels")
    test_labels = split.get("test_labels")
    if train_labels is None or test_labels is None:
        raise ValueError(
            "Split file is missing train_labels/test_labels. "
            "Ensure it was produced by split_dataset()."
        )

    train_labels = list(train_labels)
    test_labels = list(test_labels)

    # Normalize labels to contiguous class IDs (e.g. {1,2} -> {0,1}).
    unique_labels = sorted(set(train_labels) | set(test_labels))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    train_labels = [label_to_id[label] for label in train_labels]
    test_labels = [label_to_id[label] for label in test_labels]

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

    # Create datasets with content-derived names so cache files are refreshed
    # when split/dataset content changes.
    train_dataset = SmilesGraphDataset(
        train_smiles,
        train_labels,
        name=_build_dataset_cache_name("train", train_smiles, train_labels),
    )
    val_dataset = SmilesGraphDataset(
        val_smiles,
        val_labels,
        name=_build_dataset_cache_name("val", val_smiles, val_labels),
    )
    test_dataset = SmilesGraphDataset(
        test_smiles,
        test_labels,
        name=_build_dataset_cache_name("test", test_smiles, test_labels),
    )

    return train_dataset, val_dataset, test_dataset


def train_gnn_model(
    split_file_path: str,
    smiles_list: list[str],
    model_class: type = GCN,
    model_save_path: Optional[str] = None,
    node_features_dim: int = 4,
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
    num_classes = len(set(train_dataset.labels) | set(val_dataset.labels) | set(test_dataset.labels))
    if num_classes < 2:
        raise ValueError(
            "Need at least 2 classes for classification training. "
            f"Detected {num_classes} class from split labels."
        )
    model = model_class(
        node_features_dim=node_features_dim,  # atomic_num, formal_charge, num_hs, is_aromatic
        hidden_channels=hidden_channels,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0.0
    for epoch in tqdm(range(epochs)):
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            targets = batch.y.view(-1).long()
            max_target = int(targets.max().item()) if targets.numel() > 0 else -1
            min_target = int(targets.min().item()) if targets.numel() > 0 else 0
            if min_target < 0 or max_target >= out.shape[1]:
                raise ValueError(
                    "Invalid class label detected for CrossEntropyLoss: "
                    f"label range=[{min_target}, {max_target}], "
                    f"model_num_classes={out.shape[1]}. "
                    "This often indicates stale cached graph datasets under data/gnn_dataset/processed."
                )

            loss = criterion(out, targets)
            batch_n = int(targets.numel())
            loss.backward()
            optimizer.step()

            preds = out.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += batch_n
            train_loss_sum += loss.item() * batch_n

        train_loss = train_loss_sum / train_total if train_total > 0 else 0.0
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                targets = batch.y.view(-1).long()
                batch_n = int(targets.numel())
                loss = criterion(out, targets)

                preds = out.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += batch_n
                val_loss_sum += loss.item() * batch_n

        val_loss = val_loss_sum / val_total if val_total > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        # Save best model by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)

        # Log epoch summary via the session logger to avoid writing to stdout.
        try:
            from chemagent.session_utils import get_session_logger as _get_session_logger
            logger = _get_session_logger()
            logger.log_event(
                "gnn_epoch",
                epoch=epoch + 1,
                epochs=epochs,
                train_loss=round(float(train_loss), 6),
                train_acc=round(float(train_acc), 6),
                val_loss=round(float(val_loss), 6),
                val_acc=round(float(val_acc), 6),
            )
        except Exception:
            # Fallback: avoid printing to stdout in MCP server context.
            pass

    # Test evaluation
    model.eval()
    test_acc = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            test_acc += (out.argmax(dim=1) == batch.y.view(-1).long()).sum().item()

    test_acc /= len(test_dataset) if len(test_dataset) > 0 else 1.0

    return {
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "model_path": model_save_path,
    }


def load_gnn_model(model_class: type, node_features_dim: int, hidden_channels: int, num_classes: int, model_path: str, device: Optional[str] = None) -> torch.nn.Module:
    """Load a trained GNN model from a saved state dict.

    Parameters
    ----------
    model_class :
        GNN model class (GCN, GraphSAGE, GAT, etc.).
    model_path :
        Path to saved model state dict.
    device :
        torch device string (default: auto-detect cuda/cpu).
    Returns
    -------
    Loaded GNN model instance with weights from the specified path.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model architecture (must match training configuration)
    model = model_class(
        node_features_dim=node_features_dim,
        hidden_channels=hidden_channels,  # Must match training hidden_channels
        num_classes=num_classes,       # Must match number of classes in training data
    ).to(device)

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model

__all__ = [
    "smiles_to_nx_graph",
    "nx_graph_to_pyg_data",
    "SmilesGraphDataset",
    "load_and_prepare_gnn_dataset",
    "train_gnn_model",
    "load_gnn_model",
]