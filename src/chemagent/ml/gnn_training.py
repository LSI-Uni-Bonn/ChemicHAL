"""GNN training utilities for compound selectivity prediction.

Converts SMILES to molecular graphs, featurizes with node/edge attributes,
and provides training loops for PyTorch Geometric GNN models.
"""

from __future__ import annotations

import copy
import hashlib
import pickle
from typing import Optional

import joblib
import networkx as nx
import numpy as np
from tqdm.auto import tqdm
import torch
from rdkit import Chem
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from chemagent.ml.gnn_models import GCN, GAT, GC_GNN, GIN, GINE, GraphSAGE
from chemagent.ml.metrics import classification_metrics, multiclass_metrics



def smiles_to_nx_graph(smiles: str) -> Optional[nx.Graph]:
    """Convert SMILES string to NetworkX graph with node/edge features.

    Args:
    smiles :
        SMILES string.

    Returns:
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

    Args:
    nx_graph :
        NetworkX graph with node/edge attributes, or None.
    label :
        Class label (0 or 1 for binary selectivity).

    Returns:
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

    Attributes:
    smiles_list : list
        List of SMILES strings.
    labels : list
        List of binary class labels.
    name : str
        Dataset identifier for caching.
    """

    CACHE_SCHEMA_VERSION = "v2"

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
        loaded = torch.load(self.processed_paths[0], weights_only=False)
        self.metadata: dict = {}

        if isinstance(loaded, tuple) and len(loaded) >= 2:
            self.data, self.slices = loaded[0], loaded[1]
            if len(loaded) >= 3 and isinstance(loaded[2], dict):
                self.metadata = loaded[2]
        else:
            raise ValueError(
                f"Unsupported processed dataset format in {self.processed_paths[0]}"
            )

        if isinstance(self.metadata.get("smiles_list"), list):
            self.smiles_list = list(self.metadata["smiles_list"])
        if isinstance(self.metadata.get("labels"), list):
            self.labels = list(self.metadata["labels"])

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{self.name}_{self.CACHE_SCHEMA_VERSION}_processed.pt"]

    def process(self) -> None:
        """Convert SMILES to PyG Data objects and cache."""
        data_list = []
        kept_smiles: list[str] = []
        kept_labels: list[int] = []

        for smiles, label in zip(self.smiles_list, self.labels):
            nx_graph = smiles_to_nx_graph(smiles)
            pyg_data = nx_graph_to_pyg_data(nx_graph, label)

            if pyg_data is not None:
                data_list.append(pyg_data)
                kept_smiles.append(smiles)
                kept_labels.append(int(label))

        data, slices = self.collate(data_list)
        metadata = {
            "smiles_list": kept_smiles,
            "labels": kept_labels,
            "dataset_name": self.name,
        }
        torch.save((data, slices, metadata), self.processed_paths[0])


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

    Args:
    split_file_path :
        Path to .pkl split file with train/test indices and labels.
    smiles_list :
        Optional full list of SMILES strings (parallel to split indices).
        Required only for index-based split files.
    test_size :
        Fraction for validation split from training data (default 0.2).
    seed :
        Random seed for reproducibility (default 42).

    Returns:
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
    num_layers: int = 4,
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> dict:
    """Train a GNN model on graph-structured selectivity data.

    Args:
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
    num_layers :
        Number of message-passing layers in the GNN backbone (default 4).
    epochs :
        Number of training epochs (default 100).
    lr :
        Learning rate (default 0.001).
    batch_size :
        Batch size for training and evaluation (default 32).
    device :
        torch device string (default: auto-detect cuda/cpu).

    Returns:
    Dictionary with keys:
        - "best_val_acc": Best validation accuracy across epochs.
        - "test_acc": Final test accuracy.
        - "train_evaluation": Train metrics (same family as tabular models).
        - "val_evaluation": Validation metrics.
        - "test_evaluation": Test metrics.
        - "model_path": Path where model was saved (or None).
    """

    def _evaluate_loader(loader: DataLoader) -> dict:
        labels_all: list[int] = []
        preds_all: list[int] = []
        probs_all: list[list[float]] = []

        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)

                labels_all.extend(batch.y.view(-1).long().detach().cpu().tolist())
                preds_all.extend(preds.detach().cpu().tolist())
                probs_all.extend(probs.detach().cpu().tolist())

        if not labels_all:
            return {"status": "empty", "n_samples": 0}

        labels_np = np.array(labels_all)
        preds_np = np.array(preds_all)
        probs_np = np.array(probs_all)
        n_classes = len(np.unique(labels_np))

        if n_classes == 2:
            return classification_metrics(
                labels=labels_np,
                pred=preds_np,
                y_proba=probs_np,
                model_id=getattr(model_class, "__name__", "GNN"),
                model_type="gnn",
            )

        return multiclass_metrics(
            labels=labels_np,
            pred=preds_np,
            model_id=getattr(model_class, "__name__", "GNN"),
            model_type="gnn",
        )

    def _extract_accuracy(eval_result: dict) -> float:
        if "Accuracy" in eval_result:
            return float(eval_result["Accuracy"])
        if "overall_metrics" in eval_result and isinstance(eval_result["overall_metrics"], dict):
            if "Accuracy" in eval_result["overall_metrics"]:
                return float(eval_result["overall_metrics"]["Accuracy"])
        return 0.0

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
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_state_dict = None

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
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "model_class_name": getattr(model_class, "__name__", str(model_class)),
                        "node_features_dim": int(node_features_dim),
                        "hidden_channels": int(hidden_channels),
                        "num_classes": int(num_classes),
                        "num_layers": int(num_layers),
                    },
                    model_save_path,
                )
            best_state_dict = copy.deepcopy(model.state_dict())

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

    # Full split metrics from the best validation checkpoint, not the last epoch.
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    train_evaluation = _evaluate_loader(train_loader)
    val_evaluation = _evaluate_loader(val_loader)
    test_evaluation = _evaluate_loader(test_loader)

    # Preserve backward-compatible scalar accuracy fields.
    test_acc = _extract_accuracy(test_evaluation)

    return {
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "train_evaluation": train_evaluation,
        "val_evaluation": val_evaluation,
        "test_evaluation": test_evaluation,
        "n_train": int(len(train_dataset)),
        "n_val": int(len(val_dataset)),
        "n_test": int(len(test_dataset)),
        "model_path": model_save_path,
    }


def load_gnn_model(
    model_class: type,
    node_features_dim: int,
    hidden_channels: int,
    num_classes: int,
    model_path: str,
    device: Optional[str] = None,
    num_layers: int = 4,
) -> torch.nn.Module:
    """Load a trained GNN model from a saved state dict.

    Args:
    model_class :
        GNN model class (GCN, GraphSAGE, GAT, etc.).
    model_path :
        Path to saved model state dict.
    num_layers :
        Number of message-passing layers for raw state_dict loads (default 4).
        Ignored when checkpoint metadata contains ``num_layers``.
    device :
        torch device string (default: auto-detect cuda/cpu).
    Returns:
    Loaded GNN model instance with weights from the specified path.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Support both raw state_dict and checkpoint dictionaries with metadata.
    checkpoint_or_state = torch.load(model_path, map_location=device)
    if isinstance(checkpoint_or_state, torch.nn.Module):
        model = checkpoint_or_state.to(device)
        model.eval()
        return model

    if isinstance(checkpoint_or_state, dict) and "state_dict" in checkpoint_or_state:
        state_dict = checkpoint_or_state["state_dict"]
        hidden_channels = int(checkpoint_or_state.get("hidden_channels", hidden_channels))
        num_classes = int(checkpoint_or_state.get("num_classes", num_classes))
        node_features_dim = int(checkpoint_or_state.get("node_features_dim", node_features_dim))
        num_layers = int(checkpoint_or_state.get("num_layers", num_layers))
    else:
        state_dict = checkpoint_or_state

    # Initialize model architecture (must match training configuration)
    model = model_class(
        node_features_dim=node_features_dim,
        hidden_channels=hidden_channels,  # Must match training hidden_channels
        num_classes=num_classes,       # Must match number of classes in training data
        num_layers=num_layers,
    ).to(device)

    # Load state dict
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