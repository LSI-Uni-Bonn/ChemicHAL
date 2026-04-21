"""Integration test: train all GNN model types on a small split, then run MolAnchor.

Workflow per model:
  1. Build a small synthetic split (20 SMILES, binary labels)
  2. Train each model for a few epochs using PyG DataLoaders
  3. Pick one compound from the test set
  4. Run MolAnchor (fragment → subgraph → GNN inference → anchor identification)
  5. Assert the result has the expected structure

GINE is the only architecture that *requires* edge weights during inference.
All models are trained here with bond-type edge weights so a single training
loop covers every architecture cleanly.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from rdkit import Chem
from sklearn.model_selection import train_test_split as sk_split
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from chemagent.explainability.MolAnchor.MolAnchor import MolecularAnchor
from chemagent.ml.gnn_models import GAT, GC_GNN, GCN, GIN, GINE, GraphSAGE

# ---------------------------------------------------------------------------
# Synthetic dataset — 20 drug-like SMILES with two classes
# ---------------------------------------------------------------------------

_SMILES_LABELS = [
    ("CC(=O)Oc1ccccc1C(=O)O", 0),        # aspirin
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", 0),   # ibuprofen
    ("CC(=O)Nc1ccc(O)cc1", 0),            # acetaminophen
    ("Clc1ccccc1NC(=O)c1ccccc1", 0),
    ("CC(=O)Nc1ccc(Cl)cc1", 0),
    ("Cc1ccc(NC(=O)c2ccccc2)cc1", 0),
    ("O=C(O)c1ccccc1NC(=O)c1ccccc1", 0),
    ("COc1ccc(CC(N)C(=O)O)cc1", 0),
    ("CCOC(=O)c1ccc(N)cc1", 0),
    ("Cc1ccc(S(=O)(=O)Nc2ccccn2)cc1", 0),
    ("Cn1c(=O)c2c(ncn2C)n(C)c1=O", 1),   # caffeine
    ("CC(C)NCC(O)c1ccc(O)c(O)c1", 1),    # epinephrine-like
    ("c1ccc(-c2ccccn2)nc1", 1),
    ("CC1=CC(=O)c2ccccc2C1=O", 1),
    ("O=C(NNc1ccccc1)c1ccncc1", 1),
    ("CCOC(=O)c1cccc(NC(=O)OCC)c1", 1),
    ("O=C1CCCN1c1ccc(F)cc1", 1),
    ("Cc1ccc(cc1)C(=O)NN", 1),
    ("CCOC(=O)c1ccc(NC(=O)c2ccccc2)cc1", 1),
    ("CC(=O)Nc1ccc(NC(=O)c2ccccc2)cc1", 1),
]

ALL_SMILES = [s for s, _ in _SMILES_LABELS]
ALL_LABELS = [l for _, l in _SMILES_LABELS]

# Fixed 16/4 train/test split by index
_TRAIN_IDX = list(range(16))
_TEST_IDX = list(range(16, 20))

TRAIN_SMILES = [ALL_SMILES[i] for i in _TRAIN_IDX]
TRAIN_LABELS = [ALL_LABELS[i] for i in _TRAIN_IDX]
TEST_SMILES = [ALL_SMILES[i] for i in _TEST_IDX]
TEST_LABELS = [ALL_LABELS[i] for i in _TEST_IDX]

# GNN hyperparameters kept tiny for fast testing
NODE_FEATURES_DIM = 4
HIDDEN_CHANNELS = 16
NUM_CLASSES = 2
EPOCHS = 5
BATCH_SIZE = 8
LR = 1e-3

MODEL_CLASSES = [GCN, GraphSAGE, GIN, GC_GNN, GAT, GINE]
MODEL_IDS = ["GCN", "GraphSAGE", "GIN", "GC_GNN", "GAT", "GINE"]


# ---------------------------------------------------------------------------
# PyG data helpers (include bond-type edge weights so every model works)
# ---------------------------------------------------------------------------

def _smiles_to_pyg(smiles: str, label: int) -> Data | None:
    """Convert SMILES to a PyG Data object with 4 node features + bond-weight edges."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([
            atom.GetAtomicNum() / 100.0,
            float(atom.GetFormalCharge()),
            atom.GetTotalNumHs() / 4.0,
            float(atom.GetIsAromatic()),
        ])
    x = torch.tensor(node_features, dtype=torch.float)

    edge_list, ew_list = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        w = bond.GetBondTypeAsDouble() / 3.0   # normalize: 1→0.33, 1.5→0.5, 2→0.67, 3→1.0
        edge_list += [[i, j], [j, i]]
        ew_list += [w, w]

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(ew_list, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                y=torch.tensor([label], dtype=torch.long))


def _build_dataset(smiles_list: list[str], labels: list[int]) -> list[Data]:
    """Convert a list of SMILES/labels to PyG Data objects, dropping failures."""
    return [d for s, l in zip(smiles_list, labels) if (d := _smiles_to_pyg(s, l)) is not None]


# ---------------------------------------------------------------------------
# Minimal training loop (shared by all architectures)
# ---------------------------------------------------------------------------

def _train(
    model: torch.nn.Module,
    train_data: list[Data],
    val_data: list[Data],
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
) -> torch.nn.Module:
    """Train model for `epochs` epochs and return best-val-acc checkpoint."""
    device = "cpu"
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    best_val_acc = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            ew = getattr(batch, "edge_weight", None)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch, edge_weight=ew)
            targets = batch.y.view(-1).long()
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                ew = getattr(batch, "edge_weight", None)
                logits = model(batch.x, batch.edge_index, batch.batch, edge_weight=ew)
                preds = logits.argmax(dim=1)
                correct += (preds == batch.y.view(-1).long()).sum().item()
                total += batch.y.numel()

        val_acc = correct / total if total > 0 else 0.0
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# MolAnchor GNN adapters (handles all architectures uniformly)
# ---------------------------------------------------------------------------

def gnn_graph_func(mol: Chem.Mol) -> nx.Graph:
    """Convert RDKit mol → NetworkX graph with 4-dim node features + bond weights."""
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            atomic_num=atom.GetAtomicNum() / 100.0,
            formal_charge=float(atom.GetFormalCharge()),
            num_hs=atom.GetTotalNumHs() / 4.0,
            is_aromatic=float(atom.GetIsAromatic()),
        )
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_weight=bond.GetBondTypeAsDouble() / 3.0,
        )
    return G


def gnn_graph_predict(model: torch.nn.Module, frag_graphs: list[nx.Graph]) -> np.ndarray:
    """Run GNN inference on NetworkX fragment subgraphs.

    Extracts node features and bond-type edge weights, batches with PyG,
    and returns class predictions as a numpy int array.
    Compatible with all architectures (GCN, GraphSAGE, GIN, GC_GNN, GAT, GINE).
    """
    data_list: list[Data] = []

    for g in frag_graphs:
        nodes = list(g.nodes())
        if not nodes:
            data_list.append(Data(
                x=torch.zeros((1, NODE_FEATURES_DIM), dtype=torch.float),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_weight=torch.zeros(0, dtype=torch.float),
            ))
            continue

        node_map = {n: i for i, n in enumerate(nodes)}
        x = torch.tensor(
            [[g.nodes[n].get("atomic_num", 0.), g.nodes[n].get("formal_charge", 0.),
              g.nodes[n].get("num_hs", 0.), g.nodes[n].get("is_aromatic", 0.)]
             for n in nodes],
            dtype=torch.float,
        )

        edge_list, ew_list = [], []
        for u, v, attrs in g.edges(data=True):
            ui, vi = node_map[u], node_map[v]
            w = float(attrs.get("bond_weight", 1.0 / 3.0))
            edge_list.append([ui, vi])
            ew_list.append(w)
            if u != v:  # self-loops added by MolAnchor — don't duplicate
                edge_list.append([vi, ui])
                ew_list.append(w)

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_weight = torch.tensor(ew_list, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weight = torch.zeros(0, dtype=torch.float)

        data_list.append(Data(x=x, edge_index=edge_index, edge_weight=edge_weight))

    batch = Batch.from_data_list(data_list)
    ew = getattr(batch, "edge_weight", None)

    model.eval()
    with torch.no_grad():
        logits = model(batch.x, batch.edge_index, batch.batch, edge_weight=ew)
        preds = logits.argmax(dim=1).cpu().numpy().astype(int)

    return preds


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def split_file(tmp_path_factory):
    """Write a small synthetic split .pkl to a temp directory."""
    tmp = tmp_path_factory.mktemp("splits")
    path = tmp / "synthetic_split.pkl"
    split_data = {
        "train_smiles": TRAIN_SMILES,
        "test_smiles": TEST_SMILES,
        "train_labels": TRAIN_LABELS,
        "test_labels": TEST_LABELS,
    }
    with open(path, "wb") as f:
        pickle.dump(split_data, f)
    return path


@pytest.fixture(scope="module")
def datasets():
    """Build PyG Data lists for train/val/test splits."""
    train_val_data = _build_dataset(TRAIN_SMILES, TRAIN_LABELS)
    test_data = _build_dataset(TEST_SMILES, TEST_LABELS)

    # Hold out last 4 train compounds as validation
    train_data = train_val_data[:-4]
    val_data = train_val_data[-4:]

    return {"train": train_data, "val": val_data, "test": test_data}


@pytest.fixture(
    scope="module",
    params=MODEL_CLASSES,
    ids=MODEL_IDS,
)
def trained_model(request, datasets):
    """Train each GNN model type and return (model_class_name, trained_model)."""
    torch.manual_seed(42)
    model_class = request.param
    model = model_class(
        node_features_dim=NODE_FEATURES_DIM,
        hidden_channels=HIDDEN_CHANNELS,
        num_classes=NUM_CLASSES,
    )
    trained = _train(model, datasets["train"], datasets["val"])
    return model_class.__name__, trained


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_model_trains_without_error(trained_model):
    """Each model type completes training without raising."""
    model_name, model = trained_model
    assert model is not None, f"{model_name}: training returned None"


def test_model_predicts_valid_labels_on_test_set(trained_model, datasets):
    """Each trained model produces only class-0 or class-1 predictions on the test set."""
    model_name, model = trained_model
    test_data = datasets["test"]
    assert test_data, f"{model_name}: test dataset is empty"

    batch = Batch.from_data_list(test_data)
    ew = getattr(batch, "edge_weight", None)

    model.eval()
    with torch.no_grad():
        logits = model(batch.x, batch.edge_index, batch.batch, edge_weight=ew)
        preds = logits.argmax(dim=1).cpu().numpy()

    assert set(preds.tolist()).issubset({0, 1}), (
        f"{model_name}: predictions contain unexpected class labels: {set(preds.tolist())}"
    )


def test_molanchor_runs_on_single_prediction(trained_model):
    """MolAnchor completes end-to-end on one test compound for every GNN architecture."""
    model_name, model = trained_model

    # Use aspirin (class 0, has multiple BRICS bonds)
    query_smiles = "CC(=O)Oc1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(query_smiles)
    assert mol is not None

    anchor = MolecularAnchor(
        mol=mol,
        model_obj=model,
        target_class=1,
        fragment_scheme="BRICS",
        representation="graphs",
        graph_func=gnn_graph_func,
        graph_predict=lambda m, gs: gnn_graph_predict(model, gs),
    )

    df_combinations = anchor.predict_frag_combinations()

    assert isinstance(df_combinations, pd.DataFrame), (
        f"{model_name}: predict_frag_combinations() did not return a DataFrame"
    )
    assert not df_combinations.empty, (
        f"{model_name}: fragment combination DataFrame is empty"
    )
    assert "Predictions" in df_combinations.columns, (
        f"{model_name}: 'Predictions' column missing from fragment combinations"
    )

    anchors_df = anchor.identify_anchors(
        df_anchors=df_combinations,
        cutoff=0.5,
        allow_frag_combinations=True,
        return_multiple_anchors=False,
    )

    assert isinstance(anchors_df, pd.DataFrame), (
        f"{model_name}: identify_anchors() did not return a DataFrame"
    )


def test_molanchor_predictions_are_binary(trained_model):
    """All fragment-combination predictions from MolAnchor must be class 0 or 1."""
    model_name, model = trained_model

    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    anchor = MolecularAnchor(
        mol=mol,
        model_obj=model,
        target_class=1,
        fragment_scheme="BRICS",
        representation="graphs",
        graph_func=gnn_graph_func,
        graph_predict=lambda m, gs: gnn_graph_predict(model, gs),
    )

    df = anchor.predict_frag_combinations()
    unique_preds = set(df["Predictions"].unique())
    assert unique_preds.issubset({0, 1}), (
        f"{model_name}: MolAnchor predictions contain unexpected values: {unique_preds}"
    )


def test_molanchor_anchor_schema_when_found(trained_model):
    """When anchors are identified, the result DataFrame has the required columns."""
    model_name, model = trained_model

    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    anchor = MolecularAnchor(
        mol=mol,
        model_obj=model,
        target_class=1,
        fragment_scheme="BRICS",
        representation="graphs",
        graph_func=gnn_graph_func,
        graph_predict=lambda m, gs: gnn_graph_predict(model, gs),
    )

    df_combinations = anchor.predict_frag_combinations()
    anchors_df = anchor.identify_anchors(
        df_anchors=df_combinations,
        cutoff=0.0,  # accept any precision — random/weak model still produces output
        allow_frag_combinations=True,
        return_multiple_anchors=True,
    )

    if anchors_df.empty:
        pytest.skip(f"{model_name}: model never predicts target_class=1 — anchor schema check skipped")

    for col in ("anchor_smile", "precision", "anchor_mol"):
        assert col in anchors_df.columns, (
            f"{model_name}: missing column '{col}' in anchors DataFrame"
        )
