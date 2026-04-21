"""MCP-path integration test: explain_with_molanchor with GNN models.

Exercises the full MCP tool call path:
  1. Train each GNN architecture on a small synthetic dataset
  2. Save the state dict to a tmp .pt file
  3. Call explain_with_molanchor(..., representation="graphs",
         gnn_model_class_name=...) — the same entrypoint the MCP agent uses
  4. Assert the returned structure is well-formed
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from chemagent.explainability.molanchor_tools import explain_with_molanchor
from chemagent.ml.gnn_models import GAT, GC_GNN, GCN, GIN, GraphSAGE

# ---------------------------------------------------------------------------
# Synthetic training data (same as integration test)
# ---------------------------------------------------------------------------

_SMILES_LABELS = [
    ("CC(=O)Oc1ccccc1C(=O)O", 0),
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", 0),
    ("CC(=O)Nc1ccc(O)cc1", 0),
    ("Clc1ccccc1NC(=O)c1ccccc1", 0),
    ("CC(=O)Nc1ccc(Cl)cc1", 0),
    ("Cc1ccc(NC(=O)c2ccccc2)cc1", 0),
    ("O=C(O)c1ccccc1NC(=O)c1ccccc1", 0),
    ("COc1ccc(CC(N)C(=O)O)cc1", 0),
    ("CCOC(=O)c1ccc(N)cc1", 0),
    ("Cc1ccc(S(=O)(=O)Nc2ccccn2)cc1", 0),
    ("Cn1c(=O)c2c(ncn2C)n(C)c1=O", 1),
    ("CC(C)NCC(O)c1ccc(O)c(O)c1", 1),
    ("c1ccc(-c2ccccn2)nc1", 1),
    ("CC1=CC(=O)c2ccccc2C1=O", 1),
    ("O=C(NNc1ccccc1)c1ccncc1", 1),
    ("CCOC(=O)c1cccc(NC(=O)OCC)c1", 1),
    ("O=C1CCCN1c1ccc(F)cc1", 1),
    ("Cc1ccc(cc1)C(=O)NN", 1),
    ("CCOC(=O)c1ccc(NC(=O)c2ccccc2)cc1", 1),
    ("CC(=O)Nc1ccc(NC(=O)c2ccccc2)cc1", 1),
]

TRAIN_SMILES = [s for s, _ in _SMILES_LABELS[:16]]
TRAIN_LABELS = [l for _, l in _SMILES_LABELS[:16]]

NODE_FEATURES_DIM = 4
HIDDEN_CHANNELS = 16
NUM_CLASSES = 2
QUERY_SMILES = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin

# GINE is intentionally excluded: it requires edge_weight but train_gnn_model
# never provides it, making GINE incompatible with the standard training pipeline.
# _load_gnn_model raises a clear ValueError for GINE — tested separately below.
MODEL_CLASSES = [GCN, GraphSAGE, GIN, GC_GNN, GAT]
MODEL_IDS = ["GCN", "GraphSAGE", "GIN", "GC_GNN", "GAT"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smiles_to_pyg(smiles: str, label: int) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    node_features = [
        [a.GetAtomicNum() / 100.0, float(a.GetFormalCharge()),
         a.GetTotalNumHs() / 4.0, float(a.GetIsAromatic())]
        for a in mol.GetAtoms()
    ]
    x = torch.tensor(node_features, dtype=torch.float)
    edge_list, ew_list = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        w = bond.GetBondTypeAsDouble() / 3.0
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


def _quick_train(model_class, train_data: list[Data], val_data: list[Data]) -> torch.nn.Module:
    torch.manual_seed(42)
    model = model_class(node_features_dim=NODE_FEATURES_DIM,
                        hidden_channels=HIDDEN_CHANNELS, num_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

    for _ in range(5):
        model.train()
        for batch in train_loader:
            ew = getattr(batch, "edge_weight", None)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch, edge_weight=ew)
            loss = criterion(logits, batch.y.view(-1).long())
            loss.backward()
            optimizer.step()

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_val_data():
    all_data = [d for s, l in _SMILES_LABELS[:16] if (d := _smiles_to_pyg(s, l))]
    return all_data[:-4], all_data[-4:]


@pytest.fixture(
    scope="module",
    params=list(zip(MODEL_CLASSES, MODEL_IDS)),
    ids=MODEL_IDS,
)
def saved_model(request, train_val_data, tmp_path_factory):
    """Train each GNN, save the state dict, return (class_name, pt_path)."""
    model_class, model_name = request.param
    train_data, val_data = train_val_data
    model = _quick_train(model_class, train_data, val_data)

    tmp = tmp_path_factory.mktemp("models")
    pt_path = tmp / f"gnn_{model_name}.pt"
    torch.save(model.state_dict(), pt_path)

    return model_name, str(pt_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_explain_with_molanchor_gnn_returns_list(saved_model):
    """explain_with_molanchor must return a list for every GNN architecture."""
    model_name, pt_path = saved_model
    result = explain_with_molanchor(
        smiles=QUERY_SMILES,
        model_path=pt_path,
        representation="graphs",
        gnn_model_class_name=model_name,
        gnn_hidden_channels=HIDDEN_CHANNELS,
        gnn_num_classes=NUM_CLASSES,
        target_class=1,
        cutoff=0.5,
        allow_frag_combinations=True,
    )
    assert isinstance(result, list), (
        f"{model_name}: explain_with_molanchor did not return a list, got {type(result)}"
    )
    assert len(result) >= 1, f"{model_name}: result list is empty"


def test_explain_with_molanchor_gnn_metadata_is_valid_json(saved_model):
    """The last element of the result must be valid JSON with expected keys."""
    model_name, pt_path = saved_model
    result = explain_with_molanchor(
        smiles=QUERY_SMILES,
        model_path=pt_path,
        representation="graphs",
        gnn_model_class_name=model_name,
        gnn_hidden_channels=HIDDEN_CHANNELS,
        gnn_num_classes=NUM_CLASSES,
        target_class=1,
        cutoff=0.5,
        allow_frag_combinations=True,
    )
    # Last element is always a JSON metadata string
    metadata_str = result[-1]
    assert isinstance(metadata_str, str), (
        f"{model_name}: last result element should be a JSON string"
    )
    metadata = json.loads(metadata_str)

    for key in ("smiles", "status", "num_fragments", "anchor_smiles", "precision"):
        assert key in metadata, f"{model_name}: missing key '{key}' in MolAnchor metadata"

    assert metadata["status"] == "completed", (
        f"{model_name}: MolAnchor status is '{metadata['status']}', expected 'completed'"
    )
    assert metadata["smiles"] == QUERY_SMILES
    assert metadata["num_fragments"] >= 2, (
        f"{model_name}: aspirin should have at least 2 BRICS fragments"
    )


def test_explain_with_molanchor_gnn_wrong_class_raises(saved_model, tmp_path_factory):
    """Passing an invalid gnn_model_class_name must raise ValueError."""
    _, pt_path = saved_model
    with pytest.raises(ValueError, match="Unknown GNN model class"):
        explain_with_molanchor(
            smiles=QUERY_SMILES,
            model_path=pt_path,
            representation="graphs",
            gnn_model_class_name="NonExistentNet",
            gnn_hidden_channels=HIDDEN_CHANNELS,
            gnn_num_classes=NUM_CLASSES,
        )


def test_explain_with_molanchor_gnn_missing_class_raises(saved_model):
    """representation='graphs' without gnn_model_class_name must raise ValueError."""
    _, pt_path = saved_model
    with pytest.raises(ValueError, match="gnn_model_class_name"):
        explain_with_molanchor(
            smiles=QUERY_SMILES,
            model_path=pt_path,
            representation="graphs",
            # gnn_model_class_name intentionally omitted
        )


def test_explain_with_molanchor_gine_raises_clear_error(tmp_path):
    """GINE must raise a descriptive ValueError — it is incompatible with the
    standard training pipeline which never provides edge_weight."""
    # Dummy .pt file — the error is raised before weights are loaded
    from chemagent.ml.gnn_models import GINE
    dummy_model = GINE(node_features_dim=4, hidden_channels=16, num_classes=2)
    pt_path = tmp_path / "gine_dummy.pt"
    torch.save(dummy_model.state_dict(), pt_path)

    with pytest.raises(ValueError, match="GINE"):
        explain_with_molanchor(
            smiles=QUERY_SMILES,
            model_path=str(pt_path),
            representation="graphs",
            gnn_model_class_name="GINE",
            gnn_hidden_channels=16,
            gnn_num_classes=2,
        )
