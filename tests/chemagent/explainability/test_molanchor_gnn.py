
"""Tests for MolAnchor compatibility with the new PyTorch GNN models.

MolAnchor's "graphs" representation calls:
  graph_func(mol)               -> NetworkX graph
  graph_predict(model, graphs)  -> numpy int array of predictions

The default implementations assume a sklearn-style model with .predict().
These tests provide GNN-compatible replacements and verify that
MolAnchor runs end-to-end with GCN, GraphSAGE, and GIN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import networkx as nx
from rdkit import Chem
from torch_geometric.data import Batch, Data

from chemagent.explainability.MolAnchor.MolAnchor import MolecularAnchor
from chemagent.ml.gnn_models import GCN, GIN, GraphSAGE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODE_FEATURES_DIM = 4   # atomic_num/100, formal_charge, num_hs/4, is_aromatic
HIDDEN_CHANNELS = 16
NUM_CLASSES = 2

# Aspirin has several BRICS bonds — guarantees ≥ 2 fragments
ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"


# ---------------------------------------------------------------------------
# GNN-compatible helpers
# ---------------------------------------------------------------------------

def gnn_graph_func(mol: Chem.Mol) -> nx.Graph:
    """Convert an RDKit molecule to a NetworkX graph with 4-dim node features.

    The feature encoding matches the 4-dimensional input expected by the GNN
    models in chemagent.ml.gnn_models (atomic_num/100, formal_charge,
    num_hs/4, is_aromatic).
    """
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
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return G


def gnn_graph_predict(model: torch.nn.Module, frag_graphs: list[nx.Graph]) -> np.ndarray:
    """Run GNN inference on a list of NetworkX fragment graphs.

    Converts each graph to a PyG Data object using the 4-dim feature scheme,
    batches them, and returns class predictions as a numpy int array.

    This is a drop-in replacement for MolAnchor's default_graph_predict when
    working with the chemagent GNN models (GCN, GraphSAGE, GIN, etc.).
    """
    data_list: list[Data] = []

    for g in frag_graphs:
        nodes = list(g.nodes())
        if not nodes:
            # Degenerate: empty subgraph — single zero-feature node, no edges
            x = torch.zeros((1, NODE_FEATURES_DIM), dtype=torch.float)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index))
            continue

        node_map = {n: i for i, n in enumerate(nodes)}
        node_features = []
        for node in nodes:
            attrs = g.nodes[node]
            node_features.append([
                attrs.get("atomic_num", 0.0),
                attrs.get("formal_charge", 0.0),
                attrs.get("num_hs", 0.0),
                attrs.get("is_aromatic", 0.0),
            ])
        x = torch.tensor(node_features, dtype=torch.float)

        edge_list = []
        for u, v in g.edges():
            u_i, v_i = node_map[u], node_map[v]
            edge_list.append([u_i, v_i])
            if u != v:  # self-loops added by MolAnchor: don't duplicate
                edge_list.append([v_i, u_i])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data_list.append(Data(x=x, edge_index=edge_index))

    batch = Batch.from_data_list(data_list)

    model.eval()
    with torch.no_grad():
        logits = model(batch.x, batch.edge_index, batch.batch)
        preds = logits.argmax(dim=1).cpu().numpy().astype(int)

    return preds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[GCN, GraphSAGE, GIN], ids=["GCN", "GraphSAGE", "GIN"])
def gnn_model(request):
    """Return an untrained GNN model with fixed seed for reproducibility."""
    torch.manual_seed(42)
    model_class = request.param
    return model_class(
        node_features_dim=NODE_FEATURES_DIM,
        hidden_channels=HIDDEN_CHANNELS,
        num_classes=NUM_CLASSES,
    )


@pytest.fixture
def aspirin_mol():
    mol = Chem.MolFromSmiles(ASPIRIN_SMILES)
    assert mol is not None, f"Could not parse SMILES: {ASPIRIN_SMILES}"
    return mol


@pytest.fixture
def molanchor(gnn_model, aspirin_mol):
    """Return a MolecularAnchor instance configured for GNN inference."""
    return MolecularAnchor(
        mol=aspirin_mol,
        model_obj=gnn_model,
        target_class=1,
        fragment_scheme="BRICS",
        representation="graphs",
        graph_func=gnn_graph_func,
        graph_predict=gnn_graph_predict,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fragmentation_produces_multiple_fragments(molanchor):
    """Aspirin should yield at least 2 BRICS fragments."""
    assert len(molanchor.mol_frags) >= 2
    assert len(molanchor.mol_atom_ids) == len(molanchor.mol_frags)


def test_frag_dict_covers_all_fragments(molanchor):
    """frag_dict keys should correspond 1-to-1 with fragment list."""
    assert set(molanchor.frag_dict.keys()) == {
        f"frag_{i}" for i in range(len(molanchor.mol_frags))
    }


def test_predict_frag_combinations_returns_dataframe(molanchor):
    """predict_frag_combinations() must return a non-empty DataFrame with a Predictions column."""
    df = molanchor.predict_frag_combinations()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "Fragment combination DataFrame should not be empty"
    assert "Predictions" in df.columns


def test_predictions_are_valid_class_labels(molanchor):
    """All predictions from the GNN must be 0 or 1 (binary classification)."""
    df = molanchor.predict_frag_combinations()
    unique_preds = set(df["Predictions"].unique())
    assert unique_preds.issubset({0, 1}), (
        f"Expected only labels {{0, 1}}, got {unique_preds}"
    )


def test_identify_anchors_returns_dataframe(molanchor):
    """identify_anchors() must return a DataFrame (may be empty for untrained models)."""
    df_combinations = molanchor.predict_frag_combinations()
    anchors_df = molanchor.identify_anchors(
        df_anchors=df_combinations,
        cutoff=0.5,
        allow_frag_combinations=True,
        return_multiple_anchors=False,
    )
    assert isinstance(anchors_df, pd.DataFrame)


def test_identify_anchors_schema_when_found(molanchor):
    """When anchors are found, the DataFrame has the expected columns."""
    df_combinations = molanchor.predict_frag_combinations()
    anchors_df = molanchor.identify_anchors(
        df_anchors=df_combinations,
        cutoff=0.0,  # accept any precision so random weights can still produce anchors
        allow_frag_combinations=True,
        return_multiple_anchors=True,
    )
    if anchors_df.empty:
        pytest.skip("No anchors found with random weights — schema check skipped")

    for col in ("anchor_smile", "precision", "anchor_mol", "frag_indices"):
        assert col in anchors_df.columns, f"Missing column '{col}' in anchors DataFrame"

    n_frags = len(molanchor.mol_frags)
    for row_idx, frag_ids in enumerate(anchors_df["frag_indices"]):
        assert isinstance(frag_ids, list), (
            f"Row {row_idx}: frag_indices must be a list, got {type(frag_ids).__name__}"
        )
        for fid in frag_ids:
            assert isinstance(fid, int), (
                f"Row {row_idx}: frag_indices entries must be ints, got {type(fid).__name__}"
            )
            assert 0 <= fid < n_frags, (
                f"Row {row_idx}: frag_indices entry {fid} out of range [0, {n_frags})"
            )


def test_frag_indices_match_anchor_structure(molanchor):
    """frag_indices must point at the exact fragments MolAnchor identified.

    Regression for a bug where the caller re-matched anchors back to mol_frags
    by atom count, so any unrelated fragment with the same heavy-atom count
    would be flagged as an anchor. This asserts the structural contract:
    canonical SMILES of mol_frags[i] for each i in frag_indices must equal
    the row's anchor SMILES (modulo BRICS dummy-atom number stripping).
    """
    from chemagent.explainability.MolAnchor.utils_anchor import (
        delete_numbers_next_to_asterisk,
    )

    df_combinations = molanchor.predict_frag_combinations()
    anchors_df = molanchor.identify_anchors(
        df_anchors=df_combinations,
        cutoff=0.0,
        allow_frag_combinations=True,
        return_multiple_anchors=True,
    )
    if anchors_df.empty:
        pytest.skip("No anchors found with random weights")

    for row_idx, row in anchors_df.iterrows():
        if row["anchor_mol"] in ("no_anchor", "all_frags"):
            assert row["frag_indices"] == [], (
                f"Row {row_idx}: sentinel rows must have empty frag_indices"
            )
            continue

        frag_ids = row["frag_indices"]
        assert frag_ids, f"Row {row_idx}: non-sentinel row has empty frag_indices"

        recovered = sorted(
            delete_numbers_next_to_asterisk(Chem.MolToSmiles(molanchor.mol_frags[i]))
            for i in frag_ids
        )
        anchor_smile = row["anchor_smile"]
        expected = sorted([anchor_smile] if isinstance(anchor_smile, str) else list(anchor_smile))

        assert recovered == expected, (
            f"Row {row_idx}: frag_indices {frag_ids} resolve to {recovered}, "
            f"but anchor_smile says {expected}"
        )


def test_full_pipeline_no_exception(gnn_model):
    """End-to-end smoke test: MolAnchor runs without raising for all supported GNNs."""
    mol = Chem.MolFromSmiles(ASPIRIN_SMILES)
    anchor = MolecularAnchor(
        mol=mol,
        model_obj=gnn_model,
        target_class=1,
        fragment_scheme="BRICS",
        representation="graphs",
        graph_func=gnn_graph_func,
        graph_predict=gnn_graph_predict,
    )
    df_combinations = anchor.predict_frag_combinations()
    _ = anchor.identify_anchors(
        df_anchors=df_combinations,
        cutoff=0.5,
        allow_frag_combinations=True,
        return_multiple_anchors=False,
    )
