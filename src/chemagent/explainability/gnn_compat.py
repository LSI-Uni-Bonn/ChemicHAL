"""chemagent.explainability.gnn_compat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Shared GNN model loading and sklearn-compatible prediction adapters.

Used by MolAnchor, MolCE, and counterfactual tools so that PyTorch GNN
models work as drop-in replacements for sklearn fingerprint-based models
without duplicating model-loading or inference code.

Public API
----------
load_chemagent_gnn        load a .pt checkpoint into a GNN module
infer_from_mols           run GNN on a list of RDKit Mol objects
infer_gnn_params          auto-detect GNN params from .pt checkpoint metadata
GNNClassifier             sklearn-compatible adapter for generator-based XAI
make_gnn_molce_predict_funcs  prediction adapters for MolCE / MolContrast
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Must match gnn_training.py: [atomic_num/100, formal_charge, num_hs/4, is_aromatic]
_GNN_NODE_FEATURES_DIM = 4


def infer_gnn_params(
    model_path: str,
    gnn_model_class_name: Optional[str] = None,
    gnn_hidden_channels: int = 64,
    gnn_num_classes: int = 2,
) -> tuple[Optional[str], int, int]:
    """Auto-detect GNN architecture params from a ``.pt`` checkpoint.

    If *gnn_model_class_name* is already set or *model_path* does not end
    with ``.pt``, returns the inputs unchanged.  Otherwise peeks at the
    checkpoint metadata and returns any discovered values, falling back to
    the caller-supplied defaults.

    Returns:
        ``(gnn_model_class_name, gnn_hidden_channels, gnn_num_classes)``
    """
    if gnn_model_class_name is not None or not str(model_path).endswith(".pt"):
        return gnn_model_class_name, gnn_hidden_channels, gnn_num_classes

    import torch

    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict):
            gnn_model_class_name = ckpt.get("model_class_name")
            gnn_hidden_channels = int(ckpt.get("hidden_channels", gnn_hidden_channels))
            gnn_num_classes = int(ckpt.get("num_classes", gnn_num_classes))
    except Exception:
        pass

    return gnn_model_class_name, gnn_hidden_channels, gnn_num_classes


def load_chemagent_gnn(
    model_path: str,
    model_class_name: Optional[str] = None,
    hidden_channels: int = 64,
    num_classes: int = 2,
    device: str = "cpu",
):
    """Load a chemagent GNN from a .pt checkpoint file.

    Handles both the standard checkpoint dict format (produced by
    ``train_gnn_model``) and legacy raw state-dicts.  Architecture
    parameters stored in the checkpoint take precedence over caller-supplied
    defaults.

    Args:
        model_path:        Path to saved checkpoint (.pt).
        model_class_name:  One of GCN | GraphSAGE | GAT | GC_GNN | GIN.
                           When None, inferred from checkpoint metadata.
        hidden_channels:   Fallback hidden dim if not stored in checkpoint.
        num_classes:       Fallback output class count if not stored in checkpoint.
        device:            Torch device string (default "cpu").

    Returns:
        Loaded torch.nn.Module in eval mode.
    """
    import torch
    from chemagent.ml.gnn_models import GCN, GAT, GC_GNN, GIN, GraphSAGE

    _MAP = {
        "GCN": GCN, "GraphSAGE": GraphSAGE, "GAT": GAT,
        "GC_GNN": GC_GNN, "GIN": GIN,
    }

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        model_class_name = model_class_name or checkpoint.get("model_class_name")
        hidden_channels = checkpoint.get("hidden_channels", hidden_channels)
        num_classes = checkpoint.get("num_classes", num_classes)
        num_layers = checkpoint.get("num_layers", 4)
        aggregation_method = checkpoint.get("aggregation_method", "mean")
    else:
        state = checkpoint
        num_layers = 4
        aggregation_method = "mean"

    if model_class_name == "GINE":
        raise ValueError(
            "GINE requires edge_weight features not provided by the standard "
            "training pipeline. Use GCN, GraphSAGE, GIN, GC_GNN, or GAT instead."
        )
    if model_class_name not in _MAP:
        raise ValueError(
            f"Unknown GNN model class {model_class_name!r}. "
            f"Available: {list(_MAP.keys())}"
        )

    kwargs: dict = dict(
        node_features_dim=_GNN_NODE_FEATURES_DIM,
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        num_layers=num_layers,
    )
    if model_class_name in ("GraphSAGE", "GC_GNN") and aggregation_method:
        kwargs["aggregation_method"] = aggregation_method

    model = _MAP[model_class_name](**kwargs)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def infer_from_mols(model, mols, device: str = "cpu"):
    """Run a chemagent GNN on a list of RDKit Mol objects.

    Args:
        model:  Loaded GNN (``torch.nn.Module``).
        mols:   List of RDKit ``Chem.Mol`` objects (must be non-empty).
        device: Torch device string.

    Returns:
        Tuple ``(pred_classes, probas)`` as numpy arrays of shape ``(N,)``
        and ``(N, num_classes)``.
    """
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Batch
    from rdkit import Chem
    from chemagent.ml.gnn_training import smiles_to_nx_graph, nx_graph_to_pyg_data

    data_list = [
        nx_graph_to_pyg_data(smiles_to_nx_graph(Chem.MolToSmiles(mol)), label=0)
        for mol in mols
    ]
    batch = Batch.from_data_list(data_list).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(batch.x, batch.edge_index, batch.batch)

    preds = logits.argmax(dim=1).cpu().numpy().astype(int)
    probas = F.softmax(logits, dim=1).cpu().numpy()
    return preds, probas


class GNNClassifier:
    """sklearn-compatible GNN wrapper for generator-based XAI tools.

    XAI generators (e.g. ``CFGenerator``) internally call
    ``_featurize(mols) -> fp_array`` and then ``model.predict(fp_array)``.
    Subclass the generator to override ``_featurize`` so it stores the Mol
    objects in ``self.model._mol_buffer`` and returns integer indices; this
    class's ``predict`` / ``predict_proba`` then look up those Mol objects
    and run GNN inference.

    See ``_GNNCFGeneratorV3`` in ``counterfactual_tools`` for an example.
    """

    def __init__(self, model, device: str = "cpu"):
        self._model = model
        self._device = device
        self._mol_buffer: list = []

    def predict(self, indices) -> np.ndarray:
        mols = [self._mol_buffer[int(i)] for i in np.asarray(indices).flatten()]
        preds, _ = infer_from_mols(self._model, mols, self._device)
        return preds

    def predict_proba(self, indices) -> np.ndarray:
        mols = [self._mol_buffer[int(i)] for i in np.asarray(indices).flatten()]
        _, probas = infer_from_mols(self._model, mols, self._device)
        return probas


def make_gnn_molce_predict_funcs(gnn_model, device: str = "cpu"):
    """Return predict adapters in MolCE / MolContrast calling convention.

    Both returned functions accept ``(model, mol, singular=False)``
    where ``mol`` is a ``Chem.Mol`` or list of ``Chem.Mol`` objects.
    The first parameter (``model``) is ignored — the GNN is captured in
    the closure via *gnn_model* — but it must be named ``model`` because
    ``MolContrast`` passes it as a keyword argument
    (``predict_func_proba(model=self.model, mol=...)``).
    """

    def predict_func(model=None, mol=None, singular: bool = False):
        mols = [mol] if singular else list(mol)
        preds, _ = infer_from_mols(gnn_model, mols, device)
        return int(preds[0]) if singular else preds

    def predict_func_proba(model=None, mol=None, singular: bool = False):
        mols = [mol] if singular else list(mol)
        _, probas = infer_from_mols(gnn_model, mols, device)
        return probas[0] if singular else probas

    return predict_func, predict_func_proba
