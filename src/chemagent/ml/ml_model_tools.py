"""chemagent.ml.ml_model_tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MCP tool functions for ML model information and inference.

Registered via ``_register()`` in ``chemagent_mcp.py``.

Functions
---------
get_ml_info             — reference card for algorithms and recommended metrics
export_predictions      — run inference on a split .pkl, save predictions CSV

Internal helpers
----------------
predict_from_split_file — raw prediction + pkl dump (used by other tools)
build_model_from_arrays — train from raw in-memory arrays
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional

import joblib
import numpy as np

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chemagent.ml.hyperparameter_tuning import HYPERPARAMETERS
from chemagent.servers.server_helpers import (
    _run_pipeline,
    _to_serialisable,
    _predict,
    evaluate_classification,
    evaluate_regression,
)
from chemagent.session_utils import get_session_logger as _get_session_logger

# Optional imports for GNN model inference
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from chemagent.ml import gnn_models as _gnn_models
    from chemagent.ml.gnn_training import smiles_to_nx_graph, nx_graph_to_pyg_data
except Exception:
    torch = None
    F = None
    PyGDataLoader = None
    _gnn_models = None
    smiles_to_nx_graph = None
    nx_graph_to_pyg_data = None


# Reference tool
def get_ml_info() -> dict[str, Any]:
    """Return all ML reference information in one call: algorithms, hyperparameter grids, and recommended metrics.

    Call once before choosing an algorithm or opt_metric for train_model().

    Returns:
        algorithms: dict of algorithm code → name, task_type, hyperparameter grid,
                    supports_multiclass, description.
        recommended_metrics: dict of task type → optimization and evaluation metric lists.
    """
    algorithms = {
        "RFR": {
            "name": "Random Forest Regressor",
            "task_type": "regression",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("RFR", {})),
            "supports_multiclass": False,
            "description": "Ensemble of decision trees for regression tasks",
        },
        "RFC": {
            "name": "Random Forest Classifier",
            "task_type": "classification",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("RFC", {})),
            "supports_multiclass": True,
            "description": "Ensemble of decision trees for classification, handles multi-class",
        },
        "SVC": {
            "name": "Support Vector Classifier",
            "task_type": "classification",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("SVC", {})),
            "supports_multiclass": True,
            "description": "SVM classifier with RBF/linear kernels, probability estimates enabled",
        },
        "DNN": {
            "name": "Feed-forward Neural Network",
            "task_type": "both",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("DNN", {})),
            "supports_multiclass": True,
            "description": "PyTorch MLP wrapped as a scikit-learn estimator; no grid-search hyperparameters configured by default",
        },
        # Graph Neural Networks (PyTorch Geometric)
        "GCN": {
            "name": "Graph Convolutional Network",
            "task_type": "classification",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("GCN", {})),
            "supports_multiclass": True,
            "description": "Graph Convolutional Network (PyG) for graph-structured molecular data",
        },
        "GraphSAGE": {
            "name": "GraphSAGE",
            "task_type": "classification",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("GraphSAGE", {})),
            "supports_multiclass": True,
            "description": "GraphSAGE inductive representation learner for graphs",
        },
        "GAT": {
            "name": "Graph Attention Network",
            "task_type": "classification",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("GAT", {})),
            "supports_multiclass": True,
            "description": "Graph Attention Network using attention-based message passing",
        },
        "GIN": {
            "name": "Graph Isomorphism Network",
            "task_type": "classification",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("GIN", {})),
            "supports_multiclass": True,
            "description": "GIN: powerful message-passing GNN for graph classification",
        },
        "GINE": {
            "name": "GINE",
            "task_type": "classification",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("GINE", {})),
            "supports_multiclass": True,
            "description": "GINE: GIN variant with edge features for molecular graphs",
        },
        "GC_GNN": {
            "name": "GC-GNN",
            "task_type": "classification",
            "hyperparameters": _to_serialisable(HYPERPARAMETERS.get("GC_GNN", {})),
            "supports_multiclass": True,
            "description": "Custom GC_GNN architecture used in the project",
        },
    }
    recommended_metrics = {
        "binary_classification": {
            "optimization": ["f1", "roc_auc", "average_precision", "balanced_accuracy"],
            "evaluation":   ["MCC", "F1", "Precision", "Recall", "AUC", "BA"],
        },
        "binary_imbalanced": {
            "optimization": ["f1", "average_precision", "roc_auc"],
            "evaluation":   ["MCC", "F1", "Precision", "Recall", "BA"],
            "note":         "Pass task='classification-cw' to train_model() for auto class-weighting",
        },
        "multiclass": {
            "optimization": ["f1_macro", "f1_weighted", "balanced_accuracy"],
            "evaluation":   ["MCC", "BA", "F1_macro", "F1_weighted", "Accuracy"],
        },
        "regression": {
            "optimization": ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"],
            "evaluation":   ["RMSE", "MAE", "R2", "Pearson r"],
        },
    }
    return {"algorithms": algorithms, "recommended_metrics": recommended_metrics}



# Inference / export tool
def export_predictions(
    model_path: str,
    split_file_path: str,
    task: Literal["regression", "classification", "classification-cw"] = "classification",
    split: Literal["train", "val", "test"] = "test",
    save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Run inference on a split .pkl and export per-sample predictions to a CSV.

    Loads the model and split from disk, predicts on the chosen partition, and
    writes a CSV with columns: cid (if available), smiles (if available),
    true_label, predicted_label, prob_class_0, prob_class_1 (classification)
    or predicted_value (regression). Also saves aggregated metrics.

    Workflow: check_training → THIS TOOL → plot_classification_results

    Args:
        model_path: Path to .pkl model from train_model() / check_training().
        split_file_path: Path to the .pkl split file from split_dataset().
        task: "classification" (default), "classification-cw", or "regression".
        split: Partition to predict on — "test" (default), "train", or "val".
        save_path: Output CSV path. Defaults to <session>/results/<stem>_<split>_predictions.csv.

    Returns:
        csv_path, metrics_path, metrics dict, n_samples, columns.
    """
    import pandas as pd

    session_logger = _get_session_logger()

    # Handle PyTorch GNN models saved as .pt/.pth
    if str(model_path).lower().endswith((".pt", ".pth")):
        return _export_predictions_gnn(
            model_path=model_path,
            split_file_path=split_file_path,
            split=split,
            save_path=save_path,
            device=None,
        )

    data   = joblib.load(split_file_path)
    labels = data[f"{split}_labels"].tolist()
    result = _predict(model_path=model_path,
                      features=data[f"{split}_features"].tolist(),
                      reg_class=task)

    model_stem    = Path(model_path).stem
    is_regression = task == "regression"

    cid_key    = f"{split}_cid"
    smiles_key = f"{split}_smiles"
    cid_col    = data[cid_key].tolist()    if cid_key    in data else None
    smiles_col = data[smiles_key].tolist() if smiles_key in data else None

    if is_regression:
        df_out = pd.DataFrame({
            "true_label":      labels,
            "predicted_value": result["predictions"],
        })
    else:
        proba = result.get("probabilities") or []
        n_classes = len(proba[0]) if proba else 0
        df_out = pd.DataFrame({"true_label": labels, "predicted_label": result["predictions"]})
        for c in range(n_classes):
            df_out[f"prob_class_{c}"] = [p[c] for p in proba]

    if smiles_col is not None:
        df_out.insert(0, "smiles", smiles_col)
    if cid_col is not None:
        df_out.insert(0, "cid", cid_col)

    metrics_dict = (
        evaluate_regression(labels=labels, predictions=result["predictions"],
                            model_id=model_stem)
        if is_regression
        else evaluate_classification(labels=labels, predictions=result["predictions"],
                                     probabilities=result.get("probabilities"),
                                     model_id=model_stem)
    )

    if save_path is None:
        out_dir = session_logger.session_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        base          = f"{model_stem}_{split}"
        save_path     = str(out_dir / f"{base}_predictions.csv")
        metrics_path  = str(out_dir / f"{base}_metrics.pkl")
    else:
        save_path    = str(Path(save_path).resolve())
        metrics_path = str(Path(save_path).with_suffix("").as_posix() + "_metrics.pkl")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(save_path, index=False)
    joblib.dump(metrics_dict, metrics_path)

    return {
        "csv_path":     save_path,
        "metrics_path": metrics_path,
        "metrics":      metrics_dict,
        "n_samples":    len(labels),
        "columns":      list(df_out.columns),
    }


def predict_from_split_file(
    model_path: str,
    split_file_path: str,
    reg_class: Literal["regression", "classification", "classification-cw"] = "classification",
    split: Literal["train", "val", "test"] = "test",
    results_save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Internal helper — raw predictions + pkl dump. Use export_predictions() for the MCP tool."""
    session_logger = _get_session_logger()

    data   = joblib.load(split_file_path)
    labels = data[f"{split}_labels"].tolist()
    result = _predict(model_path=model_path,
                      features=data[f"{split}_features"].tolist(),
                      reg_class=reg_class)
    result["labels"] = labels

    model_stem    = Path(model_path).stem
    is_regression = reg_class == "regression"
    metrics_dict = (
        evaluate_regression(labels=labels, predictions=result["predictions"],
                            model_id=model_stem)
        if is_regression
        else evaluate_classification(labels=labels, predictions=result["predictions"],
                                     probabilities=result.get("probabilities"),
                                     model_id=model_stem)
    )
    result["metrics"] = metrics_dict

    if results_save_path is None:
        out_dir = session_logger.session_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        base              = f"{model_stem}_{split}"
        results_save_path = str(out_dir / f"{base}_predictions.pkl")
        metrics_save_path = str(out_dir / f"{base}_metrics.pkl")
    else:
        results_save_path = str(Path(results_save_path).resolve())
        metrics_save_path = str(Path(results_save_path).with_name(
            Path(results_save_path).stem + "_metrics.pkl"
        ))

    Path(results_save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result,       results_save_path)
    joblib.dump(metrics_dict, metrics_save_path)
    result["results_path"] = results_save_path
    result["metrics_path"] = metrics_save_path
    return result


def _export_predictions_gnn(
    model_path: str,
    split_file_path: str,
    split: Literal["train", "val", "test"] = "test",
    save_path: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 64,
    model_class: Optional[str] = None,
    hidden_channels: int = 64,
) -> dict[str, Any]:
    """Export predictions for PyTorch Geometric GNN models saved as .pt/.pth.

    Expects the split file to contain the per-split smiles arrays (e.g. "test_smiles").

    To avoid architecture-mismatch load errors, this function infers model
    metadata in the following priority:
    1) explicit function args,
    2) checkpoint metadata fields,
    3) filename hints,
    4) state_dict tensor shapes/keys,
    5) safe defaults.
    """
    if torch is None or PyGDataLoader is None or smiles_to_nx_graph is None:
        raise ImportError("PyTorch / PyG not available in this environment")

    session_logger = _get_session_logger()

    # Load split (support pickle or joblib like other helpers)
    split_obj = None
    load_errors: list[str] = []
    try:
        import pickle as _pickle

        with open(split_file_path, "rb") as f:
            split_obj = _pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        load_errors.append(f"pickle: {type(exc).__name__}: {exc}")

    if split_obj is None:
        try:
            split_obj = joblib.load(split_file_path)
        except Exception as exc:  # noqa: BLE001
            load_errors.append(f"joblib: {type(exc).__name__}: {exc}")

    if split_obj is None:
        raise ValueError(f"Could not load split file '{split_file_path}': {'; '.join(load_errors)}")

    smiles_key = f"{split}_smiles"
    labels_key = f"{split}_labels"
    if smiles_key not in split_obj:
        raise ValueError(f"Split file '{split_file_path}' does not contain '{smiles_key}'. Provide a split file with per-split SMILES.")

    smiles_list = list(split_obj[smiles_key])
    labels = list(split_obj[labels_key]) if labels_key in split_obj else [None] * len(smiles_list)

    # Build Data objects
    data_list = []
    for s, lbl in zip(smiles_list, labels):
        nx_g = smiles_to_nx_graph(s)
        pyg = nx_graph_to_pyg_data(nx_g, lbl if lbl is not None else 0)
        if pyg is not None:
            data_list.append(pyg)

    if not data_list:
        raise ValueError("No valid graph data could be created from the provided SMILES.")

    loader = PyGDataLoader(data_list, batch_size=batch_size)

    model_stem = Path(model_path).stem
    possible = ["GCN", "GraphSAGE", "GAT", "GC_GNN", "GINE", "GIN"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load state/checkpoint first so we can infer architecture dimensions.
    state = torch.load(model_path, map_location=device)
    checkpoint: dict[str, Any] | None = state if isinstance(state, dict) else None
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    elif isinstance(state, dict):
        sd = state
    else:
        sd = None

    # Inference priority: explicit arg -> checkpoint metadata -> filename -> state_dict keys -> fallback.
    if model_class is None and isinstance(checkpoint, dict):
        model_class = checkpoint.get("model_class_name") or checkpoint.get("model_class")
    if model_class is None:
        for name in possible:
            if name.lower() in model_stem.lower():
                model_class = name
                break
    if model_class is None and isinstance(sd, dict):
        state_keys = list(sd.keys())
        if any("att_src" in k or "att_dst" in k for k in state_keys):
            model_class = "GAT"
        elif any("lin_rel" in k or "lin_root" in k for k in state_keys):
            model_class = "GC_GNN"
        elif any("lin_l" in k and "lin_r" in k for k in state_keys):
            model_class = "GraphSAGE"
    if model_class is None:
        model_class = "GCN"

    if not hasattr(_gnn_models, model_class):
        raise ValueError(f"Unknown GNN model class '{model_class}'. Available: {', '.join([n for n in possible if hasattr(_gnn_models, n)])}")

    # Infer model dimensions from checkpoint/state dict when available.
    node_features_dim = 4
    inferred_num_classes = len(sorted(set([lbl for lbl in labels if lbl is not None]))) or 2
    inferred_hidden_channels = hidden_channels

    if isinstance(checkpoint, dict):
        node_features_dim = int(checkpoint.get("node_features_dim", node_features_dim))
        inferred_hidden_channels = int(checkpoint.get("hidden_channels", inferred_hidden_channels))
        inferred_num_classes = int(checkpoint.get("num_classes", inferred_num_classes))

    if isinstance(sd, dict) and "lin.weight" in sd:
        lin_w = sd["lin.weight"]
        if hasattr(lin_w, "shape") and len(lin_w.shape) == 2:
            inferred_num_classes = int(lin_w.shape[0])
            inferred_hidden_channels = int(lin_w.shape[1])

    ModelCls = getattr(_gnn_models, model_class)
    model = ModelCls(
        node_features_dim=node_features_dim,
        hidden_channels=inferred_hidden_channels,
        num_classes=inferred_num_classes,
    ).to(device)

    try:
        if isinstance(state, dict):
            # Accept either raw state_dict or checkpoint with 'state_dict'.
            if sd is None:
                raise RuntimeError("Checkpoint dictionary did not include a valid state dict")
            model.load_state_dict(sd)
        else:
            # If a full model object was saved
            model = state.to(device)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load model state: {exc}")

    # Inference
    all_preds: list[int] = []
    all_probs: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(out, dim=1).cpu().tolist()
            preds = out.argmax(dim=1).cpu().tolist()
            all_probs.extend(probs)
            all_preds.extend(preds)

    # Build DataFrame
    import pandas as pd

    df_out = pd.DataFrame({
        "smiles": smiles_list,
        "true_label": labels,
        "predicted_label": all_preds,
    })
    n_classes = len(all_probs[0]) if all_probs else 0
    for c in range(n_classes):
        df_out[f"prob_class_{c}"] = [p[c] for p in all_probs]

    metrics_dict = evaluate_classification(labels=labels, predictions=all_preds, probabilities=all_probs, model_id=model_stem)

    if save_path is None:
        out_dir = session_logger.session_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        base = f"{model_stem}_{split}"
        save_path = str(out_dir / f"{base}_predictions.csv")
        metrics_path = str(out_dir / f"{base}_metrics.pkl")
    else:
        save_path = str(Path(save_path).resolve())
        metrics_path = str(Path(save_path).with_suffix("").as_posix() + "_metrics.pkl")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(save_path, index=False)
    joblib.dump(metrics_dict, metrics_path)

    return {
        "csv_path": save_path,
        "metrics_path": metrics_path,
        "metrics": metrics_dict,
        "n_samples": len(labels),
        "columns": list(df_out.columns),
    }


def build_model_from_arrays(
    train_features: list[list[float]],
    train_labels: list[float],
    test_features: list[list[float]],
    test_labels: list[float],
    algorithm: Literal["RFC", "RFR", "SVC", "DNN"] = "RFC",
    task: Literal["classification", "classification-cw", "regression"] = "classification",
    cv_fold: int = 5,
    opt_metric: Optional[str] = "balanced_accuracy",
    random_seed: int = 42,
    model_save_path: Optional[str] = None,
    val_features: Optional[list[list[float]]] = None,
    val_labels: Optional[list[float]] = None,
) -> dict[str, Any]:
    """Train pipeline from raw arrays (internal helper — use train_model when possible)."""
    from chemagent.ml.training_tools import _default_model_path

    if len(train_features) != len(train_labels):
        raise ValueError("train_features and train_labels length mismatch")
    if len(test_features) != len(test_labels):
        raise ValueError("test_features and test_labels length mismatch")

    X_val = np.array(val_features) if val_features is not None else None
    y_val = np.array(val_labels)   if val_labels  is not None else None

    if model_save_path is None:
        model_save_path = _default_model_path(algorithm)

    return _run_pipeline(
        X_train=np.array(train_features), y_train=np.array(train_labels),
        X_test=np.array(test_features),   y_test=np.array(test_labels),
        X_val=X_val, y_val=y_val,
        algorithm=algorithm, task=task,
        cv_fold=cv_fold, opt_metric=opt_metric,
        random_seed=random_seed,
        model_save_path=model_save_path,
    )
