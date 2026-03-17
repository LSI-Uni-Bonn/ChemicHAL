"""chemagent.ml.ml_model_tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MCP tool functions for ML model information and inference.

Registered via ``_register()`` in ``chemagent_mcp.py``.

Functions
---------
get_ml_info             — reference card for algorithms and recommended metrics
export_predictions      — run inference on a split .pkl, save predictions CSV
run_pipeline            — one-call shortcut: load → featurize → split → train (non-blocking)

Internal helpers
----------------
predict_from_split_file — raw prediction + pkl dump (used by other tools)
build_model_from_arrays — train from raw in-memory arrays
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Literal, Optional

import joblib
import numpy as np

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chemagent.datasets import (
    load_csv,
    featurize_df,
    build_processed_entry,
    split_processed,
    save_split as _save_split,
)
from chemagent.datasets.dataset_tools import _loaded_datasets, _processed_datasets
from chemagent.ml.hyperparameter_tuning import HYPERPARAMETERS
from chemagent.servers.server_helpers import (
    _run_pipeline,
    _to_serialisable,
    _predict,
    _workspace_root,
    evaluate_classification,
    evaluate_regression,
)
from chemagent.session_utils import get_session_logger as _get_session_logger


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
    }
    recommended_metrics = {
        "binary_classification": {
            "optimization": ["f1", "roc_auc", "average_precision", "balanced_accuracy"],
            "evaluation":   ["MCC", "F1", "Precision", "Recall", "AUC", "Balanced Accuracy"],
        },
        "binary_imbalanced": {
            "optimization": ["f1", "average_precision", "roc_auc"],
            "evaluation":   ["MCC", "F1", "Precision", "Recall", "Balanced Accuracy"],
            "note":         "Pass task='classification-cw' to train_model() for auto class-weighting",
        },
        "multiclass": {
            "optimization": ["f1_macro", "f1_weighted", "balanced_accuracy"],
            "evaluation":   ["MCC", "Balanced Accuracy", "F1_macro", "F1_weighted", "Accuracy"],
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


def build_model_from_arrays(
    train_features: list[list[float]],
    train_labels: list[float],
    test_features: list[list[float]],
    test_labels: list[float],
    algorithm: Literal["RFC", "RFR", "SVC"] = "RFC",
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


def run_pipeline(
    file_path: str,
    algorithm: Literal["RFC", "RFR", "SVC"] = "RFC",
    task: Literal["classification", "classification-cw", "regression"] = "classification",
    method: str = "ECFP",
    featurizer_kwargs: Optional[dict] = None,
    train_size: float = 0.7,
    test_size: float = 0.3,
    stratified: bool = True,
    cv_fold: int = 5,
    opt_metric: Optional[str] = "balanced_accuracy",
    random_seed: int = 42,
    label_col: str = "class_label",
    smiles_col: str = "smiles",
    id_col: Optional[str] = None,
) -> dict[str, Any]:
    """One-call shortcut: load → featurize → split → train+evaluate (non-blocking).

    Steps 1-3 (load, featurize, split) run immediately; training is submitted as a
    background job. Poll with check_training(job_id, model_save_path=model_save_path) every 30 s until done.

    Args:
        file_path: CSV dataset path (workspace-relative or absolute).
        algorithm: "RFC" (default), "RFR", or "SVC". See get_ml_info().
        task: "classification" (default), "classification-cw", or "regression".
        method: Featurization method (default "ECFP"). See list_featurizers().
        featurizer_kwargs: Method-specific parameters forwarded to the featurizer,
            e.g. {"n_bits": 2048, "radius": 2} for ECFP. Defaults to None (method defaults).
        train_size: Training fraction (default 0.7).
        test_size: Test fraction (default 0.3).
        stratified: Stratify the split (default True).
        cv_fold: GridSearchCV folds (default 5).
        opt_metric: Scoring metric for GridSearchCV (default "balanced_accuracy").
        random_seed: Random seed (default 42).
        label_col: Target column in the CSV (default "class_label").
        smiles_col: SMILES column in the CSV (default "smiles").
        id_col: Compound ID column (optional).

    Returns:
        job_id, status="running", split_file_path, dataset_id, n_samples, n_features, model_save_path.
        Poll check_training(job_id, model_save_path=model_save_path) for the final result.
        Always pass model_save_path to check_training — it enables on-disk fallback if the
        server restarts and the in-memory job state is lost.
    """
    import uuid
    from chemagent.ml.training_tools import _default_model_path, _run_job_in_background

    session_logger = _get_session_logger()

    # 1. Load
    df, meta = load_csv(
        file_path=file_path, label_col=label_col, smiles_col=smiles_col,
        id_col=id_col, directory="",
    )
    ds_id = meta["dataset_id"]
    _loaded_datasets[ds_id] = df
    threading.Thread(
        target=session_logger.save_dataframe,
        args=(df, ds_id),
        daemon=True,
    ).start()

    # 2. Featurize
    features = featurize_df(df, method=method, **(featurizer_kwargs or {}))
    _processed_datasets[ds_id] = build_processed_entry(
        df=df, features=features, label_col=label_col
    )

    # 3. Split
    split_result = split_processed(
        processed=_processed_datasets[ds_id],
        split_type="random",
        train_size=train_size,
        val_size=0.0,
        test_size=test_size,
        seed=random_seed,
        stratified=stratified,
    )
    out_dir = session_logger.session_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = str(out_dir / f"{ds_id}_random.pkl")
    _save_split(
        save_dict=split_result["save_dict"],
        dataset_id=ds_id,
        split_type="random",
        save_path=split_path,
    )

    # 4. Train — submit as background job (non-blocking)
    split   = joblib.load(split_path)
    X_train = np.array(split["train_features"])
    y_train = np.array(split["train_labels"])
    X_test  = np.array(split["test_features"])
    y_test  = np.array(split["test_labels"])

    model_save_path = _default_model_path(algorithm, stem=Path(split_path).stem)
    job_id = str(uuid.uuid4())
    _run_job_in_background(
        job_id, _run_pipeline,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        X_val=None,      y_val=None,
        algorithm=algorithm, task=task,
        cv_fold=cv_fold, opt_metric=opt_metric,
        random_seed=random_seed,
        model_save_path=model_save_path,
    )

    return {
        "job_id":          job_id,
        "status":          "running",
        "dataset_id":      ds_id,
        "n_samples":       int(features.shape[0]),
        "n_features":      int(features.shape[1]),
        "split_file_path": split_path,
        "model_save_path": model_save_path,
        "message": (
            f"Load/featurize/split completed. Training started in the background. "
            f"Call check_training('{job_id}', model_save_path='{model_save_path}') to poll for completion. "
            "Poll every 30 seconds until status is 'completed' or 'failed'."
        ),
    }
