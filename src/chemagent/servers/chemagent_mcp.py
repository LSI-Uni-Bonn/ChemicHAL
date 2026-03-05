"""chemagent_mcp.py — single consolidated FastMCP server (17 tools).

STANDARD WORKFLOW (data stays on disk — preferred):
    find_datasets()                                          # discover CSVs
    load_dataset("data/datasets/chembl_activity_data_O00329_P42336.csv")
    compute_features(dataset_id, method="ECFP", n_bits=2048)
    split_dataset(dataset_id, train_size=0.7, test_size=0.3, stratified=True)
    job = train_model(split_file_path, algorithm="RFC",
                      task="classification", opt_metric="balanced_accuracy")
    result = check_training(job["job_id"], model_save_path=job["model_save_path"])  # poll every 30 s
    plot_classification_results(result["model_path"], split_file_path)

SHORTCUT (load+featurize+split synchronously, then trains in background):
    job = run_pipeline("data/datasets/chembl_activity_data_O00329_P42336.csv",
                       algorithm="RFC", task="classification",
                       featurizer_kwargs={"n_bits": 2048, "radius": 2})
    result = check_training(job["job_id"], model_save_path=job["model_save_path"])  # poll every 30 s

TOOLS
─────────────────────────────────────────────
Dataset
  find_datasets          list CSV files in a directory
  list_loaded_datasets   inspect in-memory state
  list_featurizers       discover available fingerprint methods
  load_dataset           load a CSV for ML
  compute_features       compute molecular fingerprints server-side
  split_dataset          create train/test splits, save .pkl
  dataset_status         inspect a dataset's current load/prepare state

ML
  get_ml_info            algorithms, hyperparameter grids, recommended metrics
  train_model            non-blocking train+tune pipeline from split .pkl
  check_training         poll a background training job
  export_predictions     run inference on a split .pkl, save predictions CSV

Plots
  plot_dataset_info           class distribution, column histograms, split stats
  plot_classification_results confusion matrix, ROC, PR, metric bar, importances
  plot_regression_results     actual vs predicted, residuals, error distribution

Utilities
  log_thought            record reasoning in the session log
  start_new_session      start a fresh session directory
  run_pipeline           non-blocking shortcut: load → featurize → split → train
"""

from __future__ import annotations

import functools
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

import joblib
import numpy as np
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Make chemagent packages importable when launched from servers/ via uv run
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parents[2]  # …/src/
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chemagent.datasets import (
    load_csv,
    featurize_df,
    build_processed_entry,
    prepare_from_external_features,
    list_csv_files,
    list_featurizers as _list_featurizers_impl,
    split_processed,
    save_split as _save_split,
    load_split_file,
    get_ml_ready_data as _get_ml_ready_data_impl,
    get_dataset_info as _get_dataset_info_impl,
    label_stats as _label_stats,
)
from chemagent.ml import MLModel, Model_Evaluation
from chemagent.ml.hyperparameter_tuning import HYPERPARAMETERS
from chemagent.logging import SessionLogger

# Must set backend before any pyplot import (plot modules import pyplot at module level)
import matplotlib
matplotlib.use("Agg")

from chemagent.plots.classification import (
    plot_confusion_matrix as _plot_confusion_matrix,
    plot_roc_curve        as _plot_roc_curve,
    plot_pr_curve         as _plot_pr_curve,
    plot_metric_bar       as _plot_metric_bar,
    plot_feature_importance as _plot_feature_importance,
    plot_threshold_metrics  as _plot_threshold_metrics,
)
from chemagent.plots.dataset import (
    plot_class_distribution  as _plot_class_distribution,
    plot_split_statistics    as _plot_split_statistics,
    plot_column_distribution as _plot_column_distribution,
    plot_class_balance_splits as _plot_class_balance_splits,
    plot_dataset_comparison  as _plot_dataset_comparison,
)
from chemagent.plots.regression import (
    plot_actual_vs_predicted  as _plot_actual_vs_predicted,
    plot_residuals            as _plot_residuals,
    plot_residual_histogram   as _plot_residual_histogram,
    plot_error_distribution   as _plot_error_distribution,
)
from chemagent.servers.server_helpers import (
    _workspace_root,
    _resolve_path,
    _to_serialisable,
    _DataContainer,
    _build_evaluator,
    _run_pipeline,
    _predict,
    evaluate_classification,
    evaluate_regression,
    get_hyperparameter_grids,
    train_on_split_file,
)

mcp = FastMCP("chemagent")

# ---------------------------------------------------------------------------
# Session logger — writes data/logs/session_<timestamp>_<id>.txt
# ---------------------------------------------------------------------------
_log_dir = Path(__file__).resolve().parents[3] / "data" / "logs"
session_logger = SessionLogger(_log_dir)

def _register(fn):
    """Wrap *fn* with call/result logging and register it as an MCP tool."""
    @functools.wraps(fn)
    def logged_fn(*args, **kwargs):
        call_id   = session_logger.start_call(fn.__name__, kwargs)
        t_start   = time.perf_counter()
        try:
            result      = fn(*args, **kwargs)
            duration_ms = (time.perf_counter() - t_start) * 1000
            session_logger.end_call(call_id, result=result, duration_ms=duration_ms)
            # Copy any files produced by this call into the session directory
            session_logger.copy_artifacts_from_result(result)
            return result
        except Exception as exc:
            duration_ms = (time.perf_counter() - t_start) * 1000
            session_logger.end_call(call_id, error=exc, duration_ms=duration_ms)
            raise
    mcp.add_tool(logged_fn)
    return logged_fn

# ---------------------------------------------------------------------------
# In-memory state  (ephemeral — lost on server restart)
# ---------------------------------------------------------------------------
_loaded_datasets:    dict[str, Any] = {}   # dataset_id → pd.DataFrame
_processed_datasets: dict[str, dict[str, Any]] = {}

# Background training jobs  {job_id → {status, result, error, started_at, finished_at}}
_jobs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Shared internal helpers  (pure helpers live in server_helpers.py)
# ---------------------------------------------------------------------------

def _default_model_path(algorithm: str, stem: str = "") -> str:
    out_dir = session_logger.session_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{stem}_{algorithm}.pkl" if stem else f"trained_model_{algorithm}.pkl"
    return str(out_dir / name)


def _default_plot_path(name: str, ext: str = "png") -> str:
    """Return an auto-generated path inside the session plots/ directory."""
    out_dir = session_logger.session_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{name}.{ext}")


def _run_job_in_background(job_id: str, fn, *args, **kwargs) -> None:
    """Run *fn* in a daemon thread; write result/error into _jobs[job_id]."""
    def _worker():
        t_start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = result
            session_logger.log_event(
                "background_job_completed",
                job_id=job_id,
                duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )
            session_logger.copy_artifacts_from_result(result)
        except Exception as exc:  # noqa: BLE001
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"]  = str(exc)
            session_logger.log_event(
                "background_job_failed",
                job_id=job_id,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )
        finally:
            _jobs[job_id]["finished_at"] = time.time()

    _jobs[job_id] = {
        "status":      "running",
        "result":      None,
        "error":       None,
        "started_at":  time.time(),
        "finished_at": None,
    }
    threading.Thread(target=_worker, daemon=True).start()


# ===========================================================================
# DATASET TOOLS
# ===========================================================================

@_register
def find_datasets(directory: str = "data/datasets") -> dict[str, Any]:
    """List CSV files available for ML in a directory.

    Workflow: THIS TOOL → load_dataset(file_path)

    Args:
        directory: Workspace-relative or absolute path to search (default: "data/datasets").

    Returns:
        datasets (list of filenames), count, directory (resolved path).
    """
    return list_csv_files(directory)


@_register
def list_loaded_datasets() -> dict[str, Any]:
    """List datasets currently in server memory (state is ephemeral — lost on restart).

    If lists are empty, re-run load_dataset() and compute_features() before continuing.

    Returns:
        loaded (raw), prepared (featurized + ready to split), totals, note.
    """
    return {
        "loaded":         list(_loaded_datasets.keys()),
        "prepared":       list(_processed_datasets.keys()),
        "total_loaded":   len(_loaded_datasets),
        "total_prepared": len(_processed_datasets),
        "note":           "State is ephemeral — lost on server restart.",
    }


@_register
def list_featurizers() -> dict[str, Any]:
    """List all available molecular featurization methods.

    Returns name, parameters, and description for each method.
    Use the name directly as the `method` argument to compute_features().
    """
    return _list_featurizers_impl()


@_register
def load_dataset(
    file_path: str,
    label_col: str = "class_label",
    smiles_col: Optional[str] = "smiles",
    id_col: Optional[str] = None,
    feature_cols: Optional[list[str]] = None,
    dataset_id: Optional[str] = None,
    directory: str = "",
) -> dict[str, Any]:
    """Load a CSV dataset into server memory for ML.

    Workflow: find_datasets → THIS TOOL → compute_features

    Supports: (1) molecular datasets with SMILES, (2) tabular datasets with
    explicit feature_cols, (3) tabular datasets with auto-detected numeric features.

    Args:
        file_path: Absolute, workspace-relative, or filename within `directory`.
        label_col: Target column (default "class_label").
        smiles_col: SMILES column, or None for non-molecular datasets.
        id_col: Compound/sample ID column (optional).
        feature_cols: Explicit feature column list (ignored when smiles_col is set).
        dataset_id: In-memory cache key (default: CSV filename stem).
        directory: Directory prefix when file_path is a bare filename.

    Returns:
        dataset_id, n_samples, label_col, label_stats, has_smiles, next_step.
    """
    df, meta = load_csv(
        file_path=file_path,
        label_col=label_col,
        smiles_col=smiles_col,
        id_col=id_col,
        feature_cols=feature_cols,
        dataset_id=dataset_id,
        directory=directory,
    )
    ds_id = meta["dataset_id"]
    _loaded_datasets[ds_id] = df
    # Save a CSV copy of the dataset to the session log directory
    session_logger.save_dataframe(df, ds_id)
    if "_features_arr" in meta:
        features_arr = meta.pop("_features_arr")
        _processed_datasets[ds_id] = build_processed_entry(
            df=df, features=features_arr,
            label_col=label_col, smiles_col=smiles_col, id_col=id_col,
        )
    return {k: v for k, v in meta.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Internal helpers — not registered as MCP tools
# ---------------------------------------------------------------------------

def get_dataset_smiles(dataset_id: str) -> dict[str, Any]:
    """Retrieve SMILES strings from a loaded dataset (internal helper)."""
    if dataset_id not in _loaded_datasets:
        raise ValueError(f"Dataset '{dataset_id}' not loaded. Call load_dataset() first.")
    df = _loaded_datasets[dataset_id]
    smiles_col = df.attrs.get("smiles_col", "smiles")
    if not smiles_col or smiles_col not in df.columns:
        raise ValueError(
            f"No SMILES column configured for '{dataset_id}'. "
            "Pass smiles_col when calling load_dataset()."
        )
    return {"dataset_id": dataset_id, "smiles": df[smiles_col].tolist(), "n_samples": len(df)}


@_register
def compute_features(
    dataset_id: str,
    method: str = "ECFP",
    n_bits: int = 2048,
    radius: int = 2,
    label_col: Optional[str] = None,
) -> dict[str, Any]:
    """Compute molecular fingerprints server-side for a loaded dataset.

    Features stay on disk — nothing large is returned to the LLM context.
    Workflow: load_dataset → THIS TOOL → split_dataset

    Args:
        dataset_id: ID from load_dataset().
        method: Fingerprint method (default: "ECFP"). Call list_featurizers() to see all.
        n_bits: Bit vector size (default: 2048).
        radius: Morgan radius (default: 2, i.e. ECFP4).
        label_col: Override label column from load_dataset().

    Returns:
        dataset_id, method, n_samples, n_features, label_stats, prepared=True, next_step.
    """
    if dataset_id not in _loaded_datasets:
        raise ValueError(f"Dataset '{dataset_id}' not loaded. Call load_dataset() first.")
    df = _loaded_datasets[dataset_id]
    lc = label_col or df.attrs.get("label_col", "class_label")
    if lc not in df.columns:
        raise ValueError(f"Label column '{lc}' not found. Available: {df.columns.tolist()}")
    features = featurize_df(df, method=method, n_bits=n_bits, radius=radius)
    _processed_datasets[dataset_id] = build_processed_entry(df=df, features=features, label_col=lc)
    return {
        "dataset_id": dataset_id,
        "method":     method,
        "n_samples":  int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "label_stats": _label_stats(df[lc].values),
        "prepared":   True,
        "next_step": (
            f"Call split_dataset('{dataset_id}', train_size=0.7, "
            "val_size=0.0, test_size=0.3, stratified=True) to create splits."
        ),
    }


def prepare_ml_dataset(
    dataset_id: str,
    features: list[list[float]],
    label_col: Optional[str] = None,
) -> dict[str, Any]:
    """Pair external features with a loaded dataset (internal helper)."""
    if dataset_id not in _loaded_datasets:
        raise ValueError(f"Dataset '{dataset_id}' not loaded. Call load_dataset() first.")
    df  = _loaded_datasets[dataset_id]
    lc  = label_col or df.attrs.get("label_col", "class_label")
    if lc not in df.columns:
        raise ValueError(f"Label column '{lc}' not found. Available: {df.columns.tolist()}")
    _processed_datasets[dataset_id] = prepare_from_external_features(
        df=df, features=features, label_col=lc
    )
    features_arr = np.array(features)
    return {
        "dataset_id": dataset_id,
        "n_samples":  int(features_arr.shape[0]),
        "n_features": int(features_arr.shape[1]),
        "label_stats": _label_stats(df[lc].values),
        "prepared":   True,
    }


@_register
def split_dataset(
    dataset_id: str,
    split_type: Literal["random", "scaffold"] = "random",
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: Optional[int] = 42,
    stratified: bool = False,
    save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Split a featurized dataset into train/val/test partitions and save to .pkl.

    Workflow: compute_features → THIS TOOL → train_model(split_file_path)

    Args:
        dataset_id: ID of a featurized dataset (from compute_features).
        split_type: "random" (default) or "scaffold".
        train_size: Training fraction (default 0.8).
        val_size: Validation fraction (default 0.1). Use 0.0 for two-way split.
        test_size: Test fraction (default 0.1).
        seed: Random seed (default 42).
        stratified: Preserve class proportions — random splits only.
        save_path: Output .pkl path. Defaults to session splits/ dir. Pass "" to skip.

    Returns:
        train/val/test metadata, statistics, saved_to path, next_step hint.
    """
    if dataset_id not in _processed_datasets:
        raise ValueError(
            f"Dataset '{dataset_id}' not featurized. "
            "Call compute_features() first."
        )
    processed    = _processed_datasets[dataset_id]
    split_result = split_processed(
        processed=processed, split_type=split_type,
        train_size=train_size, val_size=val_size, test_size=test_size,
        seed=seed, stratified=stratified,
    )
    train_idx  = split_result["train_idx"]
    val_idx    = split_result["val_idx"]
    test_idx   = split_result["test_idx"]
    statistics = split_result["statistics"]

    saved_to = None
    if save_path != "":
        if save_path is None:
            out_dir = session_logger.session_dir / "splits"
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(out_dir / f"{dataset_id}_{split_type}.pkl")
        else:
            save_path = _resolve_path(save_path)
        saved_to = _save_split(
            save_dict=split_result["save_dict"],
            dataset_id=dataset_id,
            split_type=split_type,
            save_path=save_path,
        )

    def _split_meta(idx):
        meta: dict[str, Any] = {
            "n_samples": len(idx),
            "indices":   idx.tolist() if hasattr(idx, "tolist") else list(idx),
        }
        if "smiles" in processed:
            meta["smiles_sample"] = processed["smiles"][idx[:3]].tolist()
        if "cid" in processed:
            meta["cid_sample"]    = processed["cid"][idx[:3]].tolist()
        return meta

    warnings_list = []
    if len(val_idx) == 0:
        warnings_list.append(
            "val split is empty (val_size=0.0). "
            "Use split='test' when calling predict_from_split_file."
        )

    result: dict[str, Any] = {
        "train":      _split_meta(train_idx),
        "val":        _split_meta(val_idx),
        "test":       _split_meta(test_idx),
        "split_type": split_type,
        "statistics": statistics,
        "seed":       seed,
        "saved_to":   saved_to,
        "next_step": (
            f"Call train_model(split_file_path='{saved_to}', "
            "algorithm='RFC', task='classification', opt_metric='balanced_accuracy') "
            "to train in the background (non-blocking). "
            "Then poll with check_training(job_id) until status='completed'. "
            "Features are on disk — not returned here."
        ) if saved_to else "No file saved. Pass save_path.",
    }
    if warnings_list:
        result["warnings"] = warnings_list
    return result


def load_split(file_path: str) -> dict[str, Any]:
    """Load a saved train/val/test split .pkl (internal helper)."""
    return load_split_file(file_path)


def get_ml_ready_data(dataset_id: str, as_lists: bool = True) -> dict[str, Any]:
    """Return the processed feature matrix and labels (internal helper)."""
    if dataset_id not in _processed_datasets:
        raise ValueError(
            f"Dataset '{dataset_id}' not prepared. "
            "Call compute_features() first."
        )
    return _get_ml_ready_data_impl(
        processed=_processed_datasets[dataset_id],
        dataset_id=dataset_id,
        as_lists=as_lists,
    )


@_register
def dataset_status(dataset_id: str) -> dict[str, Any]:
    """Return load/prepare status and metadata for a dataset.

    Args:
        dataset_id: Dataset ID to inspect.

    Returns:
        loaded, prepared, raw_data (if loaded), ml_ready (if prepared).
    """
    return _get_dataset_info_impl(
        dataset_id=dataset_id,
        loaded_datasets=_loaded_datasets,
        processed_datasets=_processed_datasets,
    )


# ===========================================================================
# ML MODEL TOOLS
# ===========================================================================

# ---------------------------------------------------------------------------
# Low-level array helpers — not registered as MCP tools
# ---------------------------------------------------------------------------

def _train_model_arrays(
    features: list[list[float]],
    labels: list[float],
    ml_algorithm: Literal["RFR", "RFC", "SVC"],
    reg_class: Literal["regression", "classification", "classification-cw"],
    opt_metric: Optional[str] = None,
    cv_fold: int = 3,
    random_seed: int = 42,
    model_save_path: Optional[str] = None,
    use_default_params: bool = False,
) -> dict[str, Any]:
    """Train from raw arrays (internal helper)."""
    data     = _DataContainer(np.array(features), np.array(labels))
    ml_model = MLModel(
        data=data,
        ml_algorithm=ml_algorithm,
        opt_metric=opt_metric,
        reg_class=reg_class,
        parameters="default" if use_default_params else "grid",
        cv_fold=cv_fold,
        random_seed=random_seed,
    )

    if model_save_path is None:
        model_save_path = _default_model_path(ml_algorithm)
    model_save_path = _resolve_path(model_save_path)
    joblib.dump(ml_model.model, model_save_path)

    return {
        "best_params":              ml_model.best_params,
        "cv_best_score":            float(ml_model.cv_results.best_score_) if ml_model.cv_results is not None else None,
        "algorithm":                ml_algorithm,
        "model_trained":            True,
        "n_samples":                len(labels),
        "n_features":               len(features[0]) if features else 0,
        "hyperparameters_searched": ml_model.h_parameters,
        "hyperparameters_mode":     "default" if use_default_params else "grid_search",
        "model_path":               model_save_path,
    }


@_register
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


@_register
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

    data   = joblib.load(split_file_path)
    labels = data[f"{split}_labels"].tolist()
    result = _predict(model_path=model_path,
                      features=data[f"{split}_features"].tolist(),
                      reg_class=task)

    model_stem    = Path(model_path).stem
    is_regression = task == "regression"

    # Optional metadata columns stored by split_dataset()
    cid_key    = f"{split}_cid"
    smiles_key = f"{split}_smiles"
    cid_col    = data[cid_key].tolist()    if cid_key    in data else None
    smiles_col = data[smiles_key].tolist() if smiles_key in data else None

    # Build per-sample DataFrame
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

    # Prepend compound metadata columns when present in the split file
    if smiles_col is not None:
        df_out.insert(0, "smiles", smiles_col)
    if cid_col is not None:
        df_out.insert(0, "cid", cid_col)

    # Metrics
    metrics_dict = (
        evaluate_regression(labels=labels, predictions=result["predictions"],
                            model_id=model_stem)
        if is_regression
        else evaluate_classification(labels=labels, predictions=result["predictions"],
                                     probabilities=result.get("probabilities"),
                                     model_id=model_stem)
    )

    # Save paths — use model_stem only (it already encodes the split file name)
    # to keep filenames short enough for Windows' 260-char path limit.
    if save_path is None:
        out_dir = session_logger.session_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        base     = f"{model_stem}_{split}"
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
        base      = f"{model_stem}_{split}"
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


# ===========================================================================
# MODEL TRAINING TOOLS
# ===========================================================================

def build_model_from_split_file(
    split_file_path: str,
    algorithm: Literal["RFC", "RFR", "SVC"] = "RFC",
    task: Literal["classification", "classification-cw", "regression"] = "classification",
    cv_fold: int = 5,
    opt_metric: Optional[str] = "balanced_accuracy",
    random_seed: int = 42,
    model_save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Blocking pipeline: tune+train+eval from a split .pkl (internal helper used by run_pipeline)."""
    split_path = Path(split_file_path)
    if not split_path.exists():
        split_path = _workspace_root() / split_file_path
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_file_path}")

    split   = joblib.load(split_path)
    X_train = np.array(split["train_features"])
    y_train = np.array(split["train_labels"])
    X_test  = np.array(split["test_features"])
    y_test  = np.array(split["test_labels"])
    has_val = "val_features" in split and len(split["val_features"]) > 0
    X_val   = np.array(split["val_features"]) if has_val else None
    y_val   = np.array(split["val_labels"])   if has_val else None

    if model_save_path is None:
        model_save_path = _default_model_path(algorithm, stem=split_path.stem)

    return _run_pipeline(
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        X_val=X_val,     y_val=y_val,
        algorithm=algorithm, task=task,
        cv_fold=cv_fold, opt_metric=opt_metric,
        random_seed=random_seed,
        model_save_path=model_save_path,
    )


@_register
def train_model(
    split_file_path: str,
    algorithm: Literal["RFC", "RFR", "SVC"] = "RFC",
    task: Literal["classification", "classification-cw", "regression"] = "classification",
    cv_fold: int = 5,
    opt_metric: Optional[str] = "balanced_accuracy",
    random_seed: int = 42,
    model_save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Train and tune a model from a split .pkl file. Non-blocking — returns a job_id immediately.

    Workflow: split_dataset → THIS TOOL → check_training(job_id, model_save_path=model_save_path)

    Poll check_training(job_id, model_save_path=model_save_path) every 30 s until
    status='completed' or 'failed'. Always pass model_save_path — it enables on-disk
    fallback detection if the MCP server restarts and in-memory job state is lost.
    The full evaluation result is in check_training(...)[\"result\"] when done.

    Args:
        split_file_path: Path to .pkl produced by split_dataset(). Must contain
                         train_features, train_labels, test_features, test_labels.
        algorithm: "RFC", "RFR", or "SVC" (default "RFC"). See get_ml_info().
        task: "classification" (default), "classification-cw", or "regression".
        cv_fold: GridSearchCV folds (default 5).
        opt_metric: Scoring metric for GridSearchCV (default "balanced_accuracy").
        random_seed: Random seed (default 42).
        model_save_path: Output .pkl path. Defaults to session models/ dir.

    Returns:
        job_id, status="running", model_save_path, message with polling instruction.
    """
    split_path = Path(split_file_path)
    if not split_path.exists():
        split_path = _workspace_root() / split_file_path
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_file_path}")

    split   = joblib.load(split_path)
    X_train = np.array(split["train_features"])
    y_train = np.array(split["train_labels"])
    X_test  = np.array(split["test_features"])
    y_test  = np.array(split["test_labels"])
    has_val = "val_features" in split and len(split["val_features"]) > 0
    X_val   = np.array(split["val_features"]) if has_val else None
    y_val   = np.array(split["val_labels"])   if has_val else None

    if model_save_path is None:
        model_save_path = _default_model_path(algorithm, stem=split_path.stem)

    job_id = str(uuid.uuid4())
    _run_job_in_background(
        job_id, _run_pipeline,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        X_val=X_val,     y_val=y_val,
        algorithm=algorithm, task=task,
        cv_fold=cv_fold, opt_metric=opt_metric,
        random_seed=random_seed,
        model_save_path=model_save_path,
    )
    return {
        "job_id":          job_id,
        "status":          "running",
        "model_save_path": model_save_path,
        "message": (
            f"Training started in the background. "
            f"Call check_training('{job_id}', model_save_path='{model_save_path}') to poll for completion. "
            "Poll every 30 seconds until status is 'completed' or 'failed'."
        ),
    }


@_register
def check_training(job_id: str, model_save_path: Optional[str] = None) -> dict[str, Any]:
    """Poll a background training job started by train_model().

    Workflow: train_model → THIS TOOL (poll until done) → plot_classification_results

    Call repeatedly every 30 seconds until status is 'completed' or 'failed'.
    The full pipeline result (best_params, train/test metrics, model_path) is in
    the 'result' key when completed.

    Always pass the model_save_path returned by train_model() so that if the MCP
    server restarts (clearing in-memory job state), this tool can detect the saved
    model on disk and return status='completed' instead of raising an error.

    Args:
        job_id: The job_id returned by train_model().
        model_save_path: Path to the expected .pkl model file (returned by train_model()).
                         Used as a fallback when the job is no longer in memory.

    Returns:
        job_id, status ("running" | "completed" | "failed"), elapsed_seconds,
        and either result (on success) or error (on failure).
    """
    if job_id not in _jobs:
        # --- Fallback: job state was lost (server restart). Check disk instead. ---
        if model_save_path and Path(model_save_path).exists():
            return {
                "job_id":  job_id,
                "status":  "completed",
                "message": (
                    "Job state was lost (MCP server restarted), but the model file "
                    f"already exists on disk at '{model_save_path}'. "
                    "Training had completed successfully before the restart. "
                    "You can proceed with plot_classification_results or export_predictions."
                ),
                "model_path": model_save_path,
            }
        raise ValueError(
            f"Job '{job_id}' not found. "
            "Job IDs are ephemeral — lost if the MCP server restarts. "
            "If you have the model_save_path, pass it to check_training() to detect the "
            "saved model on disk. Otherwise re-run train_model() to create a new job."
        )
    job = _jobs[job_id]
    elapsed = round(time.time() - job["started_at"], 1)
    response: dict[str, Any] = {
        "job_id":          job_id,
        "status":          job["status"],
        "elapsed_seconds": elapsed,
    }
    if job["status"] == "running":
        response["message"] = (
            f"Still training ({elapsed}s elapsed). "
            f"Call check_training('{job_id}', model_save_path='{model_save_path}') again in 30 seconds."
        )
    elif job["status"] == "completed":
        response["result"]  = job["result"]
        response["message"] = "Training completed successfully. See 'result' for full evaluation."
    elif job["status"] == "failed":
        response["error"]   = job["error"]
        response["message"] = "Training failed. See 'error' for details."
    return response


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


# ===========================================================================
# Plot tools
# ===========================================================================

@_register
def plot_dataset_info(
    dataset_id: str,
    split_file_path: Optional[str] = None,
    column: Optional[str] = None,
    plots: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Generate exploratory plots for a loaded dataset.

    Workflow: load_dataset → compute_features → split_dataset → THIS TOOL

    Available plots (pass in `plots` list, or omit/use ["all"] for everything):
        "class_distribution"  — bar chart of label counts (always generated)
        "column_distribution" — histogram+KDE for `column` (or all numeric cols if column=None)
        "split_statistics"    — train/val/test proportions (requires split_file_path)
        "class_balance_splits"— class % per split (requires split_file_path)

    Args:
        dataset_id: ID from load_dataset().
        split_file_path: Path to a .pkl from split_dataset() — required for split plots.
        column: Specific column for column_distribution. Omit to plot all numeric cols.
        plots: List of plot names to generate, or ["all"] / None for all available.

    Returns:
        Dict mapping plot name → saved PNG path for each generated figure.
    """
    if dataset_id not in _loaded_datasets:
        raise KeyError(f"Dataset '{dataset_id}' not loaded. Call load_dataset() first.")

    df        = _loaded_datasets[dataset_id]
    label_col = df.attrs.get("label_col", "class_label")
    want_all  = not plots or plots == ["all"]
    results: dict[str, Any] = {}

    # --- class_distribution (always included) ---
    if want_all or "class_distribution" in plots:
        path = _default_plot_path(f"{dataset_id}_class_distribution")
        _plot_class_distribution(df[label_col].tolist(), title=f"{dataset_id} — class distribution", save_path=path)
        results["class_distribution"] = path

    # --- column_distribution ---
    if want_all or "column_distribution" in plots:
        cols = [column] if column else [c for c in df.columns if df[c].dtype.kind in "iufc" and c != label_col]
        for col in cols:
            p = _default_plot_path(f"{dataset_id}_col_dist_{col}")
            _plot_column_distribution(df, column=col, title=f"{dataset_id} — {col}", save_path=p)
            results[f"column_distribution_{col}"] = p

    # --- split_statistics and class_balance_splits ---
    if split_file_path and (want_all or "split_statistics" in plots or "class_balance_splits" in plots):
        split_path = Path(split_file_path)
        if not split_path.exists():
            split_path = _workspace_root() / split_file_path
        if split_path.exists():
            split_data = joblib.load(split_path)
            # Build statistics dict expected by _plot_split_statistics
            split_stats: dict[str, dict] = {}
            for part in ("train", "val", "test"):
                lbl_key = f"{part}_labels"
                if lbl_key in split_data and len(split_data[lbl_key]) > 0:
                    split_stats[part] = {"count": len(split_data[lbl_key])}
            total = sum(v["count"] for v in split_stats.values()) or 1
            for v in split_stats.values():
                v["percentage"] = round(v["count"] / total * 100, 1)

            if want_all or "split_statistics" in plots:
                p = _default_plot_path(f"{dataset_id}_split_statistics")
                _plot_split_statistics(split_stats, title=f"{dataset_id} — split sizes", save_path=p)
                results["split_statistics"] = p

            if want_all or "class_balance_splits" in plots:
                import collections
                class_dist: dict[str, dict] = {}
                for part in ("train", "val", "test"):
                    lbl_key = f"{part}_labels"
                    if lbl_key in split_data and len(split_data[lbl_key]) > 0:
                        counts = collections.Counter(split_data[lbl_key].tolist())
                        class_dist[part] = dict(counts)
                if class_dist:
                    p = _default_plot_path(f"{dataset_id}_class_balance_splits")
                    _plot_class_balance_splits(class_dist, title=f"{dataset_id} — class balance per split", save_path=p)
                    results["class_balance_splits"] = p

    results["generated"] = list(results.keys())
    return results


@_register
def plot_classification_results(
    model_path: str,
    split_file_path: str,
    split: Literal["train", "val", "test"] = "test",
    plots: Optional[list[str]] = None,
    top_n_features: int = 20,
) -> dict[str, Any]:
    """Generate classification evaluation plots from a model and split file.

    Workflow: check_training → THIS TOOL

    Loads the model and split data from disk, runs predictions internally,
    and produces all requested figures without passing arrays through the LLM.

    Available plots (use ["all"] or omit for all):
        "confusion_matrix"   — annotated heatmap (binary + multiclass)
        "roc_curve"          — ROC curve with AUC (binary only)
        "pr_curve"           — Precision-Recall curve with AP (binary only)
        "metric_bar"         — horizontal bar of scalar metrics
        "feature_importance" — top-N Gini importances (tree models only)
        "threshold_metrics"  — Precision/Recall/F1 vs threshold (binary only)

    Args:
        model_path: Absolute path to a saved .pkl model file.
        split_file_path: Path to .pkl split file from split_dataset().
        split: Which partition to evaluate (default "test").
        plots: Plot names to generate, or ["all"] / None for all.
        top_n_features: N features for the importance plot (default 20).

    Returns:
        Dict mapping plot name → saved PNG path for each generated figure.
    """
    model_path_r = Path(model_path)
    if not model_path_r.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    split_path = Path(split_file_path)
    if not split_path.exists():
        split_path = _workspace_root() / split_file_path
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_file_path}")

    model      = joblib.load(model_path_r)
    split_data = joblib.load(split_path)
    X          = np.array(split_data[f"{split}_features"])
    y_true     = split_data[f"{split}_labels"].tolist()

    y_pred  = model.predict(X).tolist()
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    is_binary = len(set(y_true)) == 2
    y_score   = y_proba[:, 1].tolist() if (y_proba is not None and is_binary) else None

    want_all = not plots or plots == ["all"]
    stem     = f"{split_path.stem}_{Path(model_path).stem}_{split}"
    results: dict[str, Any] = {}

    if want_all or "confusion_matrix" in plots:
        p = _default_plot_path(f"{stem}_confusion_matrix")
        _plot_confusion_matrix(y_true, y_pred, title=f"Confusion matrix — {split}", save_path=p)
        results["confusion_matrix"] = p

    if (want_all or "roc_curve" in plots) and is_binary and y_score is not None:
        p = _default_plot_path(f"{stem}_roc_curve")
        _plot_roc_curve(y_true, y_score, title=f"ROC curve — {split}", save_path=p)
        results["roc_curve"] = p

    if (want_all or "pr_curve" in plots) and is_binary and y_score is not None:
        p = _default_plot_path(f"{stem}_pr_curve")
        _plot_pr_curve(y_true, y_score, title=f"PR curve — {split}", save_path=p)
        results["pr_curve"] = p

    if want_all or "metric_bar" in plots:
        metrics = evaluate_classification(labels=y_true, predictions=y_pred,
                                          probabilities=y_proba.tolist() if y_proba is not None else None)
        p = _default_plot_path(f"{stem}_metric_bar")
        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        _plot_metric_bar(scalar_metrics, title=f"Metrics — {split}", save_path=p)
        results["metric_bar"] = p
        results["metrics"] = scalar_metrics

    if want_all or "feature_importance" in plots:
        try:
            p = _default_plot_path(f"{stem}_feature_importance")
            _plot_feature_importance(model, top_n=top_n_features,
                                      title=f"Feature importance (top {top_n_features})", save_path=p)
            results["feature_importance"] = p
        except (AttributeError, ValueError):
            results["feature_importance"] = "skipped (model does not expose feature_importances_)"

    if (want_all or "threshold_metrics" in plots) and is_binary and y_score is not None:
        p = _default_plot_path(f"{stem}_threshold_metrics")
        _plot_threshold_metrics(y_true, y_score, title=f"Threshold metrics — {split}", save_path=p)
        results["threshold_metrics"] = p

    results["generated"] = [k for k in results if k != "metrics"]
    return results


@_register
def plot_regression_results(
    model_path: str,
    split_file_path: str,
    split: Literal["train", "val", "test"] = "test",
    plots: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Generate regression evaluation plots from a model and split file.

    Workflow: check_training → THIS TOOL

    Loads model and split data from disk; runs predictions internally.

    Available plots (use ["all"] or omit for all):
        "actual_vs_predicted" — scatter with identity line, R² and MAE
        "residuals"           — residuals vs fitted scatter
        "residual_histogram"  — histogram + KDE of residuals
        "error_distribution"  — histogram + KDE of |y_true − y_pred|

    Args:
        model_path: Absolute path to a saved .pkl model file.
        split_file_path: Path to .pkl split file from split_dataset().
        split: Which partition to evaluate (default "test").
        plots: Plot names to generate, or ["all"] / None for all.

    Returns:
        Dict mapping plot name → saved PNG path for each generated figure.
    """
    model_path_r = Path(model_path)
    if not model_path_r.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    split_path = Path(split_file_path)
    if not split_path.exists():
        split_path = _workspace_root() / split_file_path
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_file_path}")

    model      = joblib.load(model_path_r)
    split_data = joblib.load(split_path)
    X          = np.array(split_data[f"{split}_features"])
    y_true     = split_data[f"{split}_labels"].tolist()
    y_pred     = model.predict(X).tolist()

    want_all = not plots or plots == ["all"]
    stem     = f"{split_path.stem}_{Path(model_path).stem}_{split}"
    results: dict[str, Any] = {}

    if want_all or "actual_vs_predicted" in plots:
        p = _default_plot_path(f"{stem}_actual_vs_predicted")
        _plot_actual_vs_predicted(y_true, y_pred, title=f"Actual vs Predicted — {split}", save_path=p)
        results["actual_vs_predicted"] = p

    if want_all or "residuals" in plots:
        p = _default_plot_path(f"{stem}_residuals")
        _plot_residuals(y_true, y_pred, title=f"Residuals — {split}", save_path=p)
        results["residuals"] = p

    if want_all or "residual_histogram" in plots:
        p = _default_plot_path(f"{stem}_residual_histogram")
        _plot_residual_histogram(y_true, y_pred, title=f"Residual histogram — {split}", save_path=p)
        results["residual_histogram"] = p

    if want_all or "error_distribution" in plots:
        p = _default_plot_path(f"{stem}_error_distribution")
        _plot_error_distribution(y_true, y_pred, title=f"Error distribution — {split}", save_path=p)
        results["error_distribution"] = p

    results["generated"] = list(results.keys())
    return results


# ===========================================================================
# Shortcut tool
# ===========================================================================

@_register
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
    # 1. Load
    df, meta = load_csv(
        file_path=file_path, label_col=label_col, smiles_col=smiles_col,
        id_col=id_col, directory="",
    )
    ds_id = meta["dataset_id"]
    _loaded_datasets[ds_id] = df
    session_logger.save_dataframe(df, ds_id)

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


# ===========================================================================
# Session / utility tools
# ===========================================================================

@_register
def log_thought(
    thought: str,
    step: Optional[str] = None,
) -> dict[str, str]:
    """Record a reasoning or planning step in the session log.

    Call this to capture chain-of-thought, observations, or decisions in the
    session log. This is the only way the LLM's reasoning reaches the log.

    Args:
        thought: Reasoning, plan, observation, or decision text.
        step: Optional phase label ("plan", "observation", "decision", "summary").

    Returns:
        {"logged": "ok", "session_id": <id>}
    """
    session_logger.log_thought(thought, step=step)
    return {"logged": "ok", "session_id": session_logger.session_id}


@_register
def start_new_session() -> dict[str, str]:
    """Start a fresh logging session, ending the current one immediately.

    Use this at the beginning of a new chat or experiment to ensure
    artifacts and logs are not mixed with a previous session.
    Without calling this, sessions are automatically continued as long as
    the last activity was within the session timeout window (default 60 min).

    Returns:
        {"new_session_id": <id>, "session_dir": <path>}
    """
    new_id = session_logger.force_new_session()
    return {
        "new_session_id": new_id,
        "session_dir":    str(session_logger.session_dir),
    }


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
