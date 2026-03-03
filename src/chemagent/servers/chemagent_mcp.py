"""chemagent_mcp.py — single consolidated FastMCP server.

Exposes all agent tools in one process:

  Dataset tools (formerly dataset_loader_mcp)
  ─────────────────────────────────────────────
  list_available_datasets      list CSV files in a directory
  list_loaded_datasets         inspect in-memory state
  list_featurizers             discover available fingerprint methods
  load_dataset                 load any CSV for ML
  get_dataset_smiles           retrieve SMILES for external featurization
  featurize_dataset            compute fingerprints server-side (preferred)
  prepare_ml_dataset           pair a dataset with externally computed features
  split_prepared_dataset       create train/val/test splits and save .pkl
  load_split                   reload a saved split .pkl
  get_ml_ready_data            return feature matrix + labels (explicit flow)
  get_dataset_info             inspect a dataset's status

  ML model tools (formerly ml_models_mcp)
  ─────────────────────────────────────────────
  train_model                  train + tune from raw arrays
  predict                      inference from a saved .pkl model
  evaluate_classification      classification metrics (binary + multi-class)
  evaluate_regression          regression metrics
  get_available_algorithms     discover algorithm info and hyperparameter grids
  get_recommended_metrics      metric guidance per task type

  Model-builder tools (formerly model_builder_mcp)
  ─────────────────────────────────────────────
  build_model_from_split_file  full pipeline (tune+train+eval) from split .pkl  [blocking]
  start_model_training         same pipeline as background job, returns job_id immediately
  get_training_result          poll a background job started by start_model_training
  build_model_from_arrays      full pipeline from feature arrays
  get_hyperparameter_grids     inspect registered hyperparameter grids

  Dataset plot tools
  ─────────────────────────────────────────────
  plot_class_distribution      bar chart of label counts with % annotations
  plot_split_statistics        stacked bar of train/val/test proportions
  plot_column_distribution     histogram + KDE of any numeric dataset column
  plot_class_balance_splits    class share (%) per split — grouped bars
  plot_dataset_comparison      sample-count comparison across datasets

  Classification plot tools
  ─────────────────────────────────────────────
  plot_confusion_matrix        annotated heatmap (binary + multiclass)
  plot_roc_curve               ROC curve with AUC (binary)
  plot_pr_curve                Precision-Recall curve with AP (binary)
  plot_metric_bar              horizontal bar of scalar evaluation metrics
  plot_feature_importance      top-N Gini importances (tree-based models)
  plot_threshold_metrics       Precision/Recall/F1 vs threshold (binary)

  Regression plot tools
  ─────────────────────────────────────────────
  plot_actual_vs_predicted     scatter with identity line, R² and MAE
  plot_residuals               residuals vs fitted scatter
  plot_residual_histogram      histogram + KDE of residuals
  plot_error_distribution      histogram + KDE of |y_true - y_pred|

Preferred end-to-end workflow (data stays on disk):
    load_dataset(...)
    featurize_dataset(dataset_id, method="ECFP", n_bits=2048, radius=2)
    split_prepared_dataset(dataset_id, train_size=0.7, test_size=0.3)
    job = start_model_training(split_file_path, algorithm="RFC",
                               task="classification",
                               opt_metric="balanced_accuracy")
    # poll until done:
    result = get_training_result(job["job_id"])
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

mcp = FastMCP("chemagent")

# ---------------------------------------------------------------------------
# Session logger — writes data/logs/session_<timestamp>_<id>.txt
# ---------------------------------------------------------------------------
_log_dir = Path(__file__).resolve().parents[3] / "data" / "logs"
session_logger = SessionLogger(_log_dir)

# Patch mcp.tool so EVERY subsequent @mcp.tool() decoration is automatically
# wrapped with logging — no changes needed on individual tool functions.
_real_mcp_tool = mcp.tool

def _logging_tool_decorator(*dargs, **dkwargs):
    """Replacement for mcp.tool() that injects call/result logging."""
    decorator = _real_mcp_tool(*dargs, **dkwargs)

    def wrap(fn):
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
        return decorator(logged_fn)

    return wrap

mcp.tool = _logging_tool_decorator

# ---------------------------------------------------------------------------
# In-memory state  (ephemeral — lost on server restart)
# ---------------------------------------------------------------------------
_loaded_datasets:    dict[str, Any] = {}   # dataset_id → pd.DataFrame
_processed_datasets: dict[str, dict[str, Any]] = {}

# Background training jobs  {job_id → {status, result, error, started_at, finished_at}}
_jobs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Shared internal helpers
# ---------------------------------------------------------------------------

def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_model_path(algorithm: str, stem: str = "") -> str:
    out_dir = _workspace_root() / "data" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{stem}_{algorithm}.pkl" if stem else f"trained_model_{algorithm}.pkl"
    return str(out_dir / name)


def _default_plot_path(name: str, ext: str = "png") -> str:
    """Return an auto-generated path inside data/plots/."""
    out_dir = _workspace_root() / "data" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{name}.{ext}")


def _to_serialisable(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to plain Python types."""
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


class _DataContainer:
    """Minimal data container accepted by :class:`chemagent.ml.training.MLModel`."""
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features     = features
        self.labels       = labels
        self.class_labels = labels


def _build_evaluator(labels, predictions, probabilities, reg_class, model_id, model_type):
    is_regression = reg_class == "regression"
    return Model_Evaluation(
        labels=labels,
        y_pred=None if is_regression else predictions,
        y_proba=probabilities,
        y_pred_reg=predictions if is_regression else None,
        model_id=model_id,
        model_type=model_type,
        reg_class=reg_class,
    )


def _run_job_in_background(job_id: str, fn, *args, **kwargs) -> None:
    """Run *fn* in a daemon thread; write result/error into _jobs[job_id]."""
    def _worker():
        t_start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            _jobs[job_id]["status"]      = "completed"
            _jobs[job_id]["result"]      = result
            session_logger.log_event(
                "background_job_completed",
                job_id=job_id,
                duration_ms=round((time.perf_counter() - t_start) * 1000, 2),
            )
            # Copy model and any other files produced by the background job
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
    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def _run_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    X_val:   np.ndarray | None,
    y_val:   np.ndarray | None,
    algorithm: str,
    task: str,
    cv_fold: int,
    opt_metric: str | None,
    random_seed: int,
    model_save_path: str,
) -> dict[str, Any]:
    """Core pipeline: tune → train → save → evaluate (used by build_model_* tools)."""
    data = _DataContainer(X_train, y_train)
    ml_model = MLModel(
        data=data,
        ml_algorithm=algorithm,
        opt_metric=opt_metric,
        reg_class=task,
        cv_fold=cv_fold,
        random_seed=random_seed,
    )

    model_save_path = str(Path(model_save_path).resolve())
    joblib.dump(ml_model.model, model_save_path)

    is_regression = task == "regression"

    def _evaluate(X: np.ndarray, y: np.ndarray, split_name: str) -> dict:
        y_pred  = ml_model.model.predict(X)
        y_proba = (
            ml_model.model.predict_proba(X)
            if not is_regression and hasattr(ml_model.model, "predict_proba")
            else None
        )
        ev = _build_evaluator(y, y_pred, y_proba, task, algorithm, split_name)
        if is_regression:
            return _to_serialisable(ev.prediction_performance_regression())
        n_classes = len(np.unique(y))
        if n_classes == 2:
            df = ev.pred_performance_class
            return _to_serialisable(
                {row["Metric"]: row["Value"] for _, row in df.iterrows()}
            )
        return _to_serialisable(ev.prediction_performance_multiclass())

    return {
        "algorithm":                algorithm,
        "task":                     task,
        "cv_fold":                  cv_fold,
        "opt_metric":               opt_metric,
        "best_params":              _to_serialisable(ml_model.best_params),
        "cv_best_score":            float(ml_model.cv_results.best_score_) if ml_model.cv_results is not None else None,
        "model_path":               model_save_path,
        "hyperparameters_searched": _to_serialisable(HYPERPARAMETERS.get(algorithm, {})),
        "train_evaluation":         _evaluate(X_train, y_train, "train"),
        "test_evaluation":          _evaluate(X_test,  y_test,  "test"),
        "val_evaluation":           _evaluate(X_val, y_val, "val") if X_val is not None else None,
        "n_train":                  int(len(y_train)),
        "n_test":                   int(len(y_test)),
        "n_val":                    int(len(y_val)) if y_val is not None else 0,
        "n_features":               int(X_train.shape[1]),
    }


# ===========================================================================
# DATASET TOOLS
# ===========================================================================

@mcp.tool()
def list_available_datasets(directory: str = "data/datasets") -> dict[str, Any]:
    """List CSV files in any workspace-relative or absolute directory.

    Args:
        directory: Path to search for CSV files. May be absolute or relative
                   to the workspace root (default: "data/datasets").

    Returns:
        - datasets: List of CSV filenames found
        - count: Number of files
        - directory: Resolved absolute path searched
    """
    return list_csv_files(directory)


@mcp.tool()
def list_loaded_datasets() -> dict[str, Any]:
    """List all datasets currently in server memory.

    IMPORTANT — state is ephemeral. All data is lost when the MCP server
    restarts. If lists are empty, re-run load_dataset() and (if needed)
    featurize_dataset() before continuing.

    Returns:
        - loaded: dataset_ids in raw memory
        - prepared: dataset_ids ready for splitting
    """
    return {
        "loaded":         list(_loaded_datasets.keys()),
        "prepared":       list(_processed_datasets.keys()),
        "total_loaded":   len(_loaded_datasets),
        "total_prepared": len(_processed_datasets),
        "note":           "State is ephemeral — lost on server restart.",
    }


@mcp.tool()
def list_featurizers() -> dict[str, Any]:
    """List all available molecular featurization methods.

    Returns name, parameters, and one-line description for every public
    featurizer in chemagent.featurization. Use the method name directly as the
    `method` argument to featurize_dataset().
    """
    return _list_featurizers_impl()


@mcp.tool()
def load_dataset(
    file_path: str,
    label_col: str = "class_label",
    smiles_col: Optional[str] = "smiles",
    id_col: Optional[str] = None,
    feature_cols: Optional[list[str]] = None,
    dataset_id: Optional[str] = None,
    directory: str = "",
) -> dict[str, Any]:
    """Load any CSV dataset for machine learning.

    Supports three dataset types:
      1. Molecular datasets with SMILES (featurize later with featurize_dataset)
      2. Tabular datasets with named numeric feature columns (feature_cols)
      3. Tabular datasets where all numeric columns except label/id are features
         (auto-detection when feature_cols=None and smiles_col=None)

    Examples:
        # Molecular dataset — featurize server-side afterwards:
        load_dataset("data/datasets/chembl_activity_data_O00329_P42336.csv",
                     label_col="class_label", smiles_col="smiles", id_col="cid")

        # Tabular, explicit feature columns:
        load_dataset("data/my_data.csv", label_col="target",
                     smiles_col=None, feature_cols=["f1", "f2", "f3"])

        # Tabular, auto-detect numeric features:
        load_dataset("data/my_data.csv", label_col="target", smiles_col=None)

    Args:
        file_path: Absolute path OR filename within `directory` OR workspace-relative path.
        label_col: Column to use as ML target (default: "class_label").
        smiles_col: Column containing SMILES strings, or None if not present.
        id_col: Column containing compound / sample IDs (optional).
        feature_cols: Explicit list of numeric feature column names. Ignored if
                      smiles_col is set.
        dataset_id: Key for in-memory caching (default: stem of file_path).
        directory: Directory prefix applied when file_path is not absolute.

    Returns:
        dataset_id, n_samples, columns, label_col, label_stats, has_smiles,
        has_precomputed_features, n_features (if pre-computed), smiles_sample,
        next_step.
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


@mcp.tool()
def get_dataset_smiles(dataset_id: str) -> dict[str, Any]:
    """Retrieve SMILES strings from a loaded dataset.

    Use this when featurizing externally. The returned list can be passed to
    any featurizer, then features passed to prepare_ml_dataset().

    Args:
        dataset_id: ID from load_dataset().

    Returns:
        - smiles: Full list of SMILES strings
        - n_samples: Number of molecules
    """
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


@mcp.tool()
def featurize_dataset(
    dataset_id: str,
    method: str = "ECFP",
    n_bits: int = 2048,
    radius: int = 2,
    label_col: Optional[str] = None,
) -> dict[str, Any]:
    """Compute molecular fingerprints server-side and prepare dataset for ML.

    Requires the dataset to have a SMILES column. Features are stored
    server-side — nothing large is returned to the LLM context.

    Any public UpperCase function in chemagent/featurization/fingerprints.py
    is automatically available as a method. Call list_featurizers() to see options.

    Common methods:
        "ECFP"  — Morgan circular fingerprints (ECFP4: radius=2, n_bits=2048)
        "MACCS" — 166-bit structural MACCS keys

    Args:
        dataset_id: ID from load_dataset().
        method: Featurizer name (default: "ECFP"). See list_featurizers().
        n_bits: Passed to featurizer if accepted (default: 2048).
        radius: Passed to featurizer if accepted (default: 2).
        label_col: Override label column. Defaults to the column set in load_dataset().

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
            f"Call split_prepared_dataset('{dataset_id}', train_size=0.7, "
            "val_size=0.0, test_size=0.3, stratified=True) to create splits."
        ),
    }


@mcp.tool()
def prepare_ml_dataset(
    dataset_id: str,
    features: list[list[float]],
    label_col: Optional[str] = None,
) -> dict[str, Any]:
    """Pair a loaded dataset with externally computed features.

    Use this when featurization was done outside the server (e.g. via a
    separate tool) and you want to store the result server-side before splitting.

    Args:
        dataset_id: ID from load_dataset().
        features: 2D feature matrix, shape (n_samples, n_features).
        label_col: Override label column (defaults to what was set in load_dataset()).

    Returns:
        dataset_id, n_samples, n_features, label_stats, prepared=True.
    """
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


@mcp.tool()
def split_prepared_dataset(
    dataset_id: str,
    split_type: Literal["random", "scaffold"] = "random",
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: Optional[int] = 42,
    stratified: bool = False,
    save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Split a prepared dataset into train / val / test sets and save to disk.

    Splits are saved as a .pkl file so train_on_split_file and
    build_model_from_split_file can load them without transferring feature
    arrays through the LLM context.

    Examples:
        # Stratified 70/30 train/test (no validation):
        split_prepared_dataset(dataset_id, train_size=0.7, val_size=0.0,
                               test_size=0.3, stratified=True)

        # Scaffold split with validation:
        split_prepared_dataset(dataset_id, split_type="scaffold",
                               train_size=0.8, val_size=0.1, test_size=0.1)

        # Skip saving:
        split_prepared_dataset(dataset_id, save_path="")

    Args:
        dataset_id: ID of the prepared dataset.
        split_type: "random" or "scaffold".
        train_size: Fraction for training (default 0.8).
        val_size: Fraction for validation (default 0.1). Set to 0.0 for a
                  two-way train/test split.
        test_size: Fraction for test (default 0.1).
        seed: Random seed (default 42).
        stratified: Preserve class proportions (random splits only).
        save_path: Where to save the .pkl. Defaults to
                   "data/splits/<dataset_id>_<split_type>.pkl". Pass "" to skip.

    Returns:
        train/val/test metadata, statistics, saved_to path, next_step hint.
    """
    if dataset_id not in _processed_datasets:
        raise ValueError(
            f"Dataset '{dataset_id}' not prepared. "
            "Call featurize_dataset() or prepare_ml_dataset() first."
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
            f"Call start_model_training(split_file_path='{saved_to}', "
            "algorithm='RFC', task='classification', opt_metric='balanced_accuracy') "
            "to train in the background (non-blocking). "
            "Then poll with get_training_result(job_id) until status='completed'. "
            "Features are on disk — not returned here."
        ) if saved_to else "No file saved. Pass save_path or use prepare_ml_dataset().",
    }
    if warnings_list:
        result["warnings"] = warnings_list
    return result


@mcp.tool()
def load_split(file_path: str) -> dict[str, Any]:
    """Load a previously saved train/val/test split from a .pkl file.

    Args:
        file_path: Absolute or workspace-relative path to a .pkl file produced
                   by split_prepared_dataset().

    Returns:
        'train', 'val', 'test' keys each containing features, labels, n_samples
        (and smiles / cid if available). Plus 'file_path'.
    """
    return load_split_file(file_path)


@mcp.tool()
def get_ml_ready_data(dataset_id: str, as_lists: bool = True) -> dict[str, Any]:
    """Return the processed feature matrix and labels for a prepared dataset.

    Use this only when features must be passed explicitly through the LLM
    context. For the standard workflow prefer split_prepared_dataset() →
    build_model_from_split_file() to keep data on disk.

    Args:
        dataset_id: ID of a featurized / prepared dataset.
        as_lists: True (default) — return features and labels as JSON lists.
                  False — return shape/metadata only.

    Returns:
        dataset_id, shape, label_column, and (if as_lists) features, labels,
        smiles, cid.
    """
    if dataset_id not in _processed_datasets:
        raise ValueError(
            f"Dataset '{dataset_id}' not prepared. "
            "Call featurize_dataset() or prepare_ml_dataset() first."
        )
    return _get_ml_ready_data_impl(
        processed=_processed_datasets[dataset_id],
        dataset_id=dataset_id,
        as_lists=as_lists,
    )


@mcp.tool()
def get_dataset_info(dataset_id: str) -> dict[str, Any]:
    """Get status and metadata for a loaded or prepared dataset.

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

@mcp.tool()
def train_model(
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
    """Train a machine learning model with hyperparameter optimization via GridSearchCV.

    Supported algorithms:
        RFC — Random Forest Classifier
        RFR — Random Forest Regressor
        SVC — Support Vector Classifier

    Standard vs optimized:
        use_default_params=True  → fast single fit ("standard hyperparameters")
        use_default_params=False → GridSearchCV tuning ("optimized hyperparameters")

    Args:
        features: 2D array of features, shape (n_samples, n_features).
        labels: Target values or class labels.
        ml_algorithm: "RFC", "RFR", or "SVC".
        reg_class: "regression", "classification", or "classification-cw".
        opt_metric: GridSearchCV scoring metric (e.g. "f1", "roc_auc").
        cv_fold: Number of CV folds (default 3).
        random_seed: Random seed (default 42).
        model_save_path: Where to save the .pkl. Defaults to
                         "data/models/trained_model_<algorithm>.pkl".
        use_default_params: Skip GridSearchCV when True (faster).

    Returns:
        best_params, cv_best_score, algorithm, model_trained, n_samples,
        n_features, model_path, hyperparameters_mode.
    """
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
        out_dir = _workspace_root() / "data" / "models"
        out_dir.mkdir(exist_ok=True)
        model_save_path = str(out_dir / f"trained_model_{ml_algorithm}.pkl")
    model_save_path = str(Path(model_save_path).resolve())
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


@mcp.tool()
def predict(
    model_path: str,
    features: list[list[float]],
    reg_class: Literal["regression", "classification", "classification-cw"],
) -> dict[str, Any]:
    """Generate predictions from a saved model.

    Args:
        model_path: Absolute path to a .pkl model produced by train_model().
        features: 2D list of input features, shape (n_samples, n_features).
        reg_class: Task type of the loaded model.

    Returns:
        predictions, probabilities (classification only), n_samples, model_path,
        reg_class.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    X     = np.array(features)
    preds = model.predict(X).tolist()
    proba: Optional[list[list[float]]] = None
    if reg_class in ("classification", "classification-cw") and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X).tolist()
    return {
        "predictions":  preds,
        "probabilities": proba,
        "n_samples":    len(preds),
        "model_path":   model_path,
        "reg_class":    reg_class,
    }


@mcp.tool()
def evaluate_classification(
    labels: list[int],
    predictions: list[int],
    probabilities: Optional[list[list[float]]] = None,
    model_id: str = "Model",
    model_type: Optional[str] = None,
) -> dict[str, Any]:
    """Evaluate classification performance (binary and multi-class).

    Binary: MCC, F1, AUC, Balanced Accuracy, Precision, Recall, TP/TN/FP/FN.
    Multi-class: overall metrics + per-class breakdown + confusion matrix.

    Args:
        labels: Ground-truth class labels.
        predictions: Predicted class labels.
        probabilities: Class probabilities (n_samples, n_classes) — optional,
                       used for AUC in binary tasks.
        model_id: Label for the model (default "Model").
        model_type: Optional type identifier.
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"predictions ({len(predictions)}) and labels ({len(labels)}) length mismatch"
        )
    evaluator = _build_evaluator(labels, predictions, probabilities,
                                 "classification", model_id, model_type)
    is_binary = len(np.unique(labels)) == 2
    if is_binary:
        df    = evaluator.pred_performance_class
        result: dict[str, Any] = {row["Metric"]: row["Value"] for _, row in df.iterrows()}
        result["is_binary"] = True
        return result
    return evaluator.prediction_performance_multiclass()


@mcp.tool()
def evaluate_regression(
    labels: list[float],
    predictions: list[float],
    model_id: str = "Model",
    model_type: Optional[str] = None,
) -> dict[str, Any]:
    """Evaluate regression model performance.

    Returns MAE, MSE, RMSE, R², Pearson r.

    Args:
        labels: Ground-truth values.
        predictions: Model predictions.
        model_id: Label for the model (default "Model").
        model_type: Optional type identifier.
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"predictions ({len(predictions)}) and labels ({len(labels)}) length mismatch"
        )
    return _build_evaluator(labels, predictions, None, "regression", model_id, model_type
                            ).prediction_performance_regression()


@mcp.tool()
def get_available_algorithms() -> dict[str, dict[str, Any]]:
    """Get information about available ML algorithms and their hyperparameter grids.

    Returns:
        Dict mapping algorithm codes to name, task_type, hyperparameters,
        supports_multiclass, supports_class_weight, description.
    """
    return {
        "RFR": {
            "name": "Random Forest Regressor",
            "task_type": "regression",
            "hyperparameters": HYPERPARAMETERS.get("RFR", {}),
            "supports_multiclass": False,
            "supports_class_weight": False,
            "description": "Ensemble of decision trees for regression tasks",
        },
        "RFC": {
            "name": "Random Forest Classifier",
            "task_type": "classification",
            "hyperparameters": HYPERPARAMETERS.get("RFC", {}),
            "supports_multiclass": True,
            "supports_class_weight": True,
            "description": "Ensemble of decision trees for classification, handles multi-class",
        },
        "SVC": {
            "name": "Support Vector Classifier",
            "task_type": "classification",
            "hyperparameters": HYPERPARAMETERS.get("SVC", {}),
            "supports_multiclass": True,
            "supports_class_weight": True,
            "description": "SVM classifier with RBF/linear kernels, probability estimates enabled",
        },
    }


@mcp.tool()
def get_recommended_metrics() -> dict[str, Any]:
    """Get recommended evaluation metrics per task type.

    Returns:
        Dict mapping task types to optimization and evaluation metric lists.
        Covers binary_classification, binary_imbalanced, multiclass, regression.
    """
    return {
        "binary_classification": {
            "optimization": ["f1", "roc_auc", "average_precision", "accuracy"],
            "evaluation":   ["MCC", "F1", "Precision", "Recall", "AUC",
                             "Average Precision", "Balanced Accuracy"],
        },
        "binary_imbalanced": {
            "optimization": ["f1", "average_precision", "roc_auc"],
            "evaluation":   ["MCC", "F1", "Precision", "Recall", "Balanced Accuracy"],
            "note":         "Use reg_class='classification-cw' for automatic class weighting",
        },
        "multiclass": {
            "optimization": ["f1_macro", "f1_weighted", "balanced_accuracy", "accuracy"],
            "evaluation":   ["MCC", "Balanced Accuracy", "F1_macro", "F1_weighted",
                             "Precision_macro", "Recall_macro", "Accuracy"],
        },
        "regression": {
            "optimization": ["neg_mean_squared_error", "neg_root_mean_squared_error",
                             "neg_mean_absolute_error", "r2"],
            "evaluation":   ["RMSE", "MAE", "MSE", "R2", "Pearson r"],
        },
    }


def train_on_split_file(
    split_file_path: str,
    ml_algorithm: Literal["RFR", "RFC", "SVC"],
    reg_class: Literal["regression", "classification", "classification-cw"],
    split: Literal["train", "val", "test"] = "train",
    opt_metric: Optional[str] = None,
    cv_fold: int = 5,
    random_seed: int = 42,
    model_save_path: Optional[str] = None,
    use_default_params: bool = False,
) -> dict[str, Any]:
    """Train a model directly from a saved .pkl split file.

    Preferred over train_model() when a split file exists. Features are
    loaded from disk and never transferred through the LLM context.

    Typical workflow:
        featurize_dataset(dataset_id, method="ECFP", n_bits=2048, radius=2)
        splits = split_prepared_dataset(dataset_id, train_size=0.7,
                                        val_size=0.0, test_size=0.3,
                                        stratified=True)
        result = train_on_split_file(splits["saved_to"], "RFC",
                                     "classification", opt_metric="f1")
        preds  = predict_from_split_file(result["model_path"], splits["saved_to"],
                                         "classification")

    Args:
        split_file_path: Path to .pkl produced by split_prepared_dataset().
        ml_algorithm: "RFR", "RFC", or "SVC".
        reg_class: "regression", "classification", or "classification-cw".
        split: Which partition to train on (default: "train").
        opt_metric: GridSearchCV scoring metric.
        cv_fold: Number of CV folds (default 5).
        random_seed: Random seed (default 42).
        model_save_path: Where to save the model. Defaults to
                         "data/models/<split_stem>_<algorithm>.pkl".
        use_default_params: Skip GridSearchCV when True.

    Returns:
        Same as train_model().
    """
    data = joblib.load(split_file_path)
    if model_save_path is None:
        out_dir = _workspace_root() / "data" / "models"
        out_dir.mkdir(exist_ok=True)
        model_save_path = str(out_dir / f"{Path(split_file_path).stem}_{ml_algorithm}.pkl")
    return train_model(
        features=data[f"{split}_features"].tolist(),
        labels=data[f"{split}_labels"].tolist(),
        ml_algorithm=ml_algorithm,
        reg_class=reg_class,
        opt_metric=opt_metric,
        cv_fold=cv_fold,
        random_seed=random_seed,
        model_save_path=model_save_path,
        use_default_params=use_default_params,
    )


def predict_from_split_file(
    model_path: str,
    split_file_path: str,
    reg_class: Literal["regression", "classification", "classification-cw"],
    split: Literal["train", "val", "test"] = "test",
    results_save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Run inference on a split .pkl, auto-evaluate, and save results to disk.

    Preferred over predict() when features were saved by split_prepared_dataset().
    Results and metrics are automatically saved to data/results/.

    Args:
        model_path: Path to .pkl model produced by train_model() or
                    train_on_split_file().
        split_file_path: Path to the .pkl split file.
        reg_class: Task type.
        split: Which partition to predict on (default: "test").
        results_save_path: Where to save predictions. Defaults to
                           "data/results/<split_stem>_<model_stem>_<split>_predictions.pkl".
                           Metrics are saved alongside with "_metrics.pkl" suffix.

    Returns:
        All fields from predict() plus: labels, metrics, results_path, metrics_path.
    """
    data   = joblib.load(split_file_path)
    labels = data[f"{split}_labels"].tolist()
    result = predict(model_path=model_path,
                     features=data[f"{split}_features"].tolist(),
                     reg_class=reg_class)
    result["labels"] = labels

    model_stem       = Path(model_path).stem
    is_regression    = reg_class == "regression"
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
        out_dir = _workspace_root() / "data" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        base      = f"{Path(split_file_path).stem}_{model_stem}_{split}"
        results_save_path = str(out_dir / f"{base}_predictions.pkl")
        metrics_save_path = str(out_dir / f"{base}_metrics.pkl")
    else:
        results_save_path = str(Path(results_save_path).resolve())
        metrics_save_path = str(Path(results_save_path).with_name(
            Path(results_save_path).stem + "_metrics.pkl"
        ))

    Path(results_save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result,      results_save_path)
    joblib.dump(metrics_dict, metrics_save_path)
    result["results_path"] = results_save_path
    result["metrics_path"] = metrics_save_path
    return result


# ===========================================================================
# MODEL BUILDER TOOLS
# ===========================================================================

@mcp.tool()
def build_model_from_split_file(
    split_file_path: str,
    algorithm: Literal["RFC", "RFR", "SVC"] = "RFC",
    task: Literal["classification", "classification-cw", "regression"] = "classification",
    cv_fold: int = 5,
    opt_metric: Optional[str] = "balanced_accuracy",
    random_seed: int = 42,
    model_save_path: Optional[str] = None,
) -> dict[str, Any]:
    """[BLOCKING] Train, tune, and evaluate a full pipeline from a split .pkl file.

    WARNING: This tool blocks until training finishes. For any representation
    larger than MACCS (e.g. ECFP, RDKitFP, AtomPairFP) or any large dataset
    this will time out. Use start_model_training() instead — it is identical
    but returns immediately and never times out.

    Runs: hyperparameter tuning → GridSearchCV → refit → save → evaluate all splits.

    PREFERRED alternative (use this):
        job = start_model_training(split_file_path, algorithm="RFC", ...)
        result = get_training_result(job["job_id"])  # poll until completed

    Only use build_model_from_split_file for quick smoke-tests with MACCS keys
    where training is known to complete in a few seconds.

    Args:
        split_file_path: Path to .pkl split file from split_prepared_dataset().
                         Must contain train_features, train_labels,
                         test_features, test_labels. Optional val_*.
        algorithm: "RFC", "RFR", or "SVC".
        task: "classification", "classification-cw", or "regression".
        cv_fold: Number of GridSearchCV folds (default 5).
        opt_metric: Scoring string for GridSearchCV (default "balanced_accuracy").
                    Use None for estimator default.
        random_seed: Random seed (default 42).
        model_save_path: Where to save the model .pkl. Defaults to
                         "data/models/<split_stem>_<algorithm>.pkl".

    Returns:
        algorithm, task, cv_fold, opt_metric, best_params, cv_best_score,
        model_path, hyperparameters_searched, train_evaluation, test_evaluation,
        val_evaluation (or None), n_train, n_test, n_val, n_features.
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

    return _run_pipeline(
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        X_val=X_val,     y_val=y_val,
        algorithm=algorithm, task=task,
        cv_fold=cv_fold, opt_metric=opt_metric,
        random_seed=random_seed,
        model_save_path=model_save_path,
    )


@mcp.tool()
def start_model_training(
    split_file_path: str,
    algorithm: Literal["RFC", "RFR", "SVC"] = "RFC",
    task: Literal["classification", "classification-cw", "regression"] = "classification",
    cv_fold: int = 5,
    opt_metric: Optional[str] = "balanced_accuracy",
    random_seed: int = 42,
    model_save_path: Optional[str] = None,
) -> dict[str, Any]:
    """PREFERRED tool for end-to-end model training. Non-blocking, works for all
    molecular representations and dataset sizes (ECFP, MACCS, RDKitFP, etc.).

    Starts training in the background and returns a job_id immediately —
    the tool call never times out regardless of how long training takes.
    Poll get_training_result(job_id) every 15-30 s until status='completed'.

    Always use this instead of build_model_from_split_file.

    Workflow:
        job = start_model_training(split_file_path, algorithm="RFC", ...)
        result = get_training_result(job["job_id"])   # poll until status='completed'

    Args:
        split_file_path: Path to .pkl split file from split_prepared_dataset().
        algorithm: "RFC", "RFR", or "SVC".
        task: "classification", "classification-cw", or "regression".
        cv_fold: Number of GridSearchCV folds (default 5).
        opt_metric: Scoring string for GridSearchCV (default "balanced_accuracy").
        random_seed: Random seed (default 42).
        model_save_path: Where to save the model .pkl. Defaults to
                         data/models/<split_stem>_<algorithm>.pkl.

    Returns:
        job_id, status="running", message with polling instructions.
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
        job_id,
        _run_pipeline,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        X_val=X_val,     y_val=y_val,
        algorithm=algorithm, task=task,
        cv_fold=cv_fold, opt_metric=opt_metric,
        random_seed=random_seed,
        model_save_path=model_save_path,
    )
    return {
        "job_id":  job_id,
        "status":  "running",
        "message": (
            f"Training started in the background. "
            f"Call get_training_result('{job_id}') to poll for completion. "
            "Keep polling every 15-30 seconds until status is 'completed' or 'failed'."
        ),
    }


@mcp.tool()
def get_training_result(job_id: str) -> dict[str, Any]:
    """Poll the status of a background training job started by start_model_training.

    Call this repeatedly (every 15-30 seconds) until status is "completed" or "failed".
    When completed, the full training result (same as build_model_from_split_file) is
    returned inside the "result" key.

    Args:
        job_id: The job_id returned by start_model_training().

    Returns:
        job_id, status ("running" | "completed" | "failed"),
        elapsed_seconds, and either result (on success) or error (on failure).
    """
    if job_id not in _jobs:
        raise ValueError(
            f"Job '{job_id}' not found. "
            "Job IDs are ephemeral — they are lost if the MCP server restarts. "
            "Re-run start_model_training() to create a new job."
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
            f"Call get_training_result('{job_id}') again in 15-30 seconds."
        )
    elif job["status"] == "completed":
        response["result"]  = job["result"]
        response["message"] = "Training completed successfully. See 'result' for full evaluation."
    elif job["status"] == "failed":
        response["error"]   = job["error"]
        response["message"] = "Training failed. See 'error' for details."
    return response


@mcp.tool()
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
    """Train, tune, and evaluate a full pipeline from feature arrays.

    Use only when a split file is not available. For larger datasets prefer
    build_model_from_split_file to avoid bloating the LLM context.

    Runs the same full pipeline as build_model_from_split_file.

    Args:
        train_features, train_labels, test_features, test_labels: Data arrays.
        algorithm: "RFC", "RFR", or "SVC".
        task: "classification", "classification-cw", or "regression".
        cv_fold: Number of CV folds (default 5).
        opt_metric: GridSearchCV scoring string.
        random_seed: Random seed (default 42).
        model_save_path: Where to save the model .pkl.
        val_features, val_labels: Optional validation set.

    Returns:
        Same structure as build_model_from_split_file.
    """
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


@mcp.tool()
def get_hyperparameter_grids() -> dict[str, Any]:
    """Return all registered hyperparameter grids for RFC, RFR, and SVC.

    Useful to verify what parameter ranges will be searched during
    build_model_from_split_file before running the full pipeline.

    Returns:
        Dict mapping algorithm keys to their parameter grids.
    """
    return _to_serialisable(HYPERPARAMETERS)


# ===========================================================================
# Plot tools — dataset
# ===========================================================================

@mcp.tool()
def plot_class_distribution(
    labels: list,
    class_names: Optional[list[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict[str, str]:
    """Bar chart of class counts with percentage annotations.

    Saves the figure to disk and returns the file path.  Useful to inspect
    the label balance of any dataset or split partition.

    Args:
        labels:      List of class labels (integers or strings).
        class_names: Display names for each class (must match number of
                     unique labels).  Omit to auto-generate.
        title:       Axes title.
        save_path:   Where to write the PNG.  Auto-generated if omitted.

    Returns:
        {"saved_to": <absolute path to PNG>}
    """
    path = save_path or _default_plot_path("class_distribution")
    _plot_class_distribution(
        labels,
        class_names=class_names,
        title=title,
        save_path=path,
    )
    return {"saved_to": path}


@mcp.tool()
def plot_split_statistics(
    split_stats: dict[str, Any],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict[str, str]:
    """Horizontal stacked bar showing train / val / test partition proportions.

    Args:
        split_stats: Statistics dict as returned by split_prepared_dataset
                     (keys: "train", "val", "test", each with "count" and
                     "percentage" sub-keys).
        title:       Axes title.
        save_path:   Where to write the PNG.  Auto-generated if omitted.

    Returns:
        {"saved_to": <absolute path to PNG>}
    """
    path = save_path or _default_plot_path("split_statistics")
    _plot_split_statistics(split_stats, title=title, save_path=path)
    return {"saved_to": path}


@mcp.tool()
def plot_column_distribution(
    dataset_id: str,
    column: str,
    hue: Optional[str] = None,
    bins: int = 0,
    kde: bool = True,
    reference_line: Optional[float] = None,
    reference_label: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict[str, str]:
    """Histogram + optional KDE of any numeric column in a loaded dataset.

    The dataset must have been loaded first with load_dataset.

    Args:
        dataset_id:      Dataset key (as returned by load_dataset).
        column:          Column name to plot.
        hue:             Optional column name used to colour sub-distributions.
        bins:            Number of histogram bins.  0 = automatic.
        kde:             Overlay KDE curve.
        reference_line:  Optional x-value for a vertical reference line.
        reference_label: Legend label for the reference line.
        title:           Axes title.
        save_path:       Where to write the PNG.  Auto-generated if omitted.

    Returns:
        {"saved_to": <absolute path to PNG>}
    """
    if dataset_id not in _loaded_datasets:
        raise KeyError(f"Dataset '{dataset_id}' not loaded. Call load_dataset first.")
    df = _loaded_datasets[dataset_id]
    bins_arg: int | str = bins if bins > 0 else "auto"
    path = save_path or _default_plot_path(f"col_dist_{column}")
    _plot_column_distribution(
        df,
        column=column,
        hue=hue,
        bins=bins_arg,
        kde=kde,
        reference_line=reference_line,
        reference_label=reference_label,
        title=title,
        save_path=path,
    )
    return {"saved_to": path}


@mcp.tool()
def plot_class_balance_splits(
    class_dist: dict[str, dict[str, int]],
    class_names: Optional[list[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict[str, str]:
    """Grouped bar chart — class share (%) within each data split.

    Args:
        class_dist:  Dict mapping partition names ("train", "val", "test")
                     to {class_label: count} sub-dicts.
        class_names: Display names for each class.
        title:       Axes title.
        save_path:   Where to write the PNG.  Auto-generated if omitted.

    Returns:
        {"saved_to": <absolute path to PNG>}
    """
    path = save_path or _default_plot_path("class_balance_splits")
    _plot_class_balance_splits(
        class_dist,
        class_names=class_names,
        title=title,
        save_path=path,
    )
    return {"saved_to": path}


@mcp.tool()
def plot_dataset_comparison(
    counts: dict[str, int],
    xlabel: str = "Dataset",
    ylabel: str = "Sample count",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict[str, str]:
    """Bar chart comparing sample counts across multiple datasets or groups.

    Args:
        counts:    {group_label: count} mapping, e.g.
                   {"PI3Kd vs PI3Ka": 1277, "PI3Kd vs PI3Kg": 980}.
        xlabel:    X-axis label (default "Dataset").
        ylabel:    Y-axis label (default "Sample count").
        title:     Axes title.
        save_path: Where to write the PNG.  Auto-generated if omitted.

    Returns:
        {"saved_to": <absolute path to PNG>}
    """
    path = save_path or _default_plot_path("dataset_comparison")
    _plot_dataset_comparison(
        counts,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        save_path=path,
    )
    return {"saved_to": path}


# ===========================================================================
# Plot tools — classification
# ===========================================================================

@mcp.tool()
def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    class_names: Optional[list[str]] = None,
    normalise: bool = False,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict[str, str]:
    """Annotated confusion-matrix heatmap.

    Works for binary and multiclass problems.  class_names must match the
    number of unique labels in y_true; omit to auto-generate from label values.

    Args:
        y_true:      Ground-truth labels.
        y_pred:      Predicted labels.
        class_names: Display labels for each class.  Omit to auto-generate.
        normalise:   Row-normalise (show rates) instead of raw counts.
        title:       Axes title.
        save_path:   Where to write the PNG.  Auto-generated if omitted.

    Returns:
        {"saved_to": <absolute path to PNG>}
    """
    path = save_path or _default_plot_path("confusion_matrix")
    _plot_confusion_matrix(
        y_true, y_pred,
        class_names=class_names,
        normalise=normalise,
        title=title,
        save_path=path,
    )
    return {"saved_to": path}


@mcp.tool()
def plot_roc_curve(
    y_true: list,
    y_score: list,
    label: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict[str, str]:
    """ROC curve with AUC annotation (binary classification only).

    Args:
        y_true:    Binary ground-truth labels (0 / 1).
        y_score:   Predicted probabilities for the positive class.
        label:     Legend entry (e.g. model name).
        title:     Axes title.
        save_path: Where to write the PNG.  Auto-generated if omitted.

    Returns:
        {"saved_to": <absolute path to PNG>}
    """
    path = save_path or _default_plot_path("roc_curve")
    _plot_roc_curve(y_true, y_score, label=label, title=title, save_path=path)
    return {"saved_to": path}


@mcp.tool()
def plot_pr_curve(
    y_true: list,
    y_score: list,
    label: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict[str, str]:
    """Precision-Recall curve with Average Precision annotation (binary only).

    Args:
        y_true:    Binary ground-truth labels (0 / 1).
        y_score:   Predicted probabilities for the positive class.
        label:     Legend entry.
        title:     Axes title.
        save_path: Where to write the PNG.  Auto-generated if omitted.

    Returns:
        {"saved_to": <absolute path to PNG>}
    """
    path = save_path or _default_plot_path("pr_curve")
    _plot_pr_curve(y_true, y_score, label=label, title=title, save_path=path)
    return {"saved_to": path}


@mcp.tool()
def plot_metric_bar(
    metrics_dict: dict[str, float],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict[str, str]:
    """Horizontal bar chart of scalar evaluation metrics.

    Only scalar metrics in [0, 1] whose names appear in the standard set
    (MCC, BA, Accuracy, F1, AUC, Precision, Recall, etc.) are plotted;
    list or dict values are silently skipped.

    For multiclass evaluation dicts that nest metrics under "overall_metrics",
    pass metrics_dict["overall_metrics"] directly.

    Args:
        metrics_dict: {metric_name: value} flat mapping of scalar scores.
        title:        Axes title.
        save_path:    Where to write the PNG.  Auto-generated if omitted.

    Returns:
        {"saved_to": <absolute path to PNG>}
    """
    path = save_path or _default_plot_path("metric_bar")
    _plot_metric_bar(metrics_dict, title=title, save_path=path)
    return {"saved_to": path}


@mcp.tool()
def plot_feature_importance(
    model_path: str,
    top_n: int = 20,
    feature_names: Optional[list[str]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict[str, str]:
    """Top-N feature importances bar chart for tree-based models (e.g. RFC).

    Loads the model from disk.  GridSearchCV wrappers are unwrapped
    automatically via best_estimator_.

    Args:
        model_path:    Absolute path to a saved .pkl model file.
        top_n:         Number of top features to display (default 20).
        feature_names: Display names for features.  Defaults to f0, f1, ...
        title:         Axes title.
        save_path:     Where to write the PNG.  Auto-generated if omitted.

    Returns:
        {"saved_to": <absolute path to PNG>}
    """
    model = joblib.load(model_path)
    path = save_path or _default_plot_path("feature_importance")
    _plot_feature_importance(
        model,
        feature_names=feature_names,
        top_n=top_n,
        title=title,
        save_path=path,
    )
    return {"saved_to": path}


@mcp.tool()
def plot_threshold_metrics(
    y_true: list,
    y_score: list,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict[str, str]:
    """Precision, Recall, and F1 vs decision threshold (binary classification).

    Useful to choose an operating threshold other than 0.5.

    Args:
        y_true:    Binary ground-truth labels (0 / 1).
        y_score:   Predicted probabilities for the positive class.
        title:     Axes title.
        save_path: Where to write the PNG.  Auto-generated if omitted.

    Returns:
        {"saved_to": <absolute path to PNG>}
    """
    path = save_path or _default_plot_path("threshold_metrics")
    _plot_threshold_metrics(y_true, y_score, title=title, save_path=path)
    return {"saved_to": path}


# ===========================================================================
# Plot tools — regression
# ===========================================================================

@mcp.tool()
def plot_actual_vs_predicted(
    y_true: list,
    y_pred: list,
    model_id: Optional[str] = None,
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict[str, str]:
    """Scatter of actual vs predicted values with identity line (regression).

    Annotates R² and MAE in the plot.

    Args:
        y_true:    Ground-truth target values.
        y_pred:    Model predictions.
        model_id:  Model name shown in the legend.
        xlabel:    X-axis label (default "Actual").
        ylabel:    Y-axis label (default "Predicted").
        title:     Axes title.
        save_path: Where to write the PNG.  Auto-generated if omitted.

    Returns:
        {"saved_to": <absolute path to PNG>}
    """
    path = save_path or _default_plot_path("actual_vs_predicted")
    _plot_actual_vs_predicted(
        y_true, y_pred,
        model_id=model_id,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        save_path=path,
    )
    return {"saved_to": path}


@mcp.tool()
def plot_residuals(
    y_true: list,
    y_pred: list,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict[str, str]:
    """Residuals (y_true − y_pred) vs fitted values scatter (regression).

    A horizontal zero line and a loess-style trend line are drawn.  Systematic
    curvature indicates model mis-specification.

    Args:
        y_true:    Ground-truth values.
        y_pred:    Model predictions.
        title:     Axes title.
        save_path: Where to write the PNG.  Auto-generated if omitted.

    Returns:
        {"saved_to": <absolute path to PNG>}
    """
    path = save_path or _default_plot_path("residuals")
    _plot_residuals(y_true, y_pred, title=title, save_path=path)
    return {"saved_to": path}


@mcp.tool()
def plot_residual_histogram(
    y_true: list,
    y_pred: list,
    bins: int = 0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict[str, str]:
    """Histogram + KDE of prediction residuals (y_true − y_pred) (regression).

    Args:
        y_true:    Ground-truth values.
        y_pred:    Model predictions.
        bins:      Number of histogram bins.  0 = automatic.
        title:     Axes title.
        save_path: Where to write the PNG.  Auto-generated if omitted.

    Returns:
        {"saved_to": <absolute path to PNG>}
    """
    bins_arg: int | str = bins if bins > 0 else "auto"
    path = save_path or _default_plot_path("residual_histogram")
    _plot_residual_histogram(y_true, y_pred, bins=bins_arg, title=title, save_path=path)
    return {"saved_to": path}


@mcp.tool()
def plot_error_distribution(
    y_true: list,
    y_pred: list,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict[str, str]:
    """Histogram + KDE of absolute prediction errors |y_true − y_pred| (regression).

    Marks the MAE with a vertical dashed line.

    Args:
        y_true:    Ground-truth values.
        y_pred:    Model predictions.
        title:     Axes title.
        save_path: Where to write the PNG.  Auto-generated if omitted.

    Returns:
        {"saved_to": <absolute path to PNG>}
    """
    path = save_path or _default_plot_path("error_distribution")
    _plot_error_distribution(y_true, y_pred, title=title, save_path=path)
    return {"saved_to": path}


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
