"""server_helpers.py — pure internal helpers for chemagent_mcp.py.

Contains functions that have **no dependency** on mcp, session_logger, the
in-memory state dicts (_loaded_datasets, _processed_datasets, _jobs), or any
other server-lifecycle objects.  These can be unit-tested independently.

Imported by chemagent_mcp.py via:
    from chemagent.servers.server_helpers import (...)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal, Optional

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Make chemagent packages importable when this module is imported standalone
# (chemagent_mcp.py already inserts _SRC before importing us, but guard here
# too for independent use, e.g. in tests or notebooks).
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chemagent.ml import MLModel, Model_Evaluation
from chemagent.ml.hyperparameter_tuning import HYPERPARAMETERS


# ===========================================================================
# Path utilities
# ===========================================================================

def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(p: str) -> str:
    """Resolve *p* against workspace root when it is a relative path.

    This prevents files from being written into the server's cwd
    (src/chemagent/servers/) when the server is launched via ``uv --directory``.
    """
    path = Path(p)
    if not path.is_absolute():
        path = _workspace_root() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


# ===========================================================================
# Serialisation utils
# ===========================================================================

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


# ===========================================================================
# ML data container
# ===========================================================================

class _DataContainer:
    """Minimal data container accepted by :class:`chemagent.ml.training.MLModel`."""
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features     = features
        self.labels       = labels
        self.class_labels = labels


# ===========================================================================
# Evaluation helpers
# ===========================================================================

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


def evaluate_classification(
    labels: list[int],
    predictions: list[int],
    probabilities: Optional[list[list[float]]] = None,
    model_id: str = "Model",
    model_type: Optional[str] = None,
) -> dict[str, Any]:
    """Classification metrics — binary and multi-class (internal helper)."""
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


def evaluate_regression(
    labels: list[float],
    predictions: list[float],
    model_id: str = "Model",
    model_type: Optional[str] = None,
) -> dict[str, Any]:
    """Regression metrics: MAE, MSE, RMSE, R², Pearson r (internal helper)."""
    if len(predictions) != len(labels):
        raise ValueError(
            f"predictions ({len(predictions)}) and labels ({len(labels)}) length mismatch"
        )
    return _build_evaluator(labels, predictions, None, "regression", model_id, model_type
                            ).prediction_performance_regression()


# ===========================================================================
# Core ML pipeline
# ===========================================================================

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

    model_save_path = _resolve_path(model_save_path)
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
        "n_train":                  int(len(y_train)),
        "n_test":                   int(len(y_test)),
        "n_val":                    int(len(y_val)) if y_val is not None else 0,
        "n_features":               int(X_train.shape[1]),
    }


def _predict(
    model_path: str,
    features: list[list[float]],
    reg_class: Literal["regression", "classification", "classification-cw"],
) -> dict[str, Any]:
    """Run inference from a saved model (internal helper)."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    X     = np.array(features)
    preds = model.predict(X).tolist()
    proba: Optional[list[list[float]]] = None
    if reg_class in ("classification", "classification-cw") and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X).tolist()
    return {
        "predictions":   preds,
        "probabilities": proba,
        "n_samples":     len(preds),
        "model_path":    model_path,
        "reg_class":     reg_class,
    }


# ===========================================================================
# Split-file training helper
# ===========================================================================

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
        model_save_path: Where to save the model. Defaults to a models/
                         directory under the workspace root.
        use_default_params: Skip GridSearchCV when True.

    Returns:
        Same as train_model().
    """
    from chemagent.ml.training import MLModel as _MLModel  # local import avoids circular ref in tests

    data = joblib.load(split_file_path)
    if model_save_path is None:
        out_dir = _workspace_root() / "data" / "models"
        out_dir.mkdir(exist_ok=True)
        model_save_path = str(out_dir / f"{Path(split_file_path).stem}_{ml_algorithm}.pkl")

    _data = _DataContainer(
        np.array(data[f"{split}_features"]),
        np.array(data[f"{split}_labels"]),
    )
    ml_model = _MLModel(
        data=_data,
        ml_algorithm=ml_algorithm,
        opt_metric=opt_metric,
        reg_class=reg_class,
        parameters="default" if use_default_params else "grid",
        cv_fold=cv_fold,
        random_seed=random_seed,
    )

    model_save_path = _resolve_path(model_save_path)
    joblib.dump(ml_model.model, model_save_path)

    labels   = data[f"{split}_labels"].tolist()
    features = data[f"{split}_features"].tolist()

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


# ===========================================================================
# Hyperparameter reference
# ===========================================================================

def get_hyperparameter_grids() -> dict[str, Any]:
    """Return all registered hyperparameter grids (internal helper)."""
    return _to_serialisable(HYPERPARAMETERS)
