"""
Low-level metric computation functions.

All functions are pure (no side effects) and operate on plain NumPy arrays,
making them easy to test and reuse independently of any model class.

Usage
-----
    from chemagent.ml.metrics import confusion_components, classification_metrics

    FP, FN, TP, TN = confusion_components(y_true, y_pred)
    result = classification_metrics(y_true, y_pred, y_proba)
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics


# Confusion-matrix decomposition
def confusion_components(
    labels: np.ndarray,
    pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a confusion matrix into FP, FN, TP, TN per class.

    Args:
    labels:
        Ground-truth class labels.
    pred:
        Predicted class labels.

    Returns:
    tuple[ndarray, ndarray, ndarray, ndarray]
        (FP, FN, TP, TN) — each is a 1-D array with one value per class.
    """
    cm = metrics.confusion_matrix(labels, pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    return FP, FN, TP, TN


# Classification metrics
def classification_metrics(
    labels: np.ndarray,
    pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    model_id: str | None = None,
    model_type: str | None = None,
) -> dict:
    """Compute classification performance metrics.

    Handles both binary and multiclass scenarios automatically.

    Args:
    labels:
        Ground-truth class labels (1-D integer array).
    pred:
        Predicted class labels (1-D integer array).
    y_proba:
        Predicted class probabilities (n_samples, n_classes). Optional.
    model_id:
        Algorithm name stored in the result dict (informational).
    model_type:
        Target / dataset identifier stored in the result dict (informational).

    Returns:
    dict
        Dictionary of all computed metrics.
    """
    labels = np.array(labels)
    pred = np.array(pred)
    y_proba = np.array(y_proba) if y_proba is not None else None

    FP, FN, TP, TN = confusion_components(labels, pred)
    is_binary = np.unique(labels).shape[0] == 2

    result: dict = {
        "MCC": float(metrics.matthews_corrcoef(labels, pred)),
        "BA": float(metrics.balanced_accuracy_score(labels, pred)),
        "Accuracy": float(metrics.accuracy_score(labels, pred)),
        "Dataset size": len(labels),
        "Target ID": model_type,
        "Algorithm": model_id,
        "FP": FP.tolist(),
        "FN": FN.tolist(),
        "TP": TP.tolist(),
        "TN": TN.tolist(),
    }

    if y_proba is not None:
        result["Probability"] = y_proba.tolist()

    if is_binary:
        score = y_proba[:, 1] if y_proba is not None else pred
        result.update(
            {
                "F1": float(metrics.f1_score(labels, pred)),
                "AUC": float(metrics.roc_auc_score(labels, score)),
                "Precision": float(metrics.precision_score(labels, pred)),
                "Recall": float(metrics.recall_score(labels, pred)),
                "Average Precision": float(
                    metrics.average_precision_score(labels, score)
                ),
            }
        )
    else:
        result.update(
            {
                "F1 weighted": float(
                    metrics.f1_score(labels, pred, average="weighted")
                ),
                "F1 macro": float(metrics.f1_score(labels, pred, average="macro")),
                "Precision macro": float(
                    metrics.precision_score(labels, pred, average="macro")
                ),
                "Recall macro": float(
                    metrics.recall_score(labels, pred, average="macro")
                ),
                "Precision micro": float(
                    metrics.precision_score(labels, pred, average="micro")
                ),
                "Recall micro": float(
                    metrics.recall_score(labels, pred, average="micro")
                ),
            }
        )

    return result


def multiclass_metrics(
    labels: np.ndarray,
    pred: np.ndarray,
    model_id: str | None = None,
    model_type: str | None = None,
) -> dict:
    """Compute per-class and overall metrics for a multiclass problem.

    Args:
    labels:
        Ground-truth class labels.
    pred:
        Predicted class labels.
    model_id:
        Algorithm name (informational).
    model_type:
        Target / dataset identifier (informational).

    Returns:
    dict
        Nested dict with ``"overall_metrics"``, ``"per_class_metrics"``,
        ``"confusion_matrix"``, and ``"class_labels"``.
    """
    labels = np.array(labels)
    pred = np.array(pred)
    class_labels = np.unique(labels)

    per_class: dict = {}
    for cls in class_labels:
        bl = (labels == cls).astype(int)
        bp = (pred == cls).astype(int)
        per_class[f"Class_{cls}"] = {
            "Precision": float(metrics.precision_score(bl, bp, zero_division=0)),
            "Recall": float(metrics.recall_score(bl, bp, zero_division=0)),
            "F1": float(metrics.f1_score(bl, bp, zero_division=0)),
            "Support": int(np.sum(labels == cls)),
        }

    return {
        "target": model_type,
        "algorithm": model_id,
        "overall_metrics": {
            "MCC": float(metrics.matthews_corrcoef(labels, pred)),
            "Accuracy": float(metrics.accuracy_score(labels, pred)),
            "BA": float(metrics.balanced_accuracy_score(labels, pred)),
            "F1 macro": float(metrics.f1_score(labels, pred, average="macro")),
            "F1 weighted": float(metrics.f1_score(labels, pred, average="weighted")),
        },
        "per_class_metrics": per_class,
        "confusion_matrix": metrics.confusion_matrix(labels, pred).tolist(),
        "class_labels": class_labels.tolist(),
    }


# Regression metrics
def regression_metrics(
    labels: np.ndarray,
    pred: np.ndarray,
    model_id: str | None = None,
    model_type: str | None = None,
) -> dict:
    """Compute regression performance metrics.

    Args:
    labels:
        Ground-truth continuous values.
    pred:
        Model-predicted continuous values.
    model_id:
        Algorithm name (informational).
    model_type:
        Target / dataset identifier (informational).

    Returns:
    dict
        Dictionary containing MAE, MSE, RMSE, R², and Pearson r.
    """
    labels = np.array(labels)
    pred = np.array(pred)
    r = float(np.corrcoef(labels, pred)[0, 1])

    return {
        "MAE": float(metrics.mean_absolute_error(labels, pred)),
        "MSE": float(metrics.mean_squared_error(labels, pred)),
        "RMSE": float(metrics.root_mean_squared_error(labels, pred)),
        "R2": float(metrics.r2_score(labels, pred)),
        "r": r,
        "Dataset size": len(labels),
        "Target ID": model_type,
        "Algorithm": model_id,
    }
