"""
chemagent.plots.plot_tools
~~~~~~~~~~~~~~~~~~~~~~~~~~
MCP tool functions that generate evaluation plots from predictions CSVs.
Imported and registered by chemagent_mcp.py via _register().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
from mcp.server.fastmcp import Image

from chemagent.plots.classification import (
    plot_confusion_matrix as _plot_confusion_matrix,
    plot_metric_bar as _plot_metric_bar,
    plot_pr_curve as _plot_pr_curve,
    plot_roc_curve as _plot_roc_curve,
    plot_threshold_metrics as _plot_threshold_metrics,
)
from chemagent.plots.regression import (
    plot_actual_vs_predicted as _plot_actual_vs_predicted,
    plot_error_distribution as _plot_error_distribution,
    plot_residual_histogram as _plot_residual_histogram,
    plot_residuals as _plot_residuals,
)
from chemagent.servers.server_helpers import (
    _workspace_root,
    evaluate_classification,
    evaluate_regression,
)
from chemagent.session_utils import get_session_logger as _get_session_logger


def _default_plot_path(name: str, ext: str = "png") -> str:
    out_dir = _get_session_logger().session_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{name}.{ext}")


def plot_classification_results(
    predictions_path: str,
    plots: Optional[list[str]] = None,
) -> list:
    """Generate classification evaluation plots from a predictions CSV.

    Workflow: export_predictions → THIS TOOL

    Reads the CSV produced by export_predictions() — no model or split file needed.
    Images are returned inline so they render directly in the chat UI.

    Available plots (use ["all"] or omit for all):
        "confusion_matrix"   — annotated heatmap (binary + multiclass)
        "roc_curve"          — ROC curve with AUC (binary only)
        "pr_curve"           — Precision-Recall curve with AP (binary only)
        "metric_bar"         — horizontal bar of scalar metrics
        "threshold_metrics"  — Precision/Recall/F1 vs threshold (binary only)

    Args:
        predictions_path: Path to the predictions CSV from export_predictions().
                          Must contain columns: true_label, predicted_label,
                          and optionally prob_class_0, prob_class_1.
        plots: Plot names to generate, or ["all"] / None for all.

    Returns:
        List starting with a summary dict (plot names → saved paths, metrics),
        followed by inline Image objects that render directly in the chat UI.
    """
    pred_path = Path(predictions_path)
    if not pred_path.exists():
        pred_path = _workspace_root() / predictions_path
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    df     = pd.read_csv(pred_path)
    y_true = df["true_label"].tolist()
    y_pred = df["predicted_label"].tolist()

    prob_cols = sorted([c for c in df.columns if c.startswith("prob_class_")])
    y_proba   = df[prob_cols].values if prob_cols else None
    is_binary = len(set(y_true)) == 2
    y_score   = df["prob_class_1"].tolist() if ("prob_class_1" in df.columns and is_binary) else None

    want_all = not plots or plots == ["all"]
    stem     = pred_path.stem
    results: dict[str, Any] = {}

    if want_all or "confusion_matrix" in plots:
        p = _default_plot_path(f"{stem}_confusion_matrix")
        _plot_confusion_matrix(y_true, y_pred, title="Confusion matrix", save_path=p)
        results["confusion_matrix"] = p

    if (want_all or "roc_curve" in plots) and is_binary and y_score is not None:
        p = _default_plot_path(f"{stem}_roc_curve")
        _plot_roc_curve(y_true, y_score, title="ROC curve", save_path=p)
        results["roc_curve"] = p

    if (want_all or "pr_curve" in plots) and is_binary and y_score is not None:
        p = _default_plot_path(f"{stem}_pr_curve")
        _plot_pr_curve(y_true, y_score, title="PR curve", save_path=p)
        results["pr_curve"] = p

    if want_all or "metric_bar" in plots:
        metrics = evaluate_classification(
            labels=y_true, predictions=y_pred,
            probabilities=y_proba.tolist() if y_proba is not None else None,
        )
        p = _default_plot_path(f"{stem}_metric_bar")
        if "overall_metrics" in metrics:
            scalar_metrics = {k: v for k, v in metrics["overall_metrics"].items() if isinstance(v, (int, float))}
        else:
            scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        _plot_metric_bar(scalar_metrics, title="Metrics", save_path=p)
        results["metric_bar"] = p
        results["metrics"] = scalar_metrics

    if (want_all or "threshold_metrics" in plots) and is_binary and y_score is not None:
        p = _default_plot_path(f"{stem}_threshold_metrics")
        _plot_threshold_metrics(y_true, y_score, title="Threshold metrics", save_path=p)
        results["threshold_metrics"] = p

    plot_keys = [k for k in results if k not in ("generated", "metrics")]
    results["generated"] = plot_keys
    results["next_step"] = (
        "Call show_plot(path) for each generated plot path to display it in the chat UI. "
        + "Paths: " + ", ".join(f"{k}={results[k]!r}" for k in plot_keys)
    )
    images = [Image(path=results[k]) for k in plot_keys]
    return [results, *images]


def plot_regression_results(
    predictions_path: str,
    plots: Optional[list[str]] = None,
) -> list:
    """Generate regression evaluation plots from a predictions CSV.

    Workflow: export_predictions → THIS TOOL

    Reads the CSV produced by export_predictions() — no model or split file needed.
    Images are returned inline so they render directly in the chat UI.

    Available plots (use ["all"] or omit for all):
        "actual_vs_predicted" — scatter with identity line, R² and MAE
        "residuals"           — residuals vs fitted scatter
        "residual_histogram"  — histogram + KDE of residuals
        "error_distribution"  — histogram + KDE of |y_true − y_pred|

    Args:
        predictions_path: Path to the predictions CSV from export_predictions().
                          Must contain columns: true_label, predicted_value.
        plots: Plot names to generate, or ["all"] / None for all.

    Returns:
        List starting with a summary dict (plot names → saved paths),
        followed by inline Image objects that render directly in the chat UI.
    """
    pred_path = Path(predictions_path)
    if not pred_path.exists():
        pred_path = _workspace_root() / predictions_path
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    df     = pd.read_csv(pred_path)
    y_true = df["true_label"].tolist()
    y_pred = df["predicted_value"].tolist()

    want_all = not plots or plots == ["all"]
    stem     = pred_path.stem
    results: dict[str, Any] = {}

    if want_all or "actual_vs_predicted" in plots:
        p = _default_plot_path(f"{stem}_actual_vs_predicted")
        _plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted", save_path=p)
        results["actual_vs_predicted"] = p

    if want_all or "residuals" in plots:
        p = _default_plot_path(f"{stem}_residuals")
        _plot_residuals(y_true, y_pred, title="Residuals", save_path=p)
        results["residuals"] = p

    if want_all or "residual_histogram" in plots:
        p = _default_plot_path(f"{stem}_residual_histogram")
        _plot_residual_histogram(y_true, y_pred, title="Residual histogram", save_path=p)
        results["residual_histogram"] = p

    if want_all or "error_distribution" in plots:
        p = _default_plot_path(f"{stem}_error_distribution")
        _plot_error_distribution(y_true, y_pred, title="Error distribution", save_path=p)
        results["error_distribution"] = p

    plot_keys = [k for k in results if k != "generated"]
    results["generated"] = plot_keys
    results["next_step"] = (
        "Call show_plot(path) for each generated plot path to display it in the chat UI. "
        + "Paths: " + ", ".join(f"{k}={results[k]!r}" for k in plot_keys)
    )
    images = [Image(path=results[k]) for k in plot_keys]
    return [results, *images]
