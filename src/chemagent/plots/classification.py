"""
chemagent.plots.classification — plots for binary and multiclass ML results.

All functions are general-purpose and work with any classification problem.

Functions
---------
plot_confusion_matrix      Annotated heatmap of the confusion matrix.
plot_roc_curve             ROC curve with AUC annotation.
plot_pr_curve              Precision-Recall curve with AP annotation.
plot_metric_bar            Horizontal bar chart of scalar evaluation metrics.
plot_feature_importance    Top-N feature importances for tree-based models.
plot_threshold_metrics     Precision / Recall / F1 vs decision threshold.

Signature convention (all functions):
    plot_*(data, ..., title=None, ax=None, save_path=None) -> Figure

Pass ``ax`` to embed a plot into an existing multi-panel layout.
Pass ``save_path`` to write the figure to disk (PNG / SVG / PDF).

Example
-------
    from chemagent.plots.classification import plot_confusion_matrix, plot_roc_curve
    from chemagent.plots.utils import set_theme

    set_theme()
    fig = plot_confusion_matrix(y_true, y_pred, class_names=["neg", "pos"])
    fig = plot_roc_curve(y_true, y_proba[:, 1], save_path="results/roc.png")
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from sklearn import metrics as skmetrics

from .utils import set_theme, PALETTE, SNS_PALETTE, CMAP_SEQ, save_figure


# Confusion matrix
def plot_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    class_names: Optional[Sequence[str]] = None,
    normalise: bool = False,
    cmap: str = CMAP_SEQ,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Annotated confusion-matrix heatmap (seaborn).

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    class_names:
        Display labels for each class.  Defaults to the unique sorted values.
    normalise:
        Row-normalise (show rates) when ``True``; show raw counts otherwise.
    cmap:
        Seaborn / matplotlib colourmap name.
    title:
        Axes title.
    ax:
        Pre-existing :class:`~matplotlib.axes.Axes`.
    save_path:
        File path to save the figure.

    Returns
    -------
    Figure
    """
    set_theme()

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    classes = class_names or [str(c) for c in np.unique(y_true_arr)]

    cm = skmetrics.confusion_matrix(y_true_arr, y_pred_arr)
    cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True) if normalise else cm.astype(float)
    fmt = ".2f" if normalise else ".0f"

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(4, len(classes) * 1.6), max(3.5, len(classes) * 1.4)))
    else:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

    # annotated heatmap
    annot = np.array(
        [[f"{cm_plot[i, j]:{fmt}}\n({cm[i, j]})" if normalise else f"{int(cm[i, j])}"
          for j in range(len(classes))]
         for i in range(len(classes))]
    )
    sns.heatmap(
        cm_plot,
        annot=annot,
        fmt="s",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set(
        xlabel="Predicted label",
        ylabel="True label",
        title=title or "Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

    if standalone:
        fig.tight_layout()
        save_figure(fig, save_path, tight=False)
    return fig


# ROC curve
def plot_roc_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    label: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """ROC curve with AUC annotation.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels (0 / 1).
    y_score:
        Predicted probabilities for the positive class.
    label:
        Legend entry (e.g. model name).
    title:
        Axes title.
    ax:
        Pre-existing :class:`~matplotlib.axes.Axes`.
    save_path:
        Output file path.

    Returns
    -------
    Figure
    """
    set_theme()

    y_true_arr = np.array(y_true)
    y_score_arr = np.array(y_score)
    fpr, tpr, _ = skmetrics.roc_curve(y_true_arr, y_score_arr)
    auc = skmetrics.roc_auc_score(y_true_arr, y_score_arr)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

    lbl = f"{label} " if label else ""
    sns.lineplot(x=fpr, y=tpr, ax=ax, color=PALETTE["primary"], lw=2,
                 label=f"{lbl}AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], lw=1.2,
            linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["primary"])

    ax.set(
        xlim=(0.0, 1.0), ylim=(0.0, 1.02),
        xlabel="False Positive Rate", ylabel="True Positive Rate",
        title=title or "ROC Curve",
    )
    ax.legend(loc="lower right")

    if standalone:
        fig.tight_layout()
        save_figure(fig, save_path, tight=False)
    return fig


# Precision-Recall curve
def plot_pr_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    label: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Precision-Recall curve with Average Precision annotation.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels (0 / 1).
    y_score:
        Predicted probabilities for the positive class.
    label:
        Legend entry.
    title:
        Axes title.
    ax:
        Pre-existing :class:`~matplotlib.axes.Axes`.
    save_path:
        Output file path.

    Returns
    -------
    Figure
    """
    set_theme()

    y_true_arr = np.array(y_true)
    y_score_arr = np.array(y_score)
    precision, recall, _ = skmetrics.precision_recall_curve(y_true_arr, y_score_arr)
    ap = skmetrics.average_precision_score(y_true_arr, y_score_arr)
    baseline = float(y_true_arr.sum()) / len(y_true_arr)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

    lbl = f"{label} " if label else ""
    sns.lineplot(x=recall, y=precision, ax=ax, color=PALETTE["secondary"], lw=2,
                 label=f"{lbl}AP = {ap:.3f}")
    ax.fill_between(recall, precision, alpha=0.08, color=PALETTE["secondary"])
    ax.axhline(baseline, color=PALETTE["neutral"], lw=1.2, linestyle="--",
               label=f"Baseline ({baseline:.2f})")

    ax.set(
        xlim=(0.0, 1.0), ylim=(0.0, 1.05),
        xlabel="Recall", ylabel="Precision",
        title=title or "Precision-Recall Curve",
    )
    ax.legend(loc="upper right")

    if standalone:
        fig.tight_layout()
        save_figure(fig, save_path, tight=False)
    return fig


# Metric comparison bar chart
_SCALAR_METRICS = {
    "MCC", "BA", "Accuracy", "F1", "AUC",
    "Precision", "Recall", "Average Precision",
    "F1 weighted", "F1 macro", "Precision macro",
    "Recall macro", "Precision micro", "Recall micro",
}


def plot_metric_bar(
    metrics_dict: dict[str, float],
    *,
    title: Optional[str] = None,
    colour: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Horizontal bar chart of scalar evaluation metrics (seaborn).

    Only rate / score metrics in [0, 1] are plotted; list or dict values
    are silently skipped.

    Parameters
    ----------
    metrics_dict:
        ``{metric_name: value}`` mapping.
    title:
        Axes title.
    colour:
        Bar fill colour.  Defaults to the primary palette colour.
    ax:
        Pre-existing :class:`~matplotlib.axes.Axes`.
    save_path:
        Output file path.

    Returns
    -------
    Figure
    """
    set_theme()

    data = {
        k: float(v)
        for k, v in metrics_dict.items()
        if k in _SCALAR_METRICS and isinstance(v, (int, float))
    }

    if not data:
        raise ValueError("No plottable scalar metrics found in metrics_dict.")

    names = list(data.keys())
    values = [data[n] for n in names]
    colour = colour or PALETTE["primary"]

    standalone = ax is None
    height = max(3.5, 0.55 * len(names))
    if standalone:
        fig, ax = plt.subplots(figsize=(6.5, height))
    else:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

    sns.barplot(x=values, y=names, ax=ax, color=colour, orient="h")
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    ax.set(
        xlim=(0, 1.15),
        xlabel="Score",
        ylabel="",
        title=title or "Evaluation Metrics",
    )
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

    if standalone:
        fig.tight_layout()
        save_figure(fig, save_path, tight=False)
    return fig


# Decision-threshold sensitivity
def plot_threshold_metrics(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Precision, Recall, and F1 as a function of the decision threshold.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    y_score:
        Predicted probabilities for the positive class.
    title:
        Axes title.
    ax:
        Pre-existing :class:`~matplotlib.axes.Axes`.
    save_path:
        Output file path.

    Returns
    -------
    Figure
    """
    set_theme()

    y_true_arr = np.array(y_true)
    y_score_arr = np.array(y_score)
    thresholds = np.linspace(0.01, 0.99, 200)

    prec, rec, f1 = [], [], []
    for t in thresholds:
        y_hat = (y_score_arr >= t).astype(int)
        prec.append(skmetrics.precision_score(y_true_arr, y_hat, zero_division=0))
        rec.append(skmetrics.recall_score(y_true_arr, y_hat, zero_division=0))
        f1.append(skmetrics.f1_score(y_true_arr, y_hat, zero_division=0))

    best_t = thresholds[int(np.argmax(f1))]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

    sns.lineplot(x=thresholds, y=prec, ax=ax, color=PALETTE["primary"], label="Precision")
    sns.lineplot(x=thresholds, y=rec, ax=ax, color=PALETTE["secondary"], label="Recall")
    sns.lineplot(x=thresholds, y=f1, ax=ax, color=PALETTE["accent"], lw=2.5, label="F1")
    ax.axvline(best_t, color=PALETTE["neutral"], lw=1.2, linestyle="--",
               label=f"Best F1 @ {best_t:.2f}")

    ax.set(
        xlim=(0, 1), ylim=(0, 1.05),
        xlabel="Decision Threshold", ylabel="Score",
        title=title or "Threshold Sensitivity",
    )
    ax.legend()

    if standalone:
        fig.tight_layout()
        save_figure(fig, save_path, tight=False)
    return fig
