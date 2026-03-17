"""
chemagent.plots.regression — plots for continuous-target regression tasks.

All functions are general-purpose and work with any regression output.

Functions
---------
plot_actual_vs_predicted   Scatter of y_true vs y_pred with identity line.
plot_residuals             Residuals vs fitted values scatter.
plot_residual_histogram    Histogram + KDE of residuals.
plot_error_distribution    Histogram + KDE of absolute prediction errors.

Signature convention (all functions):
    plot_*(data, ..., title=None, ax=None, save_path=None) -> Figure

Example
-------
    from chemagent.plots.regression import plot_actual_vs_predicted
    from chemagent.plots.utils import set_theme

    set_theme()
    fig = plot_actual_vs_predicted(y_true, y_pred, save_path="results/pred.png")
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from sklearn import metrics as skmetrics

from .utils import set_theme, PALETTE, save_figure


# Actual vs predicted
def plot_actual_vs_predicted(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    model_id: Optional[str] = None,
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Scatter of ground-truth vs predicted values with an identity line.

    Annotates R² and MAE in the plot.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Model predictions.
    model_id:
        Model name shown in the legend.
    xlabel / ylabel:
        Axis labels (default "Actual" / "Predicted").
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

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    r2  = skmetrics.r2_score(yt, yp)
    mae = skmetrics.mean_absolute_error(yt, yp)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

    sns.scatterplot(x=yt, y=yp, ax=ax, alpha=0.6, s=30,
                    color=PALETTE["primary"], edgecolor="none",
                    label=model_id or "Predictions")

    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color=PALETTE["neutral"], lw=1.4, linestyle="--", label="Identity")

    ax.set(
        xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad),
        xlabel=xlabel, ylabel=ylabel,
        title=title or "Actual vs. Predicted",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.05, 0.95,
        f"R² = {r2:.3f}\nMAE = {mae:.3f}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
    )
    ax.legend(loc="lower right")

    if standalone:
        fig.tight_layout()
        save_figure(fig, save_path, tight=False)
    return fig


# Residuals vs fitted
def plot_residuals(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Residuals (y_true − y_pred) vs fitted (y_pred) scatter.

    A horizontal zero line is drawn.  Systematic curvature indicates
    model mis-specification.

    Parameters
    ----------
    y_true:
        Ground-truth values.
    y_pred:
        Model predictions.
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

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    residuals = yt - yp

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

    sns.scatterplot(x=yp, y=residuals, ax=ax, alpha=0.55, s=28,
                    color=PALETTE["accent"], edgecolor="none")
    ax.axhline(0, color=PALETTE["neutral"], lw=1.4, linestyle="--")
    sns.regplot(x=yp, y=residuals, ax=ax, scatter=False,
                line_kws={"color": PALETTE["secondary"], "lw": 1.4, "linestyle": ":"})

    ax.set(
        xlabel="Fitted Values (ŷ)",
        ylabel="Residuals (y − ŷ)",
        title=title or "Residuals vs. Fitted",
    )

    if standalone:
        fig.tight_layout()
        save_figure(fig, save_path, tight=False)
    return fig


# Residual histogram
def plot_residual_histogram(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    bins: int | str = "auto",
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Histogram + KDE of prediction residuals (y_true − y_pred).

    Parameters
    ----------
    y_true:
        Ground-truth values.
    y_pred:
        Model predictions.
    bins:
        Number of histogram bins or a string strategy (``"auto"``, etc.).
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

    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

    sns.histplot(residuals, bins=bins, kde=True, ax=ax,
                 color=PALETTE["primary"], edgecolor="white", alpha=0.75,
                 line_kws={"lw": 2, "color": PALETTE["secondary"]})
    ax.axvline(0, color=PALETTE["neutral"], lw=1.4, linestyle="--", label="Zero")

    ax.set(
        xlabel="Residual (y − ŷ)",
        ylabel="Count",
        title=title or "Residual Distribution",
    )
    ax.legend()

    if standalone:
        fig.tight_layout()
        save_figure(fig, save_path, tight=False)
    return fig


# Absolute-error distribution
def plot_error_distribution(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Histogram + KDE of absolute prediction errors |y_true − y_pred|.

    Parameters
    ----------
    y_true:
        Ground-truth values.
    y_pred:
        Model predictions.
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

    abs_errors = np.abs(
        np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    )
    mae = float(abs_errors.mean())

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

    sns.histplot(abs_errors, kde=True, ax=ax,
                 color=PALETTE["accent"], edgecolor="white", alpha=0.75,
                 line_kws={"lw": 2, "color": PALETTE["secondary"]})
    ax.axvline(mae, color=PALETTE["secondary"], lw=2, linestyle="--",
               label=f"MAE = {mae:.3f}")
    sns.rugplot(abs_errors, ax=ax, color=PALETTE["neutral"], alpha=0.25, height=0.04)

    ax.set(
        xlabel="|y − ŷ|",
        ylabel="Count",
        title=title or "Absolute Error Distribution",
    )
    ax.legend()

    if standalone:
        fig.tight_layout()
        save_figure(fig, save_path, tight=False)
    return fig
