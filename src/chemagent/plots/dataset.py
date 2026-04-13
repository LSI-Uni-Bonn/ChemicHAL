"""
chemagent.plots.dataset — plots for dataset exploration and split diagnostics.

All functions are general-purpose and are not tied to any specific domain or
column naming convention.

Functions
---------
plot_class_distribution     Bar chart of class counts with percentage labels.
plot_split_statistics       Stacked bar showing train / val / test proportions.
plot_column_distribution    Histogram + KDE of any numeric DataFrame column.
plot_class_balance_splits   Grouped bars — class share across data splits.
plot_dataset_comparison     Compound / sample count comparison across datasets.

Signature convention:
    plot_*(data, ..., title=None, ax=None, save_path=None) -> Figure

Example
-------
    import pandas as pd
    from chemagent.plots.dataset import plot_class_distribution, plot_column_distribution
    from chemagent.plots.utils import set_theme

    set_theme()
    df = pd.read_csv("data/my_dataset.csv")
    fig = plot_class_distribution(df["label"])
    fig = plot_column_distribution(df, column="activity", save_path="results/dist.png")
"""

from __future__ import annotations

from typing import Optional, Sequence, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

from .utils import set_theme, PALETTE, SNS_PALETTE, save_figure


# Class distribution (single partition)
def plot_class_distribution(
    labels: Sequence,
    *,
    class_names: Optional[Sequence[str]] = None,
    palette: Optional[list[str]] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Bar chart of class counts with percentage annotations (seaborn).

    Args:
    labels:
        Array-like of class labels (integers or strings).
    class_names:
        Display names for each class.  Defaults to the unique sorted values.
    palette:
        List of hex colours, one per class.  Defaults to ``SNS_PALETTE``.
    title:
        Axes title.
    ax:
        Pre-existing :class:`~matplotlib.axes.Axes`.
    save_path:
        Output file path.

    Returns:
    Figure
    """
    set_theme()

    lbls = np.array(labels)
    classes = sorted(np.unique(lbls).tolist(), key=str)
    counts = [int(np.sum(lbls == c)) for c in classes]
    if class_names is not None and len(class_names) != len(classes):
        raise ValueError(
            f"class_names has {len(class_names)} entries but data has "
            f"{len(classes)} unique classes {classes}. "
            "Pass class_names=None to auto-generate labels."
        )
    names = list(class_names) if class_names else [str(c) for c in classes]
    colours = (palette or SNS_PALETTE)[: len(classes)]
    total = sum(counts)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(4, len(classes) * 1.2), 4))
    else:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

    ax.bar(names, counts, color=colours, edgecolor="white", width=0.55)

    for i, (name, count) in enumerate(zip(names, counts)):
        pct = count / total * 100 if total else 0
        ax.text(i, count + total * 0.01, f"{count}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9)

    ax.set(
        ylabel="Count",
        xlabel="",
        title=title or "Class Distribution",
        ylim=(0, max(counts) * 1.22),
    )

    if standalone:
        fig.tight_layout()
        save_figure(fig, save_path, tight=False)
    return fig


# Split proportions (train / val / test stacked bar)
def plot_split_statistics(
    split_stats: Dict[str, Dict[str, Any]],
    *,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Horizontal stacked bar showing partition proportions (seaborn style).

    Args:
    split_stats:
        Dict returned by
        :func:`chemagent.splitting.statistics.get_split_statistics`.
        Expected keys per partition: ``"count"`` and ``"percentage"``.
    title:
        Axes title.
    ax:
        Pre-existing :class:`~matplotlib.axes.Axes`.
    save_path:
        Output file path.

    Returns:
    Figure
    """
    set_theme(style="white")  # type: ignore[arg-type]

    partition_order = [k for k in ("train", "val", "test") if k in split_stats]
    colours: dict[str, str] = dict(zip(("train", "val", "test"),
                                       (PALETTE["primary"], PALETTE["accent"], PALETTE["secondary"])))

    total = split_stats.get("total", {}).get("count", None) or sum(
        v["count"] for k, v in split_stats.items() if k != "total"
    )

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 2.5))
    else:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

    left = 0.0
    for part in partition_order:
        count = split_stats[part]["count"]
        pct = count / total * 100 if total else 0
        colour = colours.get(part, PALETTE["light"])
        ax.barh(["Dataset"], [count], left=left, color=colour, height=0.5,
                label=f"{part.capitalize()} ({count}, {pct:.1f}%)")
        ax.text(left + count / 2, 0, f"{part}\n{count}",
                ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
        left += count

    ax.set(xlabel="Samples", title=title or "Data Split")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.7), ncol=3)
    ax.yaxis.set_visible(False)
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)

    if standalone:
        fig.tight_layout()
        save_figure(fig, save_path, tight=False)
    return fig


# Generic numeric column distribution
def plot_column_distribution(
    df,
    *,
    column: str,
    hue: Optional[str] = None,
    bins: int | str = "auto",
    kde: bool = True,
    reference_line: Optional[float] = None,
    reference_label: Optional[str] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Histogram + optional KDE of any numeric DataFrame column (seaborn).

    Args:
    df:
        :class:`pandas.DataFrame` containing *column*.
    column:
        Name of the column to plot.
    hue:
        Optional column used to colour sub-distributions.
    bins:
        Number of histogram bins or ``"auto"``.
    kde:
        Overlay a KDE curve when ``True``.
    reference_line:
        If given, draw a vertical dashed line at this x-value.
    reference_label:
        Legend label for *reference_line*.
    title:
        Axes title.
    ax:
        Pre-existing :class:`~matplotlib.axes.Axes`.
    save_path:
        Output file path.

    Returns:
    Figure
    """
    set_theme()

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

    palette = SNS_PALETTE if hue else None
    sns.histplot(data=df, x=column, hue=hue, bins=bins, kde=kde,
                 palette=palette, color=PALETTE["primary"],
                 edgecolor="white", alpha=0.75, ax=ax)

    if reference_line is not None:
        ax.axvline(reference_line, color=PALETTE["neutral"], lw=1.4,
                   linestyle="--", label=reference_label or f"x = {reference_line}")
        ax.legend()

    values = df[column].dropna()
    ax.text(
        0.97, 0.95,
        f"n = {len(values)}\nμ = {values.mean():.3f}\nσ = {values.std():.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
    )

    ax.set(
        xlabel=column, ylabel="Count",
        title=title or f"Distribution of \'{column}\'",
    )

    if standalone:
        fig.tight_layout()
        save_figure(fig, save_path, tight=False)
    return fig


# Class balance across splits (grouped bars)
def plot_class_balance_splits(
    class_dist: Dict[str, Dict[str, int]],
    *,
    class_names: Optional[Sequence[str]] = None,
    palette: Optional[list[str]] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Grouped bar chart — class share (%) within each data split (seaborn).

    Args:
    class_dist:
        Dict returned by
        :func:`chemagent.splitting.statistics.class_distribution`.
        ``{partition: {class_label: count, ...}, ...}``
    class_names:
        Display names for each class.
    palette:
        List of hex colours, one per class.
    title:
        Axes title.
    ax:
        Pre-existing :class:`~matplotlib.axes.Axes`.
    save_path:
        Output file path.

    Returns:
    Figure
    """
    import pandas as pd

    set_theme()

    partitions = [k for k in ("train", "val", "test") if k in class_dist]
    all_classes = sorted(
        {cls for v in class_dist.values() for cls in v.keys()}, key=str
    )
    names = list(class_names) if class_names else [f"Class {c}" for c in all_classes]
    colours = (palette or SNS_PALETTE)[: len(all_classes)]

    rows = []
    for part in partitions:
        counts = class_dist[part]
        total = sum(counts.values())
        for cls, name in zip(all_classes, names):
            pct = counts.get(str(cls), 0) / total * 100 if total else 0.0
            rows.append({"Split": part.capitalize(), "Class": name, "Share (%)": pct})

    plot_df = pd.DataFrame(rows)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

    sns.barplot(data=plot_df, x="Split", y="Share (%)", hue="Class",
                palette=colours, ax=ax, edgecolor="white")
    ax.set(
        ylim=(0, 110),
        title=title or "Class Balance per Split",
    )

    if standalone:
        fig.tight_layout()
        save_figure(fig, save_path, tight=False)
    return fig


# Generic dataset comparison bar chart
def plot_dataset_comparison(
    counts: Dict[str, int],
    *,
    xlabel: str = "Dataset",
    ylabel: str = "Sample count",
    palette: Optional[list[str]] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Bar chart comparing sample counts across multiple datasets or groups.

    Args:
    counts:
        ``{group_label: count}`` mapping, e.g.
        ``{"Dataset A": 1200, "Dataset B": 980}``.
    xlabel / ylabel:
        Axis labels.
    palette:
        List of hex colours.  Defaults to ``SNS_PALETTE``.
    title:
        Axes title.
    ax:
        Pre-existing :class:`~matplotlib.axes.Axes`.
    save_path:
        Output file path.

    Returns:
    Figure
    """
    set_theme()

    groups = list(counts.keys())
    values = [counts[g] for g in groups]
    colours = (palette or SNS_PALETTE)[: len(groups)]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(max(4, len(groups) * 1.8), 4))
    else:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

    ax.bar(groups, values, color=colours, edgecolor="white", width=0.55)
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, str(v),
                ha="center", va="bottom", fontsize=9)

    ax.set(
        ylabel=ylabel, xlabel=xlabel,
        title=title or "Dataset Comparison",
        ylim=(0, max(values) * 1.18),
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    if standalone:
        fig.tight_layout()
        save_figure(fig, save_path, tight=False)
    return fig
