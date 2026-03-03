"""
chemagent.plots — publication-ready plots for ML analysis.

All functions are general-purpose and work with any classification or
regression problem.

Sub-modules
-----------
classification
    Confusion matrix, ROC curve, Precision-Recall curve, metric bar chart,
    feature importance, and threshold-sensitivity plot.
regression
    Actual-vs-predicted scatter, residual plot, residual histogram, and
    absolute-error distribution.
dataset
    Class distribution, split statistics, numeric column distribution,
    class-balance per split, and dataset comparison bar chart.
utils
    Shared colour palettes, ``set_theme()``, and ``save_figure()`` helper.

Quick start
-----------
    from chemagent.plots import (
        set_theme,
        plot_confusion_matrix,
        plot_roc_curve,
        plot_pr_curve,
        plot_metric_bar,
        plot_feature_importance,
        plot_threshold_metrics,
        plot_actual_vs_predicted,
        plot_residuals,
        plot_residual_histogram,
        plot_error_distribution,
        plot_class_distribution,
        plot_split_statistics,
        plot_column_distribution,
        plot_class_balance_splits,
        plot_dataset_comparison,
    )

    set_theme()
    fig = plot_confusion_matrix(y_true, y_pred, class_names=["neg", "pos"])
"""

from .utils import set_theme, save_figure, PALETTE, SNS_PALETTE

from .classification import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_metric_bar,
    plot_feature_importance,
    plot_threshold_metrics,
)

from .regression import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_residual_histogram,
    plot_error_distribution,
)

from .dataset import (
    plot_class_distribution,
    plot_split_statistics,
    plot_column_distribution,
    plot_class_balance_splits,
    plot_dataset_comparison,
)

__all__ = [
    # utils
    "set_theme",
    "save_figure",
    "PALETTE",
    "SNS_PALETTE",
    # classification
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_metric_bar",
    "plot_feature_importance",
    "plot_threshold_metrics",
    # regression
    "plot_actual_vs_predicted",
    "plot_residuals",
    "plot_residual_histogram",
    "plot_error_distribution",
    # dataset
    "plot_class_distribution",
    "plot_split_statistics",
    "plot_column_distribution",
    "plot_class_balance_splits",
    "plot_dataset_comparison",
]
