"""
High-level model evaluation reporting.

``Model_Evaluation`` aggregates predictions and produces DataFrames / dicts
suitable for display or downstream analysis.

Usage
-----
    from chemagent.ml.evaluation import Model_Evaluation

    ev = Model_Evaluation(
        labels=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        model_id="RFC",
        model_type="O00329_P42336",
        reg_class="classification",
    )
    print(ev.pred_performance_class)        # pandas DataFrame
    print(ev.prediction_performance_multiclass())
"""

from __future__ import annotations

import numpy as np

from .metrics import (
    classification_metrics,
    multiclass_metrics,
    regression_metrics,
)


class Model_Evaluation:
    """Collect predictions and compute evaluation reports.

    Parameters
    ----------
    labels:
        Ground-truth labels (classification) or values (regression).
    y_pred:
        Predicted class labels (classification tasks).
    y_proba:
        Predicted class probabilities, shape (n_samples, n_classes).
    y_pred_reg:
        Predicted continuous values (regression tasks).
    model_id:
        Algorithm name stored in reports (informational).
    model_type:
        Target / dataset identifier stored in reports (informational).
    reg_class:
        Task type: ``"classification"``, ``"classification-cw"``, or
        ``"regression"``.

    Attributes
    ----------
    pred_performance_class:
        ``pandas.DataFrame`` with classification metrics, or ``None`` for
        regression tasks.
    """

    def __init__(
        self,
        labels,
        y_pred=None,
        y_proba=None,
        y_pred_reg=None,
        model_id: str | None = None,
        model_type: str | None = None,
        reg_class: str | None = None,
    ) -> None:
        self.labels = np.array(labels)
        self.y_pred = np.array(y_pred) if y_pred is not None else None
        self.y_proba = np.array(y_proba) if y_proba is not None else None
        self.labels_reg = np.array(labels) if y_pred_reg is not None else None
        self.y_prediction_reg = np.array(y_pred_reg) if y_pred_reg is not None else None
        self.model_id = model_id
        self.model_type = model_type
        self.reg_class = reg_class

        is_classification = reg_class in ("classification", "classification-cw")
        self.pred_performance_class = (
            self._build_classification_df() if is_classification else None
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_classification_df(self):
        """Return a tidy ``pandas.DataFrame`` of classification metrics."""
        import pandas as pd

        assert self.y_pred is not None, "No classification predictions available"

        result = classification_metrics(
            labels=self.labels,
            pred=self.y_pred,
            y_proba=self.y_proba,
            model_id=self.model_id,
            model_type=self.model_type,
        )

        rows = [
            {"Metric": k, "Value": v}
            for k, v in result.items()
            if k not in ("Target ID", "Algorithm", "Dataset size")
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Public evaluation methods
    # ------------------------------------------------------------------

    def prediction_performance_classification(self):
        """Return a tidy ``pandas.DataFrame`` of classification metrics.

        Delegates to :func:`~chemagent.ml.metrics.classification_metrics`.

        Returns
        -------
        pandas.DataFrame
        """
        return self._build_classification_df()

    def prediction_performance_multiclass(self) -> dict:
        """Return a detailed multiclass evaluation report.

        Includes overall metrics, per-class metrics, and the confusion matrix.

        Returns
        -------
        dict
        """
        assert self.y_pred is not None, "No classification predictions available"

        return multiclass_metrics(
            labels=self.labels,
            pred=self.y_pred,
            model_id=self.model_id,
            model_type=self.model_type,
        )

    def prediction_performance_regression(self) -> dict:
        """Return regression metrics (MAE, MSE, RMSE, R², Pearson r).

        Returns
        -------
        dict
        """
        assert (
            self.labels_reg is not None and self.y_prediction_reg is not None
        ), "No regression predictions available"

        return regression_metrics(
            labels=self.labels_reg,
            pred=self.y_prediction_reg,
            model_id=self.model_id,
            model_type=self.model_type,
        )
