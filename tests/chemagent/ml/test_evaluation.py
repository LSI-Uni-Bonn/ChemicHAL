"""
Tests for chemagent.ml.evaluation.Model_Evaluation.

Covers:
- Classification task: pred_performance_class is a DataFrame
- Regression task: pred_performance_class is None
- All three public methods return the expected types and keys
- Metric values are within sensible ranges
- AssertionError is raised when predictions are missing
"""

import numpy as np
import pandas as pd
import pytest

from chemagent.ml.evaluation import Model_Evaluation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(1)


@pytest.fixture(scope="module")
def binary_labels():
    """100 balanced binary ground-truth labels."""
    return RNG.integers(0, 2, size=100).astype(int)


@pytest.fixture(scope="module")
def binary_preds(binary_labels):
    """Noisy binary predictions (~80 % accuracy)."""
    preds = binary_labels.copy()
    flip = RNG.choice(len(preds), size=20, replace=False)
    preds[flip] = 1 - preds[flip]
    return preds


@pytest.fixture(scope="module")
def binary_proba(binary_labels):
    """Soft probabilities, shape (100, 2)."""
    raw = np.where(binary_labels == 1,
                   RNG.uniform(0.55, 0.95, size=100),
                   RNG.uniform(0.05, 0.45, size=100))
    raw = np.clip(raw, 0.01, 0.99)
    return np.column_stack([1 - raw, raw])


@pytest.fixture(scope="module")
def regression_labels():
    """100-sample regression targets."""
    X = RNG.standard_normal(100)
    return X * 3.0 + RNG.standard_normal(100) * 0.5


@pytest.fixture(scope="module")
def regression_preds(regression_labels):
    """Noisy regression predictions."""
    return regression_labels + RNG.standard_normal(len(regression_labels)) * 0.3


@pytest.fixture
def clf_eval(binary_labels, binary_preds, binary_proba):
    """Model_Evaluation instance for a binary classification task."""
    return Model_Evaluation(
        labels=binary_labels,
        y_pred=binary_preds,
        y_proba=binary_proba,
        model_id="RFC",
        model_type="O00329_P42336",
        reg_class="classification",
    )


@pytest.fixture
def reg_eval(regression_labels, regression_preds):
    """Model_Evaluation instance for a regression task."""
    return Model_Evaluation(
        labels=regression_labels,
        y_pred_reg=regression_preds,
        model_id="RFR",
        model_type="O00329_P42336",
        reg_class="regression",
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_clf_pred_performance_class_is_dataframe(clf_eval):
    """pred_performance_class must be a DataFrame for classification tasks."""
    assert isinstance(clf_eval.pred_performance_class, pd.DataFrame)


def test_reg_pred_performance_class_is_none(reg_eval):
    """pred_performance_class must be None for regression tasks."""
    assert reg_eval.pred_performance_class is None


def test_clf_dataframe_has_metric_and_value_columns(clf_eval):
    """DataFrame must have 'Metric' and 'Value' columns."""
    df = clf_eval.pred_performance_class
    assert "Metric" in df.columns
    assert "Value" in df.columns


def test_clf_dataframe_contains_expected_metrics(clf_eval):
    """DataFrame rows must include core binary classification metrics."""
    metrics_in_df = set(clf_eval.pred_performance_class["Metric"])
    for expected in ("MCC", "BA", "Accuracy", "F1", "AUC", "Precision", "Recall"):
        assert expected in metrics_in_df, f"'{expected}' missing from pred_performance_class"


def test_labels_and_preds_stored_as_arrays(binary_labels, binary_preds, clf_eval):
    """labels and y_pred must be stored as numpy arrays."""
    assert isinstance(clf_eval.labels, np.ndarray)
    assert isinstance(clf_eval.y_pred, np.ndarray)
    assert len(clf_eval.labels) == len(binary_labels)


# ---------------------------------------------------------------------------
# prediction_performance_classification
# ---------------------------------------------------------------------------


def test_classification_method_returns_dataframe(clf_eval):
    """prediction_performance_classification() must return a DataFrame."""
    result = clf_eval.prediction_performance_classification()
    assert isinstance(result, pd.DataFrame)


def test_classification_method_accuracy_in_range(clf_eval):
    """Accuracy metric value must be in [0, 1]."""
    df = clf_eval.prediction_performance_classification()
    accuracy = df.loc[df["Metric"] == "Accuracy", "Value"].iloc[0]
    assert 0.0 <= accuracy <= 1.0


def test_classification_method_mcc_in_range(clf_eval):
    """MCC value must be in [-1, 1]."""
    df = clf_eval.prediction_performance_classification()
    mcc = df.loc[df["Metric"] == "MCC", "Value"].iloc[0]
    assert -1.0 <= mcc <= 1.0


def test_classification_missing_preds_raises():
    """Constructing without y_pred for a classification task must raise AssertionError."""
    with pytest.raises(AssertionError):
        Model_Evaluation(
            labels=np.array([0, 1, 0, 1]),
            reg_class="classification",
        )


# ---------------------------------------------------------------------------
# prediction_performance_multiclass
# ---------------------------------------------------------------------------


def test_multiclass_returns_dict(clf_eval):
    """prediction_performance_multiclass() must return a dict."""
    result = clf_eval.prediction_performance_multiclass()
    assert isinstance(result, dict)


def test_multiclass_has_expected_keys(clf_eval):
    """Result dict must contain all standard top-level keys."""
    result = clf_eval.prediction_performance_multiclass()
    for key in ("overall_metrics", "per_class_metrics", "confusion_matrix", "class_labels"):
        assert key in result, f"Key '{key}' missing from multiclass result"


def test_multiclass_overall_metrics_range(clf_eval):
    """Overall accuracy and BA must be in [0, 1]."""
    overall = clf_eval.prediction_performance_multiclass()["overall_metrics"]
    assert 0.0 <= overall["Accuracy"] <= 1.0
    assert 0.0 <= overall["BA"] <= 1.0


def test_multiclass_confusion_matrix_shape(clf_eval, binary_labels):
    """Confusion matrix must be square with size == number of classes."""
    cm = clf_eval.prediction_performance_multiclass()["confusion_matrix"]
    n_classes = len(np.unique(binary_labels))
    assert len(cm) == n_classes
    assert all(len(row) == n_classes for row in cm)


def test_multiclass_per_class_keys(clf_eval, binary_labels):
    """per_class_metrics must have one entry per class."""
    per_class = clf_eval.prediction_performance_multiclass()["per_class_metrics"]
    n_classes = len(np.unique(binary_labels))
    assert len(per_class) == n_classes


# ---------------------------------------------------------------------------
# prediction_performance_regression
# ---------------------------------------------------------------------------


def test_regression_returns_dict(reg_eval):
    """prediction_performance_regression() must return a dict."""
    result = reg_eval.prediction_performance_regression()
    assert isinstance(result, dict)


def test_regression_has_expected_keys(reg_eval):
    """Result dict must contain MAE, MSE, RMSE, R2, and Pearson r."""
    result = reg_eval.prediction_performance_regression()
    for key in ("MAE", "MSE", "RMSE", "R2", "r"):
        assert key in result, f"Key '{key}' missing from regression result"


def test_regression_r2_positive(reg_eval):
    """R² should be positive for a near-perfect prediction."""
    result = reg_eval.prediction_performance_regression()
    assert result["R2"] > 0.0


def test_regression_mae_non_negative(reg_eval):
    """MAE must be non-negative."""
    result = reg_eval.prediction_performance_regression()
    assert result["MAE"] >= 0.0


def test_regression_missing_preds_raises():
    """Calling without y_pred_reg must raise AssertionError."""
    ev = Model_Evaluation(
        labels=np.array([1.0, 2.0, 3.0]),
        reg_class="regression",
    )
    with pytest.raises(AssertionError):
        ev.prediction_performance_regression()
