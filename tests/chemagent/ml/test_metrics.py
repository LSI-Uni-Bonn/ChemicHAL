"""
Tests for chemagent.ml.metrics.

Covers:
- confusion_components: correct FP/FN/TP/TN decomposition
- classification_metrics: binary (with/without proba) and multiclass
- multiclass_metrics: structure, per-class keys, confusion matrix shape
- regression_metrics: expected keys, sign constraints, perfect-prediction edge case
"""

import numpy as np
import pytest

from chemagent.ml.metrics import (
    classification_metrics,
    confusion_components,
    multiclass_metrics,
    regression_metrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(2)


@pytest.fixture(scope="module")
def perfect_binary():
    """Labels and perfectly matching predictions (binary)."""
    y = np.array([0, 0, 1, 1, 0, 1])
    return y, y.copy()


@pytest.fixture(scope="module")
def binary_data():
    """100-sample binary classification data with ~80 % accuracy."""
    labels = RNG.integers(0, 2, size=100)
    preds = labels.copy()
    flip = RNG.choice(100, size=20, replace=False)
    preds[flip] = 1 - preds[flip]
    raw = np.where(labels == 1,
                   RNG.uniform(0.55, 0.95, size=100),
                   RNG.uniform(0.05, 0.45, size=100))
    proba = np.column_stack([1 - raw, raw])
    return labels, preds, proba


@pytest.fixture(scope="module")
def multiclass_data():
    """100-sample 3-class classification data."""
    labels = RNG.integers(0, 3, size=100)
    preds = labels.copy()
    flip = RNG.choice(100, size=15, replace=False)
    preds[flip] = (preds[flip] + 1) % 3
    return labels, preds


@pytest.fixture(scope="module")
def regression_data():
    """Near-linear regression data."""
    X = RNG.standard_normal(100)
    labels = X * 2.0 + 1.0
    preds = labels + RNG.standard_normal(100) * 0.2
    return labels, preds


# ---------------------------------------------------------------------------
# confusion_components
# ---------------------------------------------------------------------------


def test_confusion_components_returns_four_arrays(binary_data):
    """Must return exactly four arrays."""
    labels, preds, _ = binary_data
    result = confusion_components(labels, preds)
    assert len(result) == 4
    FP, FN, TP, TN = result
    for arr in (FP, FN, TP, TN):
        assert isinstance(arr, np.ndarray)


def test_confusion_components_perfect_prediction(perfect_binary):
    """Perfect predictions must yield FP = FN = 0 for all classes."""
    labels, preds = perfect_binary
    FP, FN, TP, TN = confusion_components(labels, preds)
    np.testing.assert_array_equal(FP, 0)
    np.testing.assert_array_equal(FN, 0)


def test_confusion_components_tp_equals_class_count(perfect_binary):
    """With perfect predictions, TP[c] must equal the count of class c."""
    labels, preds = perfect_binary
    FP, FN, TP, TN = confusion_components(labels, preds)
    for c in np.unique(labels):
        assert TP[c] == np.sum(labels == c)


def test_confusion_components_fp_fn_tp_tn_sum(binary_data):
    """FP + FN + TP + TN per class must equal total number of samples."""
    labels, preds, _ = binary_data
    FP, FN, TP, TN = confusion_components(labels, preds)
    n = len(labels)
    np.testing.assert_array_equal(FP + FN + TP + TN, n)


def test_confusion_components_non_negative(binary_data):
    """All components must be non-negative."""
    labels, preds, _ = binary_data
    for arr in confusion_components(labels, preds):
        assert np.all(arr >= 0)


# ---------------------------------------------------------------------------
# classification_metrics — binary
# ---------------------------------------------------------------------------


def test_classification_metrics_binary_keys(binary_data):
    """Binary result must contain all core and binary-specific keys."""
    labels, preds, proba = binary_data
    result = classification_metrics(labels, preds, y_proba=proba)
    for key in ("MCC", "BA", "Accuracy", "F1", "AUC", "Precision", "Recall",
                "Average Precision", "FP", "FN", "TP", "TN"):
        assert key in result, f"Key '{key}' missing"


def test_classification_metrics_no_multiclass_keys_in_binary(binary_data):
    """Binary result must not contain multiclass-only keys."""
    labels, preds, _ = binary_data
    result = classification_metrics(labels, preds)
    assert "F1 weighted" not in result
    assert "F1 macro" not in result


def test_classification_metrics_accuracy_range(binary_data):
    """Accuracy must be in [0, 1]."""
    labels, preds, _ = binary_data
    result = classification_metrics(labels, preds)
    assert 0.0 <= result["Accuracy"] <= 1.0


def test_classification_metrics_mcc_range(binary_data):
    """MCC must be in [-1, 1]."""
    labels, preds, _ = binary_data
    result = classification_metrics(labels, preds)
    assert -1.0 <= result["MCC"] <= 1.0


def test_classification_metrics_auc_range(binary_data):
    """AUC with soft probabilities must be in [0, 1]."""
    labels, preds, proba = binary_data
    result = classification_metrics(labels, preds, y_proba=proba)
    assert 0.0 <= result["AUC"] <= 1.0


def test_classification_metrics_proba_stored(binary_data):
    """When y_proba is provided, 'Probability' must appear in the result."""
    labels, preds, proba = binary_data
    result = classification_metrics(labels, preds, y_proba=proba)
    assert "Probability" in result


def test_classification_metrics_informational_fields(binary_data):
    """model_id and model_type must be stored under 'Algorithm' and 'Target ID'."""
    labels, preds, _ = binary_data
    result = classification_metrics(
        labels, preds, model_id="RFC", model_type="O00329_P42336"
    )
    assert result["Algorithm"] == "RFC"
    assert result["Target ID"] == "O00329_P42336"


def test_classification_metrics_dataset_size(binary_data):
    """'Dataset size' must equal the number of samples."""
    labels, preds, _ = binary_data
    result = classification_metrics(labels, preds)
    assert result["Dataset size"] == len(labels)


# ---------------------------------------------------------------------------
# classification_metrics — multiclass
# ---------------------------------------------------------------------------


def test_classification_metrics_multiclass_keys(multiclass_data):
    """Multiclass result must contain weighted/macro F1, not binary-only keys."""
    labels, preds = multiclass_data
    result = classification_metrics(labels, preds)
    for key in ("F1 weighted", "F1 macro", "Precision macro", "Recall macro",
                "Precision micro", "Recall micro"):
        assert key in result, f"Key '{key}' missing"
    assert "AUC" not in result
    assert "F1" not in result


# ---------------------------------------------------------------------------
# multiclass_metrics
# ---------------------------------------------------------------------------


def test_multiclass_metrics_top_level_keys(multiclass_data):
    """Top-level keys must be present."""
    labels, preds = multiclass_data
    result = multiclass_metrics(labels, preds)
    for key in ("overall_metrics", "per_class_metrics",
                "confusion_matrix", "class_labels"):
        assert key in result


def test_multiclass_metrics_overall_range(multiclass_data):
    """Overall accuracy and BA must be in [0, 1]."""
    labels, preds = multiclass_data
    overall = multiclass_metrics(labels, preds)["overall_metrics"]
    assert 0.0 <= overall["Accuracy"] <= 1.0
    assert 0.0 <= overall["BA"] <= 1.0


def test_multiclass_metrics_confusion_matrix_shape(multiclass_data):
    """Confusion matrix must be n_classes × n_classes."""
    labels, preds = multiclass_data
    n_classes = len(np.unique(labels))
    cm = multiclass_metrics(labels, preds)["confusion_matrix"]
    assert len(cm) == n_classes
    assert all(len(row) == n_classes for row in cm)


def test_multiclass_metrics_per_class_count(multiclass_data):
    """per_class_metrics must have one entry per class."""
    labels, preds = multiclass_data
    n_classes = len(np.unique(labels))
    per_class = multiclass_metrics(labels, preds)["per_class_metrics"]
    assert len(per_class) == n_classes


def test_multiclass_metrics_per_class_sub_keys(multiclass_data):
    """Each per-class entry must have Precision, Recall, F1, Support."""
    labels, preds = multiclass_data
    per_class = multiclass_metrics(labels, preds)["per_class_metrics"]
    for cls_key, cls_dict in per_class.items():
        for sub in ("Precision", "Recall", "F1", "Support"):
            assert sub in cls_dict, f"'{sub}' missing in {cls_key}"


def test_multiclass_metrics_informational(multiclass_data):
    """model_id and model_type must be stored under 'algorithm' and 'target'."""
    labels, preds = multiclass_data
    result = multiclass_metrics(labels, preds, model_id="RFC", model_type="TEST")
    assert result["algorithm"] == "RFC"
    assert result["target"] == "TEST"


# ---------------------------------------------------------------------------
# regression_metrics
# ---------------------------------------------------------------------------


def test_regression_metrics_keys(regression_data):
    """Result must contain MAE, MSE, RMSE, R2, r, Dataset size."""
    labels, preds = regression_data
    result = regression_metrics(labels, preds)
    for key in ("MAE", "MSE", "RMSE", "R2", "r", "Dataset size"):
        assert key in result, f"Key '{key}' missing"


def test_regression_metrics_non_negative_errors(regression_data):
    """MAE, MSE, and RMSE must all be non-negative."""
    labels, preds = regression_data
    result = regression_metrics(labels, preds)
    assert result["MAE"] >= 0.0
    assert result["MSE"] >= 0.0
    assert result["RMSE"] >= 0.0


def test_regression_metrics_rmse_geq_mae(regression_data):
    """RMSE >= MAE must hold by definition."""
    labels, preds = regression_data
    result = regression_metrics(labels, preds)
    assert result["RMSE"] >= result["MAE"] - 1e-9


def test_regression_metrics_r2_positive(regression_data):
    """R² should be positive for a good predictor."""
    labels, preds = regression_data
    result = regression_metrics(labels, preds)
    assert result["R2"] > 0.0


def test_regression_metrics_pearson_range(regression_data):
    """Pearson r must be in [-1, 1]."""
    labels, preds = regression_data
    result = regression_metrics(labels, preds)
    assert -1.0 <= result["r"] <= 1.0


def test_regression_metrics_perfect_prediction():
    """Perfect predictions must give MAE = 0, RMSE = 0, R2 = 1, r = 1."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = regression_metrics(y, y.copy())
    assert result["MAE"] == pytest.approx(0.0)
    assert result["RMSE"] == pytest.approx(0.0)
    assert result["R2"] == pytest.approx(1.0)
    assert result["r"] == pytest.approx(1.0)


def test_regression_metrics_informational(regression_data):
    """model_id and model_type must be stored under 'Algorithm' and 'Target ID'."""
    labels, preds = regression_data
    result = regression_metrics(labels, preds, model_id="RFR", model_type="TEST")
    assert result["Algorithm"] == "RFR"
    assert result["Target ID"] == "TEST"
