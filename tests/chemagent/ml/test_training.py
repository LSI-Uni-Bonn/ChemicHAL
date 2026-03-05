"""
Tests for chemagent.ml.training.MLModel.

Covers:
- Fitted model has a predict() method and returns correct-shape predictions
- With parameters="grid", cv_results is set and best_params is a dict
- With parameters="none", cv_results is None and best_params is empty
- Classification predictions are discrete labels in the training label set
- Regression predictions are continuous floats
- The stored best_params come from the allowed grid values
"""

import types

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV

from chemagent.ml import models as _models
from chemagent.ml.training import MLModel


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(3)


def _make_data(n_samples=80, n_features=8, task="classification"):
    """Return a SimpleNamespace mimicking the dataset object MLModel expects."""
    X = RNG.standard_normal((n_samples, n_features))
    if task == "regression":
        y = X[:, 0] * 2.0 + RNG.standard_normal(n_samples) * 0.3
        return types.SimpleNamespace(features=X, labels=y, class_labels=y)
    else:
        y = RNG.integers(0, 2, size=n_samples)
        return types.SimpleNamespace(features=X, labels=y, class_labels=y)


@pytest.fixture(scope="module")
def clf_data():
    """Binary classification dataset."""
    return _make_data(task="classification")


@pytest.fixture(scope="module")
def reg_data():
    """Regression dataset."""
    return _make_data(task="regression")


@pytest.fixture(autouse=True)
def small_param_grids():
    """Replace PARAM_GRIDS with tiny grids so grid-search tests are fast.

    Restores the original grids after each test.
    """
    snapshot = {k: dict(v) for k, v in _models.PARAM_GRIDS.items()}
    _models.PARAM_GRIDS["RFC"] = {"n_estimators": [10, 20]}
    _models.PARAM_GRIDS["RFR"] = {"n_estimators": [10, 20]}
    _models.PARAM_GRIDS["SVC"] = {"C": [0.1, 1.0]}
    yield
    _models.PARAM_GRIDS.clear()
    _models.PARAM_GRIDS.update(snapshot)


# ---------------------------------------------------------------------------
# Basic fitting
# ---------------------------------------------------------------------------


def test_clf_model_is_fitted(clf_data):
    """After training, model must have a predict() method."""
    ml = MLModel(clf_data, ml_algorithm="RFC", reg_class="classification",
                 parameters="none", cv_fold=3)
    assert hasattr(ml.model, "predict")


def test_reg_model_is_fitted(reg_data):
    """Regression model must have a predict() method."""
    ml = MLModel(reg_data, ml_algorithm="RFR", reg_class="regression",
                 parameters="none", cv_fold=3)
    assert hasattr(ml.model, "predict")


def test_clf_predictions_correct_shape(clf_data):
    """predict() must return an array of length n_samples."""
    ml = MLModel(clf_data, ml_algorithm="RFC", reg_class="classification",
                 parameters="none", cv_fold=3)
    preds = ml.model.predict(clf_data.features)
    assert preds.shape == (len(clf_data.labels),)


def test_clf_predictions_are_valid_labels(clf_data):
    """Classification predictions must only contain values present in the training labels."""
    ml = MLModel(clf_data, ml_algorithm="RFC", reg_class="classification",
                 parameters="none", cv_fold=3)
    preds = ml.model.predict(clf_data.features)
    allowed = set(np.unique(clf_data.labels))
    assert set(np.unique(preds)).issubset(allowed)


def test_reg_predictions_are_floats(reg_data):
    """Regression predictions must be floating-point values."""
    ml = MLModel(reg_data, ml_algorithm="RFR", reg_class="regression",
                 parameters="none", cv_fold=3)
    preds = ml.model.predict(reg_data.features)
    assert np.issubdtype(preds.dtype, np.floating)


# ---------------------------------------------------------------------------
# parameters="none" — skip grid search
# ---------------------------------------------------------------------------


def test_no_grid_search_cv_results_is_none(clf_data):
    """With parameters='none', cv_results must be None."""
    ml = MLModel(clf_data, ml_algorithm="RFC", reg_class="classification",
                 parameters="none", cv_fold=3)
    assert ml.cv_results is None


def test_no_grid_search_best_params_empty(clf_data):
    """With parameters='none', best_params must be an empty dict."""
    ml = MLModel(clf_data, ml_algorithm="RFC", reg_class="classification",
                 parameters="none", cv_fold=3)
    assert ml.best_params == {}


# ---------------------------------------------------------------------------
# parameters="grid" — run GridSearchCV
# ---------------------------------------------------------------------------


def test_grid_search_cv_results_is_gridsearchcv(clf_data):
    """With parameters='grid', cv_results must be a fitted GridSearchCV."""
    ml = MLModel(clf_data, ml_algorithm="RFC", reg_class="classification",
                 parameters="grid", cv_fold=3)
    assert isinstance(ml.cv_results, GridSearchCV)


def test_grid_search_best_params_is_dict(clf_data):
    """With parameters='grid', best_params must be a non-empty dict."""
    ml = MLModel(clf_data, ml_algorithm="RFC", reg_class="classification",
                 parameters="grid", cv_fold=3)
    assert isinstance(ml.best_params, dict)
    assert len(ml.best_params) > 0


def test_grid_search_best_params_values_from_grid(clf_data):
    """best_params values must come from the (tiny) param grid."""
    ml = MLModel(clf_data, ml_algorithm="RFC", reg_class="classification",
                 parameters="grid", cv_fold=3)
    assert ml.best_params["n_estimators"] in [10, 20]


def test_grid_search_regression(reg_data):
    """Grid-search path must work for regression tasks."""
    ml = MLModel(reg_data, ml_algorithm="RFR", reg_class="regression",
                 parameters="grid", cv_fold=3, opt_metric="r2")
    assert isinstance(ml.cv_results, GridSearchCV)
    preds = ml.model.predict(reg_data.features)
    assert preds.shape == (len(reg_data.labels),)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_same_predictions(clf_data):
    """Two MLModel instances with the same seed must give identical predictions."""
    ml1 = MLModel(clf_data, ml_algorithm="RFC", reg_class="classification",
                  parameters="none", random_seed=7)
    ml2 = MLModel(clf_data, ml_algorithm="RFC", reg_class="classification",
                  parameters="none", random_seed=7)
    np.testing.assert_array_equal(
        ml1.model.predict(clf_data.features),
        ml2.model.predict(clf_data.features),
    )
