"""
Tests for chemagent.ml.cross_validation.

Covers:
- run_cross_validation returns a fitted GridSearchCV
- stratified vs. non-stratified splitters are used correctly
- get_cv_best_params returns a dict with the expected keys
- best CV score is within a sensible range
- result is deterministic across identical calls
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

from chemagent.ml.cross_validation import get_cv_best_params, run_cross_validation


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def classification_data():
    """Small balanced binary classification dataset (100 samples, 10 features)."""
    X = RNG.standard_normal((100, 10))
    y = RNG.integers(0, 2, size=100)
    return X, y


@pytest.fixture(scope="module")
def regression_data():
    """Small regression dataset (100 samples, 5 features)."""
    X = RNG.standard_normal((100, 5))
    y = X[:, 0] * 2.0 + RNG.standard_normal(100) * 0.5
    return X, y


@pytest.fixture
def rfc():
    """Fresh RandomForestClassifier with a fixed random state."""
    return RandomForestClassifier(random_state=0)


@pytest.fixture
def ridge():
    """Fresh Ridge regressor."""
    return Ridge()


@pytest.fixture
def cv_result_clf(classification_data, rfc):
    """Pre-fitted GridSearchCV on binary classification data.

    Reused by tests that only care about the shape/type of the result,
    not the specific parameter grid.
    """
    X, y = classification_data
    return run_cross_validation(rfc, {"n_estimators": [10, 50]}, X, y, cv_fold=3)


# ---------------------------------------------------------------------------
# run_cross_validation
# ---------------------------------------------------------------------------


def test_returns_fitted_gridsearchcv(classification_data, rfc):
    """Function must return a fitted GridSearchCV instance."""
    X, y = classification_data
    result = run_cross_validation(rfc, {"n_estimators": [10, 20]}, X, y, cv_fold=3)

    assert isinstance(result, GridSearchCV)
    assert hasattr(result, "best_index_")
    assert hasattr(result, "cv_results_")


def test_cv_results_has_expected_keys(classification_data, rfc):
    """cv_results_ must contain 'params' and mean test score."""
    X, y = classification_data
    result = run_cross_validation(rfc, {"n_estimators": [10, 30]}, X, y, cv_fold=3)

    assert "params" in result.cv_results_
    assert "mean_test_score" in result.cv_results_
    assert len(result.cv_results_["params"]) == 2  # two param combinations


def test_stratified_uses_stratifiedkfold(classification_data, rfc):
    """stratified=True (default) must use StratifiedKFold internally."""
    X, y = classification_data
    result = run_cross_validation(
        rfc, {"n_estimators": [10]}, X, y, cv_fold=4, stratified=True
    )

    assert isinstance(result.cv, StratifiedKFold)


def test_non_stratified_uses_kfold(regression_data, ridge):
    """stratified=False must use plain KFold internally."""
    X, y = regression_data
    result = run_cross_validation(
        ridge, {"alpha": [0.1, 1.0]}, X, y,
        cv_fold=4, scoring="r2", stratified=False,
    )

    assert isinstance(result.cv, KFold)


def test_correct_number_of_folds(classification_data, rfc):
    """The number of splits in the CV object must match cv_fold."""
    X, y = classification_data

    for n_folds in (3, 5):
        result = run_cross_validation(
            rfc, {"n_estimators": [10]}, X, y, cv_fold=n_folds
        )
        assert result.cv.n_splits == n_folds


def test_scoring_balanced_accuracy(classification_data, rfc):
    """Scoring with 'balanced_accuracy' should produce scores in [0, 1]."""
    X, y = classification_data
    result = run_cross_validation(
        rfc, {"n_estimators": [20]}, X, y,
        cv_fold=3, scoring="balanced_accuracy",
    )

    scores = result.cv_results_["mean_test_score"]
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_scoring_roc_auc(classification_data, rfc):
    """Scoring with 'roc_auc' should produce scores in [0, 1]."""
    X, y = classification_data
    result = run_cross_validation(
        rfc, {"n_estimators": [20]}, X, y,
        cv_fold=3, scoring="roc_auc",
    )

    scores = result.cv_results_["mean_test_score"]
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_deterministic_with_same_seed(classification_data, rfc):
    """Two calls with the same random_seed must produce identical best_index_."""
    X, y = classification_data
    param_grid = {"n_estimators": [10, 20, 50]}

    r1 = run_cross_validation(rfc, param_grid, X, y, cv_fold=3, random_seed=7)
    r2 = run_cross_validation(rfc, param_grid, X, y, cv_fold=3, random_seed=7)

    assert r1.best_index_ == r2.best_index_
    np.testing.assert_array_almost_equal(
        r1.cv_results_["mean_test_score"],
        r2.cv_results_["mean_test_score"],
    )


def test_regression_r2_score(regression_data, ridge):
    """R² scores from cross-validation on a linear problem should be positive."""
    X, y = regression_data
    result = run_cross_validation(
        ridge, {"alpha": [0.01, 0.1, 1.0]}, X, y,
        cv_fold=5, scoring="r2", stratified=False,
    )

    best_score = result.cv_results_["mean_test_score"][result.best_index_]
    assert best_score > 0.0


# ---------------------------------------------------------------------------
# get_cv_best_params
# ---------------------------------------------------------------------------


def test_get_cv_best_params_returns_dict(cv_result_clf):
    """get_cv_best_params must return a dict."""
    best = get_cv_best_params(cv_result_clf)

    assert isinstance(best, dict)


def test_get_cv_best_params_keys_match_grid(classification_data, rfc):
    """Best params must contain exactly the keys defined in param_grid."""
    X, y = classification_data
    param_grid = {"n_estimators": [10, 50], "max_depth": [3, 5]}

    cv_result = run_cross_validation(rfc, param_grid, X, y, cv_fold=3)
    best = get_cv_best_params(cv_result)

    assert set(best.keys()) == {"n_estimators", "max_depth"}


def test_get_cv_best_params_values_in_grid(classification_data, rfc):
    """Best param values must come from the param_grid options."""
    X, y = classification_data
    param_grid = {"n_estimators": [10, 30, 100]}

    cv_result = run_cross_validation(rfc, param_grid, X, y, cv_fold=3)
    best = get_cv_best_params(cv_result)

    assert best["n_estimators"] in [10, 30, 100]
