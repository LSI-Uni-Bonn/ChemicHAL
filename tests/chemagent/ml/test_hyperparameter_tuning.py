"""
Tests for chemagent.ml.hyperparameter_tuning.

Covers:
- get_param_grid returns a dict for known algorithms
- get_param_grid returns an empty dict for unknown algorithms
- Every value in a returned grid is a non-empty list (GridSearchCV requirement)
- register_param_grid adds a new algorithm and makes it retrievable
- register_param_grid replaces an existing grid
- HYPERPARAMETERS is the same object as models.PARAM_GRIDS (single source of truth)
"""

import pytest

from chemagent.ml import models
from chemagent.ml.hyperparameter_tuning import (
    HYPERPARAMETERS,
    get_param_grid,
    register_param_grid,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

KNOWN_ALGORITHMS = ["RFC", "RFR", "SVC"]


@pytest.fixture(autouse=True)
def restore_param_grids():
    """Snapshot PARAM_GRIDS before each test and restore it afterwards.

    This prevents register_param_grid tests from leaking state into other tests.
    """
    snapshot = {k: dict(v) for k, v in models.PARAM_GRIDS.items()}
    yield
    models.PARAM_GRIDS.clear()
    models.PARAM_GRIDS.update(snapshot)


# ---------------------------------------------------------------------------
# get_param_grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo", KNOWN_ALGORITHMS)
def test_get_param_grid_returns_dict(algo):
    """get_param_grid must return a dict for every registered algorithm."""
    result = get_param_grid(algo)
    assert isinstance(result, dict)


@pytest.mark.parametrize("algo", KNOWN_ALGORITHMS)
def test_get_param_grid_non_empty(algo):
    """Grids for registered algorithms must not be empty."""
    result = get_param_grid(algo)
    assert len(result) > 0


@pytest.mark.parametrize("algo", KNOWN_ALGORITHMS)
def test_get_param_grid_values_are_lists(algo):
    """Every value in a registered grid must be a non-empty list."""
    grid = get_param_grid(algo)
    for param, values in grid.items():
        assert isinstance(values, list), f"{algo}.{param} is not a list"
        assert len(values) > 0, f"{algo}.{param} is an empty list"


def test_get_param_grid_unknown_returns_empty():
    """An unregistered algorithm key must return an empty dict."""
    result = get_param_grid("NONEXISTENT_ALGO")
    assert result == {}


def test_get_param_grid_rfc_expected_keys():
    """RFC grid must contain the standard Random Forest hyperparameters."""
    grid = get_param_grid("RFC")
    for key in ("n_estimators", "max_features", "min_samples_split", "min_samples_leaf"):
        assert key in grid, f"Expected key '{key}' in RFC grid"


def test_get_param_grid_svc_expected_keys():
    """SVC grid must contain C, kernel, and gamma."""
    grid = get_param_grid("SVC")
    for key in ("C", "kernel", "gamma"):
        assert key in grid, f"Expected key '{key}' in SVC grid"


# ---------------------------------------------------------------------------
# register_param_grid
# ---------------------------------------------------------------------------


def test_register_new_algorithm():
    """A newly registered algorithm must be retrievable via get_param_grid."""
    register_param_grid("DUMMY", {"alpha": [0.1, 1.0]})
    result = get_param_grid("DUMMY")
    assert result == {"alpha": [0.1, 1.0]}


def test_register_replaces_existing_grid():
    """Registering an existing key must overwrite the previous grid."""
    original = get_param_grid("RFC")
    new_grid = {"n_estimators": [999]}
    register_param_grid("RFC", new_grid)
    assert get_param_grid("RFC") == new_grid
    assert get_param_grid("RFC") != original


def test_registered_grid_reflected_in_hyperparameters():
    """After registration HYPERPARAMETERS must contain the new key."""
    register_param_grid("TEST_ALGO", {"depth": [3, 5]})
    assert "TEST_ALGO" in HYPERPARAMETERS
    assert HYPERPARAMETERS["TEST_ALGO"] == {"depth": [3, 5]}


# ---------------------------------------------------------------------------
# HYPERPARAMETERS is the single source of truth
# ---------------------------------------------------------------------------


def test_hyperparameters_is_param_grids():
    """HYPERPARAMETERS must be the same object as models.PARAM_GRIDS."""
    assert HYPERPARAMETERS is models.PARAM_GRIDS


def test_mutation_via_hyperparameters_visible_in_get_param_grid():
    """Direct mutation of HYPERPARAMETERS must be reflected in get_param_grid."""
    HYPERPARAMETERS["DIRECT"] = {"C": [1, 10]}
    assert get_param_grid("DIRECT") == {"C": [1, 10]}
