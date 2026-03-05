"""
Tests for chemagent.ml.models.

Covers:
- build_estimator returns the correct estimator type per algorithm/task
- class_weight="balanced" is set when reg_class="classification-cw"
- probability=True is set on SVC
- ValueError is raised for unknown algorithms
- MODEL_INFO has the required structure for every registered algorithm
- PARAM_GRIDS and MODEL_INFO share the same keys (consistency)
"""

import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC

from chemagent.ml.models import MODEL_INFO, PARAM_GRIDS, build_estimator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLASSIFICATION_ALGOS = ["RFC", "SVC"]
REGRESSION_ALGOS = ["RFR"]
ALL_ALGOS = list(MODEL_INFO.keys())


# ---------------------------------------------------------------------------
# build_estimator — return types
# ---------------------------------------------------------------------------


def test_rfc_classification_returns_random_forest_classifier():
    est = build_estimator("RFC", "classification", random_seed=0)
    assert isinstance(est, RandomForestClassifier)


def test_rfc_classification_cw_returns_random_forest_classifier():
    est = build_estimator("RFC", "classification-cw", random_seed=0)
    assert isinstance(est, RandomForestClassifier)


def test_rfr_regression_returns_random_forest_regressor():
    est = build_estimator("RFR", "regression", random_seed=0)
    assert isinstance(est, RandomForestRegressor)


def test_svc_classification_returns_svc():
    est = build_estimator("SVC", "classification", random_seed=0)
    assert isinstance(est, SVC)


# ---------------------------------------------------------------------------
# build_estimator — random_seed is forwarded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo,task", [
    ("RFC", "classification"),
    ("RFR", "regression"),
])
def test_random_seed_is_stored(algo, task):
    """The estimator's random_state must equal the seed passed in."""
    est = build_estimator(algo, task, random_seed=99)
    assert est.random_state == 99


# ---------------------------------------------------------------------------
# build_estimator — class weighting
# ---------------------------------------------------------------------------


def test_rfc_classification_no_class_weight():
    """Plain classification must not set class_weight."""
    est = build_estimator("RFC", "classification", random_seed=0)
    assert est.class_weight is None


def test_rfc_classification_cw_balanced():
    """class_weight must be 'balanced' when reg_class='classification-cw'."""
    est = build_estimator("RFC", "classification-cw", random_seed=0)
    assert est.class_weight == "balanced"


def test_svc_classification_no_class_weight():
    est = build_estimator("SVC", "classification", random_seed=0)
    assert est.class_weight is None


def test_svc_classification_cw_balanced():
    est = build_estimator("SVC", "classification-cw", random_seed=0)
    assert est.class_weight == "balanced"


# ---------------------------------------------------------------------------
# build_estimator — SVC probability flag
# ---------------------------------------------------------------------------


def test_svc_has_probability_true():
    """SVC must always be built with probability=True."""
    est = build_estimator("SVC", "classification", random_seed=0)
    assert est.probability is True


# ---------------------------------------------------------------------------
# build_estimator — unknown algorithm raises ValueError
# ---------------------------------------------------------------------------


def test_unknown_algorithm_raises_value_error():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        build_estimator("NONEXISTENT", "classification", random_seed=0)


def test_error_message_lists_known_algorithms():
    """The ValueError message must mention at least one registered algorithm."""
    with pytest.raises(ValueError) as exc_info:
        build_estimator("BAD", "classification", random_seed=0)
    error_msg = str(exc_info.value)
    assert any(algo in error_msg for algo in ALL_ALGOS)


# ---------------------------------------------------------------------------
# MODEL_INFO structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo", ALL_ALGOS)
def test_model_info_has_required_keys(algo):
    """Every MODEL_INFO entry must have 'description', 'task', 'extra_deps'."""
    info = MODEL_INFO[algo]
    for key in ("description", "task", "extra_deps"):
        assert key in info, f"MODEL_INFO['{algo}'] missing key '{key}'"


@pytest.mark.parametrize("algo", ALL_ALGOS)
def test_model_info_task_is_valid(algo):
    """'task' must be one of 'classification', 'regression', or 'both'."""
    task = MODEL_INFO[algo]["task"]
    assert task in ("classification", "regression", "both"), \
        f"MODEL_INFO['{algo}']['task'] = {task!r} is not a valid value"


@pytest.mark.parametrize("algo", ALL_ALGOS)
def test_model_info_extra_deps_is_list(algo):
    """'extra_deps' must be a list."""
    assert isinstance(MODEL_INFO[algo]["extra_deps"], list)


@pytest.mark.parametrize("algo", ALL_ALGOS)
def test_model_info_description_is_non_empty_string(algo):
    """'description' must be a non-empty string."""
    desc = MODEL_INFO[algo]["description"]
    assert isinstance(desc, str) and len(desc) > 0


# ---------------------------------------------------------------------------
# PARAM_GRIDS ↔ MODEL_INFO consistency
# ---------------------------------------------------------------------------


def test_param_grids_keys_subset_of_model_info():
    """Every key in PARAM_GRIDS must also be in MODEL_INFO."""
    for algo in PARAM_GRIDS:
        assert algo in MODEL_INFO, \
            f"PARAM_GRIDS has '{algo}' but MODEL_INFO does not"


def test_param_grids_values_are_non_empty_lists():
    """Every parameter grid value must be a non-empty list."""
    for algo, grid in PARAM_GRIDS.items():
        for param, values in grid.items():
            assert isinstance(values, list) and len(values) > 0, \
                f"PARAM_GRIDS['{algo}']['{param}'] is not a non-empty list"
