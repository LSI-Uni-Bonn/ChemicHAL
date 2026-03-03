"""
Shared pytest fixtures for all chemagent.plots tests.

All fixtures use fixed random seeds so tests are deterministic.
The 'agg' backend makes every test headless (no display required).
"""

import matplotlib
matplotlib.use("Agg")          # must be set before importing pyplot

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Classification data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def binary_labels() -> np.ndarray:
    """200 binary ground-truth labels, roughly balanced."""
    return RNG.integers(0, 2, size=200).astype(int)


@pytest.fixture(scope="session")
def binary_preds(binary_labels) -> np.ndarray:
    """Noisy binary predictions (flip ~25 % of labels)."""
    preds = binary_labels.copy()
    flip = RNG.choice(len(preds), size=50, replace=False)
    preds[flip] = 1 - preds[flip]
    return preds


@pytest.fixture(scope="session")
def binary_proba(binary_labels) -> np.ndarray:
    """Soft probabilities, shape (200, 2)."""
    raw = RNG.uniform(0, 1, size=200)
    raw = np.where(binary_labels == 1, raw * 0.4 + 0.5, raw * 0.4 + 0.1)
    raw = np.clip(raw, 0.01, 0.99)
    return np.column_stack([1 - raw, raw])


@pytest.fixture(scope="session")
def multiclass_labels() -> np.ndarray:
    """300 three-class ground-truth labels."""
    return RNG.integers(0, 3, size=300).astype(int)


@pytest.fixture(scope="session")
def multiclass_preds(multiclass_labels) -> np.ndarray:
    """Noisy multiclass predictions."""
    preds = multiclass_labels.copy()
    flip = RNG.choice(len(preds), size=75, replace=False)
    preds[flip] = RNG.integers(0, 3, size=75)
    return preds


# ---------------------------------------------------------------------------
# Classification metrics dict
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def binary_metrics_dict() -> dict:
    return {
        "Accuracy": 0.85,
        "F1": 0.82,
        "AUC": 0.91,
        "MCC": 0.70,
        "BA": 0.83,
        "Precision": 0.84,
        "Recall": 0.80,
        "Average Precision": 0.88,
    }


# ---------------------------------------------------------------------------
# Regression data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def regression_true() -> np.ndarray:
    """200 ground-truth continuous values."""
    return RNG.uniform(-3, 3, size=200)


@pytest.fixture(scope="session")
def regression_pred(regression_true) -> np.ndarray:
    """Noisy predictions around the true values."""
    return regression_true + RNG.normal(0, 0.5, size=len(regression_true))


# ---------------------------------------------------------------------------
# Fitted tree model (for feature importance)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fitted_rf(binary_labels) -> RandomForestClassifier:
    X = RNG.standard_normal((200, 50))
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X, binary_labels)
    return clf


@pytest.fixture(scope="session")
def feature_names_50() -> list[str]:
    return [f"feature_{i:03d}" for i in range(50)]


# ---------------------------------------------------------------------------
# Dataset / split data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_dataframe() -> pd.DataFrame:
    n = 150
    return pd.DataFrame({
        "activity": RNG.normal(0, 1, size=n),
        "mw": RNG.uniform(200, 800, size=n),
        "label": RNG.integers(0, 2, size=n),
        "group": RNG.choice(["A", "B", "C"], size=n),
    })


@pytest.fixture(scope="session")
def split_stats() -> dict:
    return {
        "train": {"count": 120, "percentage": 60.0},
        "val":   {"count": 30,  "percentage": 15.0},
        "test":  {"count": 50,  "percentage": 25.0},
        "total": {"count": 200, "percentage": 100.0},
    }


@pytest.fixture(scope="session")
def class_dist() -> dict:
    return {
        "train": {"0": 65, "1": 55},
        "val":   {"0": 14, "1": 16},
        "test":  {"0": 24, "1": 26},
    }
