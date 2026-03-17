"""
Hyperparameter grids and helpers for scikit-learn models.

Grids and estimator factories now live in :mod:`chemagent.ml.models`.
This module re-exports them for backward compatibility and provides
``get_param_grid`` / ``register_param_grid`` as convenience helpers.

Usage
-----
    from chemagent.ml.hyperparameter_tuning import get_param_grid

    grid = get_param_grid("RFC")   # {'n_estimators': [...], ...}
"""

from .models import PARAM_GRIDS as HYPERPARAMETERS


def get_param_grid(algorithm: str) -> dict:
    """Return the hyperparameter grid for *algorithm*.

    Reads directly from :data:`chemagent.ml.models.PARAM_GRIDS`, so adding
    an entry there is sufficient — no changes needed here.

    Parameters
    ----------
    algorithm:
        Registered key, e.g. ``"RFC"``.

    Returns
    -------
    dict
        Parameter grid suitable for ``GridSearchCV(param_grid=...)``.
        Returns an empty dict if *algorithm* is not registered.
    """
    return HYPERPARAMETERS.get(algorithm, {})


def register_param_grid(algorithm: str, grid: dict) -> None:
    """Register or replace the hyperparameter grid for *algorithm*.

    Mutates :data:`chemagent.ml.models.PARAM_GRIDS` directly, so the change
    is reflected everywhere (``get_param_grid``, ``get_available_algorithms``,
    ``MLModel``, etc.).

    Parameters
    ----------
    algorithm:
        Algorithm key, e.g. ``"XGB"``.
    grid:
        Dict mapping parameter names to lists of candidate values.
    """
    HYPERPARAMETERS[algorithm] = grid
