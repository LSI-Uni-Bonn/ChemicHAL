"""
chemagent.ml.models — central catalogue of all supported ML estimators.

This is the **only file you need to edit** to add a new model.

Sections
--------
1. Hyperparameter grids  — ``PARAM_GRIDS``
2. Estimator factories   — ``build_estimator()``
3. Metadata              — ``MODEL_INFO``

How to add a new model
----------------------
Step 1 — import the estimator (add to the sklearn imports block below).

Step 2 — add a param grid::

    PARAM_GRIDS["RFC"] = {
        "n_estimators": [100, 300],
        "max_depth":    [3, 6],
    }

Step 3 — add a factory branch inside ``build_estimator()``::

    if algorithm == "RFC":
        if reg_class == "regression":
            return RandomForestRegressor(random_state=random_seed)
        return RandomForestClassifier(
            random_state=random_seed,
            class_weight="balanced" if class_weighted else None,
            n_jobs=-1,
        )

Step 4 — add metadata::

    MODEL_INFO["RFC"] = {
        "description": "Random Forest Classifier — ensemble of decision trees; handles multi-class, supports class weighting (task='classification-cw').",
        "task":        "both",

    }

That's it — the new key is automatically available everywhere:
``train_model``, ``build_model_from_split_file``,
``get_available_algorithms``, and ``get_param_grid``.
"""

from __future__ import annotations

from typing import Any

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC

from .dnn_model import DNNClassifier, DNNRegressor


# 1. Hyperparameter grids
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "RFC": {
        "n_estimators":      [50, 100, 200],
        "max_features":      ["sqrt", "log2"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
    },
    "RFR": {
        "n_estimators":      [50, 100, 200],
        "max_features":      ["sqrt", "log2"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
    },
    "SVC": {
        "C":      [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma":  ["scale", "auto"],
    },
    "BRF": {
        "n_estimators":      [50, 100, 200],
        "max_features":      ["sqrt", "log2"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
    },
    # ── add new grids below ──────────────────────────────────────────────────
}


# 2. Estimator factories
def build_estimator(algorithm: str, reg_class: str, random_seed: int) -> Any:
    """Return an unfitted scikit-learn estimator for *algorithm*.

    Args:
    algorithm:
        Registered key, e.g. ``"RFC"``.
    reg_class:
        Task type: ``"classification"``, ``"classification-cw"``,
        or ``"regression"``.
    random_seed:
        Forwarded to the estimator's ``random_state`` parameter.

    Raises:
    ValueError
        If *algorithm* is not defined in this function.
    """
    class_weighted = reg_class == "classification-cw"

    #built-in models

    if algorithm == "RFC":
        return RandomForestClassifier(
            random_state=random_seed,
            class_weight="balanced" if class_weighted else None,
        )

    if algorithm == "RFR":
        return RandomForestRegressor(random_state=random_seed)

    if algorithm == "SVC":
        return SVC(
            random_state=random_seed,
            class_weight="balanced" if class_weighted else None,
            probability=True,
        )

    if algorithm == "DNN":
        if reg_class == "regression":
            return DNNRegressor(random_seed=random_seed)
        return DNNClassifier(
            random_seed=random_seed,
            class_weight="balanced" if class_weighted else None,
        )

    if algorithm == "BRF":
        return BalancedRandomForestClassifier(
            random_state=random_seed,
            n_jobs=-1,
        )

    #add new algorithms below

    raise ValueError(
        f"Unknown algorithm {algorithm!r}. "
        f"Defined models: {list(MODEL_INFO)}. "
        "Add a new branch in build_estimator() inside models.py."
    )


# 3. Metadata  (used by get_available_algorithms MCP tool)

MODEL_INFO: dict[str, dict] = {
    "RFC": {
        "description": "Random Forest Classifier — ensemble of decision trees; handles multi-class, supports class weighting (task='classification-cw').",
        "task":        "classification",
        "extra_deps":  [],
    },
    "RFR": {
        "description": "Random Forest Regressor — ensemble of decision trees for continuous target prediction (e.g. pPot_diff).",
        "task":        "regression",
        "extra_deps":  [],
    },
    "SVC": {
        "description": "Support Vector Classifier — SVM with RBF/linear kernels; probability estimates enabled; supports class weighting (task='classification-cw').",
        "task":        "classification",
        "extra_deps":  [],
    },
    "DNN": {
        "description": "Feed-forward PyTorch neural network; supports classification, class-weighted classification, and regression.",
        "task":        "both",
        "extra_deps":  ["torch"],
    },
    "BRF": {
        "description": "Balanced Random Forest Classifier — samples each bootstrap with replacement to balance classes; suitable for imbalanced datasets.",
        "task":        "classification",
        "extra_deps":  ["imbalanced-learn"],
    },
    #add new metadata below
}
