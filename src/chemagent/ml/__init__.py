"""
chemagent.ml — modular machine-learning utilities.

Sub-modules
-----------
models              : ALL estimator factories, param grids, and metadata ← edit here
hyperparameter_tuning  : get_param_grid / register_param_grid helpers
cross_validation       : k-fold cross-validation wrappers
training               : model construction and fitting (MLModel)
metrics                : low-level metric computations
evaluation             : high-level evaluation reports (Model_Evaluation)
"""

from .models import build_estimator, PARAM_GRIDS, MODEL_INFO
from .hyperparameter_tuning import HYPERPARAMETERS, get_param_grid
from .cross_validation import run_cross_validation
from .training import MLModel
from .metrics import confusion_components, classification_metrics, regression_metrics
from .evaluation import Model_Evaluation

__all__ = [
    # models.py — single source of truth
    "build_estimator",
    "PARAM_GRIDS",
    "MODEL_INFO",
    # hyperparameter_tuning.py
    "HYPERPARAMETERS",
    "get_param_grid",
    # cross_validation.py
    "run_cross_validation",
    # training.py
    "MLModel",
    # metrics.py
    "confusion_components",
    "classification_metrics",
    "regression_metrics",
    # evaluation.py
    "Model_Evaluation",
]
