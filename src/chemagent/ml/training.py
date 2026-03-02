"""
Model construction, hyperparameter optimisation, and fitting.

The ``MLModel`` class encapsulates the full training pipeline:
  1. Instantiate the base scikit-learn estimator.
  2. Run GridSearchCV cross-validation to find the best hyperparameters.
  3. Refit the model with those parameters on the full training set.

Usage
-----
    from chemagent.ml.training import MLModel

    trained = MLModel(data, ml_algorithm="RFC", reg_class="classification")
    predictions = trained.model.predict(X_test)
"""

import warnings

import numpy as np

from .models import build_estimator, PARAM_GRIDS
from .hyperparameter_tuning import get_param_grid
from .cross_validation import run_cross_validation, get_cv_best_params


# Backward-compatible alias so any code still calling _build_estimator() works.
_build_estimator = build_estimator


class MLModel:
    """Train a scikit-learn model with GridSearchCV hyperparameter tuning.

    Parameters
    ----------
    data:
        Object with `.features`, `.labels`, and `.class_labels` attributes,
        as produced by the dataset-loader pipeline.
    ml_algorithm:
        Algorithm key: ``"RFC"``, ``"RFR"``, or ``"SVC"``.
    opt_metric:
        Scoring string for ``GridSearchCV`` (e.g. ``"balanced_accuracy"``).
        ``None`` uses the estimator's default.
    reg_class:
        Task type: ``"classification"``, ``"classification-cw"``, or
        ``"regression"``.
    parameters:
        ``"grid"`` to use predefined hyperparameter grids, or ``"none"``/
        any other value to skip grid search and use default parameters.
    cv_fold:
        Number of cross-validation folds (default 5).
    random_seed:
        Global random seed (default 42).

    Attributes
    ----------
    model:
        Fitted scikit-learn estimator (best hyperparameters, full data).
    cv_results:
        Fitted ``GridSearchCV`` object from the tuning step.
    best_params:
        Best parameter dict extracted from *cv_results*.
    """

    def __init__(
        self,
        data,
        ml_algorithm: str,
        opt_metric: str | None = None,
        reg_class: str | None = None,
        parameters: str = "grid",
        cv_fold: int = 5,
        random_seed: int = 42,
    ) -> None:
        self.data = data
        self.ml_algorithm = ml_algorithm
        self.opt_metric = opt_metric
        self.reg_class = reg_class
        self.cv_fold = cv_fold
        self.seed = random_seed

        self.h_parameters = get_param_grid(ml_algorithm) if parameters == "grid" else {}

        self.model = build_estimator(ml_algorithm, reg_class, random_seed)

        if self.h_parameters:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.cv_results = run_cross_validation(
                    model=self.model,
                    param_grid=self.h_parameters,
                    features=data.features,
                    labels=data.class_labels,
                    cv_fold=cv_fold,
                    scoring=opt_metric,
                    stratified=(reg_class != "regression"),
                    random_seed=random_seed,
                )
            self.best_params = get_cv_best_params(self.cv_results)
        else:
            self.cv_results = None
            self.best_params = {}

        self.model = self.model.set_params(**self.best_params).fit(
            data.features, data.labels
        )
