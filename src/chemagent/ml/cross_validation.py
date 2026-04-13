"""
K-fold cross-validation wrappers using scikit-learn GridSearchCV.

Usage
-----
    from chemagent.ml.cross_validation import run_cross_validation

    cv_results = run_cross_validation(
        model, param_grid={"n_estimators": [50, 100]},
        features=X_train, labels=y_train,
        cv_fold=5, scoring="balanced_accuracy",
    )
    best_params = cv_results.cv_results_["params"][cv_results.best_index_]
"""

import warnings
from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold


def run_cross_validation(
    model: Any,
    param_grid: dict,
    features: np.ndarray,
    labels: np.ndarray,
    cv_fold: int = 5,
    scoring: str | None = None,
    stratified: bool = True,
    random_seed: int = 42,
    n_jobs: int = -1,
) -> GridSearchCV:
    """Run GridSearchCV cross-validation and return the fitted searcher.

    Args:
    model:
        Unfitted scikit-learn estimator.
    param_grid:
        Parameter grid for ``GridSearchCV``.
    features:
        2-D feature array (n_samples, n_features).
    labels:
        1-D label array (n_samples,).
    cv_fold:
        Number of cross-validation folds.
    scoring:
        Scoring metric string accepted by scikit-learn, e.g.
        ``"balanced_accuracy"``, ``"roc_auc"``, ``"r2"``.
        ``None`` uses the estimator's default scorer.
    stratified:
        If ``True`` (default), use ``StratifiedKFold`` to preserve class
        distribution; set to ``False`` for regression tasks.
    random_seed:
        Random state for fold splitting.
    n_jobs:
        Number of parallel jobs for ``GridSearchCV``.

    Returns:
    GridSearchCV
        Fitted searcher with ``best_index_`` and ``cv_results_`` populated.
    """
    if stratified:
        cv = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=random_seed)
    else:
        cv = KFold(n_splits=cv_fold, shuffle=True, random_state=random_seed)

    searcher = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=0,
        refit=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        searcher.fit(features, labels)
    return searcher


def get_cv_best_params(cv_results: GridSearchCV) -> dict:
    """Extract the best parameter dict from a fitted ``GridSearchCV`` object.

    Args:
    cv_results:
        A fitted ``GridSearchCV`` instance.

    Returns:
    dict
        Best parameter combination found during cross-validation.
    """
    return cv_results.cv_results_["params"][cv_results.best_index_]
