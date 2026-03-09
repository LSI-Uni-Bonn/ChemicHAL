"""
chemagent.explainability.shap_explainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Thin wrapper that selects the right SHAP explainer for a trained sklearn model.

Supported models
----------------
* RandomForestClassifier / RandomForestRegressor  →  ``shap.TreeExplainer``
* SVC                                             →  ``shap.KernelExplainer``

Usage
-----
    from chemagent.explainability.shap_explainer import SHAPExplainer
    import joblib

    model   = joblib.load("model.pkl")
    X_train = ...   # background / reference data (required for SVC)
    X_test  = ...

    explainer   = SHAPExplainer(model, background=X_train)
    shap_values = explainer.explain(X_test)   # shape (n_samples, n_features)
"""

from __future__ import annotations

import numpy as np
import joblib
import shap


# Tree-based model class names that TreeExplainer supports natively.
_TREE_MODEL_NAMES: frozenset[str] = frozenset(
    {
        "RandomForestClassifier",
        "RandomForestRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "DecisionTreeClassifier",
        "DecisionTreeRegressor",
    }
)


class SHAPExplainer:
    """Compute SHAP values for a trained sklearn estimator.

    Parameters
    ----------
    model:
        Fitted scikit-learn estimator.
    background:
        Reference dataset for KernelExplainer (required for SVC).
        Ignored for tree-based models.
    n_background_clusters:
        Number of k-means clusters used to compress the background dataset
        for KernelExplainer (default 50).  Ignored for tree-based models.
    """

    def __init__(
        self,
        model,
        background: np.ndarray | None = None,
        n_background_clusters: int = 50,
    ) -> None:
        self.model = model
        self._is_tree = type(model).__name__ in _TREE_MODEL_NAMES
        self._explainer = self._build_explainer(model, background)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for *X*.

        Parameters
        ----------
        X:
            Feature matrix, shape ``(n_samples, n_features)``.

        Returns
        -------
        np.ndarray, shape ``(n_samples, n_features)``
            For binary classifiers the values correspond to the positive
            class (index 1).  For regressors a single 2-D array is returned.
        """
        sv = self._explainer.shap_values(X)
        # Tree binary classifiers return a list [class0_sv, class1_sv].
        if isinstance(sv, list) and len(sv) == 2:
            return np.asarray(sv[1])
        return np.asarray(sv)

    @property
    def expected_value(self) -> float:
        """Base value (mean model output) for the positive class / regression."""
        ev = self._explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = np.atleast_1d(ev)
            return float(ev[1]) if len(ev) == 2 else float(ev[0])
        return float(ev)

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        background: np.ndarray | None = None,
    ) -> "SHAPExplainer":
        """Load model from *model_path* and build the explainer.

        Parameters
        ----------
        model_path:
            Path to a ``joblib``-serialised sklearn model (``.pkl``).
        background:
            Reference dataset for KernelExplainer (required for SVC).
        """
        model = joblib.load(model_path)
        return cls(model, background=background)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_explainer(model, background,):
        model_name = type(model).__name__
        if model_name in _TREE_MODEL_NAMES:
            return shap.TreeExplainer(model)

        # Fallback: model-agnostic KernelExplainer (e.g. SVC)
        if background is None:
            raise ValueError(
                f"background data is required for KernelExplainer (model={model_name!r}). "
                "Pass the training feature matrix as the 'background' argument."
            )
        return shap.KernelExplainer(model.predict_proba, background)
