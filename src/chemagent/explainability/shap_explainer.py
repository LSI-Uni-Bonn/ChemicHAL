"""
chemagent.explainability.shap_explainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Thin wrapper that selects the right SHAP explainer for a trained sklearn model.

Important routing rule for LLM tool callers
-------------------------------------------
These SHAP helpers are for tabular sklearn-style models saved as .pkl/.joblib
(RFC/RFR/SVC/DNN tabular path). They are not for GNN checkpoints (.pt/.pth).
For GNN explanations, use explain_gnn_with_edgeshaper.

Supported models
----------------
* RandomForestClassifier / RandomForestRegressor  →  ``shap.TreeExplainer``
* DNNClassifier / DNNRegressor                    →  ``shap.GradientExplainer`` (Deep fallback)
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

import inspect as _inspect
from pathlib import Path
from typing import Any, Literal, Optional

import joblib
import numpy as np
import shap
from mcp.server.fastmcp import Image as MCPImage
from rdkit import Chem

from chemagent.datasets.featurizer import available_featurizers
from chemagent.explainability.mol_shap_draw import (
    get_atom_wise_weight_map,
    get_ecfp_morgan_generator_bit_info,
    get_top_k_bit_environments_with_contribution,
    render_top_k_parent_molecule_environment_highlights,
    shap_to_atom_weight,
)
from chemagent.session_utils import (
    get_session_logger as _get_session_logger,
    resolve_path as _resolve_path,
)


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

    Args:
    model:
        Fitted scikit-learn estimator.
    background:
        Reference dataset for KernelExplainer (required for SVC).
        Ignored for tree-based models.
    """

    def __init__(
        self,
        model,
        background: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self._is_tree = type(model).__name__ in _TREE_MODEL_NAMES
        self._is_dnn = self._is_dnn_model(model)
        self._explainer = self._build_explainer(model, background)

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for *X*.

        Args:
        X:
            Feature matrix, shape ``(n_samples, n_features)``.

        Returns:
        np.ndarray, shape ``(n_samples, n_features)``
            For binary classifiers the values correspond to the positive
            class (index 1).  For regressors a single 2-D array is returned.
        """
        X_in = self.model.as_torch_tensor(X) if self._is_dnn else X
        sv = self._normalise_shap_values(self._explainer.shap_values(X_in))
        # binary classifiers return a list [class0_sv, class1_sv].
        if isinstance(sv, list) and len(sv) == 2:
            return np.asarray(sv[1])
        sv = np.asarray(sv)
        if sv.ndim == 3 and sv.shape[-1] == 1:
            return sv[..., 0]
        #binary classifiers return 3-D (n_samples, n_features, n_classes).
        if sv.ndim == 3 and sv.shape[-1] == 2:
            return sv[..., 1]
        return sv

    def explain_per_predicted_class(
        self, X: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """Compute SHAP values selecting each sample's predicted class slice.

        For classifiers this returns, per compound, the SHAP values that
        correspond to the class the model actually predicted—rather than a
        fixed class index.

        For an output of shape ``(n_samples, n_features, n_classes)`` the
        selection is::

            result[i] = sv[i, :, predicted_class_index[i]]

        For 2-D output (regression or already-reduced classifiers) the
        array is returned unchanged.

        Args:
        X:
            Feature matrix, shape ``(n_samples, n_features)``.
        y_pred:
            Predicted class labels, shape ``(n_samples,)``.

        Returns:
        np.ndarray, shape ``(n_samples, n_features)``
        """
        X_in = self.model.as_torch_tensor(X) if self._is_dnn else X
        sv = self._normalise_shap_values(self._explainer.shap_values(X_in))

        # Normalise to ndarray
        if isinstance(sv, list):
            sv = np.stack(sv, axis=-1)   # list of (n, f) → (n, f, c)
        else:
            sv = np.asarray(sv)

        # 2-D: regression or single-output
        if sv.ndim == 2:
            return sv

        if sv.ndim == 3 and sv.shape[-1] == 1:
            return sv[..., 0]

        # 3-D (n_samples, n_features, n_classes): select predicted class per sample
        classes = list(getattr(self.model, 'classes_', range(sv.shape[-1])))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx = np.array([class_to_idx[p] for p in y_pred], dtype=int)
        return sv[np.arange(len(idx)), :, idx]

    @property
    def expected_value(self) -> float:
        """Backward-compatible scalar baseline.

        For binary classification this returns class-1 baseline, for single-output
        models it returns the only baseline, and for multiclass it returns the
        first class baseline (legacy behaviour).
        """
        ev = self.expected_values
        return float(ev[1]) if ev.size == 2 else float(ev[0])

    @property
    def expected_values(self) -> np.ndarray:
        """All SHAP base values as a 1-D float array."""
        ev = self._explainer.expected_value
        if ev is None:
            raise TypeError("SHAP explainer returned None for expected_value")

        if callable(ev):
            raise TypeError("Unsupported callable SHAP expected_value")

        ev = self._to_numpy(ev)

        if isinstance(ev, (list, tuple, np.ndarray)):
            ev_arr = np.atleast_1d(np.asarray(ev, dtype=float)).reshape(-1)
            if ev_arr.size == 0:
                raise TypeError("SHAP explainer returned empty expected_value")
            return ev_arr
        if isinstance(ev, (int, float, np.number)):
            return np.array([float(ev)], dtype=float)

        raise TypeError(f"Unsupported type for SHAP expected_value: {type(ev)!r}")

    def expected_values_for_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        """Return per-instance baselines aligned to predicted class labels.

        For multiclass outputs (len(expected_values) > 2), this selects each
        instance baseline from the predicted class.
        For binary and single-output models, it returns a constant vector using
        the backward-compatible scalar baseline.
        """
        y_pred_arr = np.asarray(y_pred)
        if y_pred_arr.ndim == 0:
            y_pred_arr = y_pred_arr.reshape(1)

        ev = self.expected_values
        if ev.size <= 2:
            return np.full(y_pred_arr.shape[0], self.expected_value, dtype=float)

        classes = list(getattr(self.model, "classes_", range(ev.size)))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        try:
            idx = np.array([class_to_idx[p] for p in y_pred_arr], dtype=int)
        except KeyError as err:
            raise ValueError(
                f"Predicted class {err.args[0]!r} not found in model classes {classes}."
            ) from err

        return ev[idx]

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        background: np.ndarray | None = None,
    ) -> "SHAPExplainer":
        """Load model from *model_path* and build the explainer.

        Args:
        model_path:
            Path to a ``joblib``-serialised sklearn model (``.pkl``).
        background:
            Reference dataset for KernelExplainer (required for SVC).
        """
        model = joblib.load(model_path)
        return cls(model, background=background)

    @staticmethod
    def _to_numpy(value: Any) -> Any:
        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
            return value.detach().cpu().numpy()
        return value

    @classmethod
    def _normalise_shap_values(cls, sv: Any) -> Any:
        if isinstance(sv, list):
            return [cls._to_numpy(x) for x in sv]
        if isinstance(sv, tuple):
            return tuple(cls._to_numpy(x) for x in sv)
        return cls._to_numpy(sv)

    @staticmethod
    def _is_dnn_model(model: Any) -> bool:
        return all(
            hasattr(model, attr)
            for attr in ("get_torch_model", "as_torch_tensor", "predict")
        )

    @staticmethod
    def _sample_background(background: np.ndarray, max_background: int = 128) -> np.ndarray:
        bg = np.asarray(background)
        if bg.ndim != 2:
            raise ValueError(f"Expected 2D background features, got shape {bg.shape}")
        n_samples = int(bg.shape[0])
        if n_samples <= max_background:
            return bg
        sampled = shap.sample(bg, nsamples=max_background, random_state=0)
        return np.asarray(sampled)

    @staticmethod
    def _build_explainer(model, background,):
        model_name = type(model).__name__
        if model_name in _TREE_MODEL_NAMES:
            return shap.TreeExplainer(model, feature_perturbation="interventional", data=background)

        if SHAPExplainer._is_dnn_model(model):
            if background is None:
                raise ValueError(
                    f"background data is required for DNN SHAP explainers (model={model_name!r}). "
                    "Pass the training feature matrix as the 'background' argument."
                )
            sampled_bg = SHAPExplainer._sample_background(background, max_background=128)
            bg_tensor = model.as_torch_tensor(sampled_bg)
            torch_model = model.get_torch_model()

            return shap.DeepExplainer(torch_model, bg_tensor)

        # Fallback: model-agnostic KernelExplainer (e.g. SVC)
        if background is None:
            raise ValueError(
                f"background data is required for KernelExplainer (model={model_name!r}). "
                "Pass the training feature matrix as the 'background' argument."
            )
        sampled_bg = SHAPExplainer._sample_background(background, max_background=128)
        if hasattr(model, "predict_proba"):
            return shap.KernelExplainer(model.predict_proba, sampled_bg)
        return shap.KernelExplainer(model.predict, sampled_bg)


def _load_split_data_with_required_background(
    split_file_path: str,
) -> tuple[dict[str, Any], np.ndarray, str]:
    """Load split data and enforce presence of 2D train_features background."""
    if split_file_path is None or not str(split_file_path).strip():
        raise ValueError(
            "split_file_path is required and must point to a split .pkl file "
            "from split_dataset() so SHAP background data is always available."
        )

    resolved_path = _resolve_path(str(split_file_path))
    split_path = Path(resolved_path)
    if not split_path.exists():
        raise ValueError(
            f"split_file_path does not exist: {resolved_path}. "
            "Provide a valid split .pkl path from split_dataset()."
        )

    split_data = joblib.load(split_path)
    if not isinstance(split_data, dict):
        raise ValueError(
            "split_file_path must reference a dict-like split payload produced by split_dataset()."
        )
    if "train_features" not in split_data:
        raise ValueError(
            "split_file_path must contain 'train_features' to provide SHAP background data."
        )

    background = np.asarray(split_data["train_features"])
    if background.ndim != 2:
        raise ValueError(
            f"Expected 2D train_features in split file, got shape {background.shape}."
        )

    return split_data, background, str(split_path)


def _find_latest_shap_payload_path() -> Optional[Path]:
    """Find the most recent SHAP payload file from session/workspace logs."""
    candidates: list[Path] = []

    session_dir = _get_session_logger().session_dir
    session_results = session_dir / "results"
    if session_results.exists():
        candidates.extend(session_results.glob("*_shap.pkl"))

    logs_root = Path(_resolve_path("data/logs"))
    if logs_root.exists():
        candidates.extend(logs_root.glob("session_*/results/*_shap.pkl"))

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_shap_payload(shap_values_path: Optional[str] = None) -> tuple[dict[str, Any], str, np.ndarray]:
    """Load SHAP payload produced by explain_with_shap/explain_smiles_with_shap."""
    if shap_values_path is None or not str(shap_values_path).strip():
        payload_path = _find_latest_shap_payload_path()
        if payload_path is None:
            raise ValueError(
                "No SHAP payload file found. Provide shap_values_path or run explain_with_shap/"
                "explain_smiles_with_shap first."
            )
    else:
        resolved_path = _resolve_path(str(shap_values_path))
        payload_path = Path(resolved_path)
        if not payload_path.exists():
            raise ValueError(f"shap_values_path does not exist: {resolved_path}.")

    payload = joblib.load(payload_path)
    if not isinstance(payload, dict):
        raise ValueError("SHAP payload must be a dict produced by explain_with_shap/explain_smiles_with_shap.")
    if "shap_values" not in payload:
        raise ValueError("SHAP payload is missing required key 'shap_values'.")

    shap_values = np.asarray(payload["shap_values"], dtype=float)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    if shap_values.ndim != 2:
        raise ValueError(
            f"Expected shap_values to be 1D/2D, got shape {shap_values.shape}."
        )

    return payload, str(payload_path), shap_values


def _get_shap_sample_context(
    payload: dict[str, Any],
    shap_values: np.ndarray,
    sample_index: int,
) -> tuple[str, np.ndarray, int, int]:
    """Extract one sample (smiles + SHAP vector + fingerprint params)."""
    smiles_arr = payload.get("smiles")
    if smiles_arr is None:
        raise ValueError(
            "No SMILES found in SHAP payload. Re-run explain_with_shap on a split with smiles, "
            "or use explain_smiles_with_shap."
        )

    if sample_index < 0 or sample_index >= shap_values.shape[0]:
        raise ValueError(
            f"sample_index {sample_index} out of range for {shap_values.shape[0]} samples."
        )

    smiles_np = np.asarray(smiles_arr)
    if smiles_np.shape[0] != shap_values.shape[0]:
        raise ValueError(
            "SHAP payload has inconsistent lengths between smiles and shap_values."
        )

    smiles = str(smiles_np[sample_index])
    shap_vector = np.asarray(shap_values[sample_index], dtype=float)
    radius = int(payload.get("radius", 2))
    n_bits = int(shap_vector.shape[0])
    return smiles, shap_vector, radius, n_bits


def explain_with_shap(
    model_path: str,
    split_file_path: str,
    split: Literal["train", "val", "test"] = "test",
    n_bits: int = 2048,
    correct_only: bool = True,
    save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Compute SHAP values for compounds already present in a split file.

    When to use:
        - You already trained a tabular model and have a split .pkl from
          split_dataset().
        - You want explanations for many compounds from train/val/test at once.
        - You want one reusable SHAP payload file for downstream visual tools.

    When not to use:
        - You only have ad-hoc chat SMILES strings: use
          explain_smiles_with_shap() instead.
        - The model is a GNN checkpoint (.pt/.pth): use
          explain_gnn_with_edgeshaper().

    Typical workflow:
        check_training() -> explain_with_shap() -> plot_shap_mol()
        Optional next steps:
        get_top_k_bit_environments_from_shap() and
        plot_top_k_parent_molecule_environments_from_shap().

    Behavior:
        - Loads model and split from disk.
        - Predicts on the selected split partition.
        - Optionally filters to correctly predicted instances only.
        - Computes SHAP values and writes a payload .pkl containing SHAP values,
          SMILES (if available), labels, and fingerprint metadata.

    Args:
        model_path: Path to a tabular sklearn-style model (.pkl/.joblib),
            usually from train_model()/check_training().
        split_file_path: Path to split .pkl from split_dataset(). Must contain
            train_features, which are always used as SHAP background.
        split: Which partition to explain: "test" (default), "train", or "val".
        n_bits: Fingerprint size metadata saved in output payload. Should match
            feature generation settings.
        correct_only: If True (default), explain only correct predictions.
            If False, explain all rows in the selected split.
        save_path: Optional output .pkl path. Defaults to
            <session>/results/<model_stem>_<split>_shap.pkl.

    Returns:
        JSON-serializable summary including shap_values_path and dataset stats.
        The saved shap_values_path is the primary input for follow-up SHAP
        visualization tools.
    """

    split_data, X_train, split_file_path = _load_split_data_with_required_background(split_file_path)
    X_all       = np.array(split_data[f"{split}_features"])
    y_all       = np.array(split_data[f"{split}_labels"])

    model        = joblib.load(model_path)
    y_pred       = model.predict(X_all)
    correct_mask = y_pred == y_all
    n_correct    = int(correct_mask.sum())

    if correct_only:
        if n_correct == 0:
            raise ValueError(
                "No correctly predicted instances found in the selected split. "
                "Try a different split, check model performance, or pass correct_only=False."
            )
        mask      = correct_mask
        X_explain = X_all[mask]
    else:
        mask      = np.ones(len(X_all), dtype=bool)
        X_explain = X_all

    # Predicted labels for the explain subset — used for per-class SHAP selection
    y_pred_explain = y_pred[mask]

    explainer                 = SHAPExplainer(model, background=X_train)
    shap_values               = explainer.explain_per_predicted_class(X_explain, y_pred_explain)
    expected_values_by_class  = explainer.expected_values
    expected_values_selected  = explainer.expected_values_for_predictions(y_pred_explain)
    expected_value_mode       = "predicted_class" if expected_values_by_class.size > 2 else "class_1_or_single"
    class_labels              = np.array(getattr(model, "classes_", np.arange(expected_values_by_class.size)))
    expected_val              = (
        float(expected_values_selected.mean())
        if expected_values_selected.size > 0
        else float(explainer.expected_value)
    )

    smiles_key = f"{split}_smiles"
    labels_key = f"{split}_labels"
    cid_key    = f"{split}_cid"

    save_dict: dict[str, Any] = {
        "shap_values":              shap_values,
        "expected_value":           expected_val,
        "expected_values_by_class": expected_values_by_class,
        "expected_values_selected": expected_values_selected,
        "expected_value_classes":   class_labels,
        "expected_value_mode":      expected_value_mode,
        "model_path":               model_path,
        "split_file_path":          split_file_path,
        "split":                    split,
        "n_bits":                   n_bits,
        "correct_only":             correct_only,
    }
    if smiles_key in split_data:
        save_dict["smiles"] = np.array(split_data[smiles_key])[mask]
    if labels_key in split_data:
        save_dict["labels"] = y_all[mask]
    if cid_key in split_data:
        save_dict["cid"] = np.array(split_data[cid_key])[mask]

    if save_path is None:
        out_dir   = _get_session_logger().session_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem      = Path(model_path).stem
        save_path = str(out_dir / f"{stem}_{split}_shap.pkl")
    else:
        save_path = _resolve_path(save_path)

    joblib.dump(save_dict, save_path)

    return {
        "split_file_path":          split_file_path,
        "shap_values_path":          save_path,
        "n_samples":                 int(shap_values.shape[0]),
        "n_samples_total":           int(X_all.shape[0]),
        "n_correct":                 n_correct,
        "correct_only":              correct_only,
        "n_features":                int(shap_values.shape[1]),
        "expected_value":            float(expected_val),
        "expected_values_by_class":  expected_values_by_class.tolist(),
        "expected_values_selected":  expected_values_selected.tolist(),
        "expected_value_classes":    class_labels.tolist(),
        "expected_value_mode":       expected_value_mode,
        "has_smiles":                smiles_key in split_data,
        "next_step": (
            f"Call plot_shap_mol('{save_path}') to visualise "
            "atom-level SHAP heatmaps for individual compounds."
        ),
    }

    
def explain_smiles_with_shap(
    model_path: str,
    smiles: list[str],
    split_file_path: str,
    method: str = "ECFP",
    n_bits: int = 2048,
    radius: int = 2,
    featurizer_kwargs: Optional[dict] = None,
    save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Compute SHAP values for ad-hoc SMILES inputs with a tabular model.

    When to use:
        - You have one or more SMILES from chat and want local explanations now.
        - You do not need to explain an entire split partition.

    When not to use:
        - You want explanations for many compounds already stored in a split:
          use explain_with_shap().
        - The model is a GNN checkpoint (.pt/.pth): use
          explain_gnn_with_edgeshaper().

    Typical workflow:
        check_training() -> explain_smiles_with_shap() -> plot_shap_mol()
        Optional next steps:
        get_top_k_bit_environments_from_shap() and
        plot_top_k_parent_molecule_environments_from_shap().

    Behavior:
        - Featurizes input SMILES using a selected registered fingerprint method.
        - Loads training background from split_file_path for SHAP baselines.
        - Predicts classes and computes SHAP values per predicted class.
        - Saves SHAP payload .pkl for downstream plotting.
        - Stored labels are model predictions (not ground-truth labels).

    Args:
        model_path: Path to tabular model (.pkl/.joblib).
        smiles: One or more SMILES to explain.
        split_file_path: Required split .pkl from split_dataset() used for
            SHAP background. This is mandatory.
        method: Featurizer name (default "ECFP"). Should match model training.
        n_bits: Desired fingerprint length metadata; overridden by split
            background feature width.
        radius: Morgan radius for ECFP-compatible featurizers.
        featurizer_kwargs: Extra keyword arguments forwarded to featurizer.
        save_path: Optional output .pkl path. Defaults to
            <session>/results/<model_stem>_smiles_shap.pkl.

    Returns:
        JSON-serializable summary including shap_values_path and prediction
        metadata. shap_values_path should be passed to downstream SHAP tools.
    """

    if not smiles:
        raise ValueError("smiles list must contain at least one SMILES string.")

    split_data, background, split_file_path = _load_split_data_with_required_background(split_file_path)
    n_bits = int(background.shape[1])

    featurizers = available_featurizers()
    if method not in featurizers:
        raise ValueError(
            f"Unknown featurizer {method!r}. "
            f"Available: {sorted(featurizers.keys())}. "
            "Call list_featurizers() for details."
        )
    fn = featurizers[method]

    #Build keyword args accepted by this featurizer
    sig = _inspect.signature(fn)
    base_kwargs = {k: v for k, v in {"n_bits": n_bits, "radius": radius}.items()
                   if k in sig.parameters}
    if featurizer_kwargs:
        base_kwargs.update(featurizer_kwargs)

    #Featurize
    X = np.array(fn(smiles, **base_kwargs))

    #Predict
    model  = joblib.load(model_path)
    y_pred = model.predict(X)

    #SHAP
    explainer                 = SHAPExplainer(model, background=background)
    shap_values               = explainer.explain_per_predicted_class(X, y_pred)
    expected_values_by_class  = explainer.expected_values
    expected_values_selected  = explainer.expected_values_for_predictions(y_pred)
    expected_value_mode       = "predicted_class" if expected_values_by_class.size > 2 else "class_1_or_single"
    class_labels              = np.array(getattr(model, "classes_", np.arange(expected_values_by_class.size)))
    expected_val              = (
        float(expected_values_selected.mean())
        if expected_values_selected.size > 0
        else float(explainer.expected_value)
    )

    #Save
    save_dict: dict[str, Any] = {
        "shap_values":              shap_values,
        "expected_value":           expected_val,
        "expected_values_by_class": expected_values_by_class,
        "expected_values_selected": expected_values_selected,
        "expected_value_classes":   class_labels,
        "expected_value_mode":      expected_value_mode,
        "model_path":               model_path,
        "smiles":                   np.array(smiles),
        "labels":                   y_pred,          # predicted class; no ground truth
        "method":                   method,
        "n_bits":                   base_kwargs.get("n_bits", n_bits),
        "radius":                   base_kwargs.get("radius", radius),
        "featurizer_kwargs":        base_kwargs,
        "source":                   "explain_smiles",
        "split_file_path":          split_file_path,
    }

    if save_path is None:
        out_dir   = _get_session_logger().session_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem      = Path(model_path).stem
        save_path = str(out_dir / f"{stem}_smiles_shap.pkl")
    else:
        save_path = _resolve_path(save_path)

    joblib.dump(save_dict, save_path)

    return {
        "split_file_path":          split_file_path,
        "shap_values_path":          save_path,
        "n_samples":                 int(shap_values.shape[0]),
        "n_features":                int(shap_values.shape[1]),
        "expected_value":            float(expected_val),
        "expected_values_by_class":  expected_values_by_class.tolist(),
        "expected_values_selected":  expected_values_selected.tolist(),
        "expected_value_classes":    class_labels.tolist(),
        "expected_value_mode":       expected_value_mode,
        "prediction":                y_pred.tolist()[0],
        "shap_sum":                  float(shap_values.sum()),
        "method":                    method,
        "has_smiles":                True,
        "note":                      "Labels in output file are model predictions, not ground truth.",
        "next_step": (
            f"Call plot_shap_mol('{save_path}') to render "
            "atom-level SHAP heatmaps. "
            "Labels shown reflect the model's predicted class."
        ),
    }


def plot_shap_mol(
    shap_values_path: str,
    sample_indices: Optional[list[int]] = None,
    mol_size: Optional[list[int]] = None,
    cmap: str = "coolwarm",
) -> list:
    """Render atom-level SHAP heatmaps from a saved SHAP payload.

    When to use:
        - After explain_with_shap() or explain_smiles_with_shap().
        - You want per-compound atom heatmaps for visual interpretation.

    When not to use:
        - You need top-k fragment environments or combined parent-molecule
          overlays. Use get_top_k_bit_environments_from_shap() and
          plot_top_k_parent_molecule_environments_from_shap().

    Input expectations:
        - shap_values_path must point to a SHAP payload .pkl containing SMILES.
        - Atom mapping is ECFP/Morgan-based via bit environment expansion.

    Output behavior:
        - Saves PNG images to <session>/plots/.
        - Returns a list with:
          1) summary dict
          2) one inline MCP Image object per rendered molecule.

    Args:
        shap_values_path: Path to .pkl from explain_with_shap() or
            explain_smiles_with_shap().
        sample_indices: 0-based rows to visualize. Defaults to first 5 rows.
        mol_size: [width, height] in pixels. Default (400, 300).
        cmap: Matplotlib colormap name.

    Returns:
        A mixed list [summary_dict, MCPImage, ...]. This is optimized for
        clients that support inline image objects.
    """

    data        = joblib.load(shap_values_path)
    shap_values = data["shap_values"]        # (n_samples, n_features)
    smiles_arr  = data.get("smiles")
    labels_arr  = data.get("labels")
    cid_arr     = data.get("cid")
    radius      = int(data.get("radius", 2))
    n_bits      = int(data.get("n_bits", 2048))

    if smiles_arr is None:
        raise ValueError(
            "No SMILES found in the SHAP values file. "
            "Re-run explain_with_shap() on a split that contains SMILES "
            "(load_dataset must be called with smiles_col set)."
        )

    n_samples = int(shap_values.shape[0])
    indices   = sample_indices if sample_indices is not None else list(range(min(5, n_samples)))
    size      = tuple(mol_size) if mol_size is not None else (400, 300)

    out_dir = _get_session_logger().session_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"generated": [], "molecules": {}}
    images: list = []

    for idx in indices:
        if idx < 0 or idx >= n_samples:
            continue
        smi = str(smiles_arr[idx])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        shap_vec     = shap_values[idx]
        bit_info     = get_ecfp_morgan_generator_bit_info(smi, radius=radius, n_bits=n_bits)
        atom_weights = shap_to_atom_weight(mol, bit_info, shap_vec)
        img          = get_atom_wise_weight_map(mol, atom_weights, mol_size=size, cmap=cmap)

        label    = str(labels_arr[idx]) if labels_arr is not None else "?"
        cid      = str(cid_arr[idx])    if cid_arr    is not None else str(idx)
        safe_cid = "".join(c if c.isalnum() or c in "-_" else "_" for c in cid)
        img_path = str(out_dir / f"shap_mol_{safe_cid}_label{label}.png")
        img.save(img_path)

        summary["generated"].append(img_path)
        summary["molecules"][str(idx)] = {
            "path":   img_path,
            "smiles": smi,
            "label":  label,
            "cid":    cid,
        }
        images.append(MCPImage(path=img_path))

    return [summary, *images]


def get_top_k_bit_environments_from_shap(
    shap_values_path: Optional[str] = None,
    sample_index: int = 0,
    top_k: int = 10,
    ranking: Literal["absolute", "positive", "negative"] = "absolute",
) -> dict[str, Any]:
    """Return top-k fingerprint-bit environments for one explained sample.

    When to use:
        - You want structured, text-friendly explanation data rather than images.
        - You need atom_indices/bond_indices for downstream custom rendering
          or rule analysis.

    When not to use:
        - You want a direct parent-molecule overlay image. Use
          plot_top_k_parent_molecule_environments_from_shap().

    Workflow:
        explain_with_shap()/explain_smiles_with_shap() -> this tool

    Args:
        shap_values_path: Optional SHAP payload .pkl path. If omitted, the
            newest available *_shap.pkl is auto-selected.
        sample_index: 0-based row index in payload.
        top_k: Number of bits to select.
        ranking: "absolute", "positive", or "negative".

    Returns:
        JSON-serializable dict with metadata and bit_environments containing
        per-bit contribution values and deduplicated fragment environments.
    """
    payload, resolved_path, shap_values = _load_shap_payload(shap_values_path)
    smiles, shap_vector, radius, n_bits = _get_shap_sample_context(
        payload,
        shap_values,
        sample_index,
    )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES at sample_index={sample_index}: {smiles!r}")

    bit_info = get_ecfp_morgan_generator_bit_info(smiles, radius=radius, n_bits=n_bits)
    top_env = get_top_k_bit_environments_with_contribution(
        mol=mol,
        dict_bit_info=bit_info,
        shapley_values=shap_vector,
        top_k=top_k,
        ranking=ranking,
    )

    n_env = int(sum(len(item.get("environments", [])) for item in top_env))
    return {
        "shap_values_path": resolved_path,
        "sample_index": int(sample_index),
        "smiles": smiles,
        "top_k": int(top_k),
        "ranking": ranking,
        "n_selected_bits": int(len(top_env)),
        "n_environments": n_env,
        "bit_environments": top_env,
    }


def plot_top_k_parent_molecule_environments_from_shap(
    shap_values_path: Optional[str] = None,
    sample_index: int = 0,
    top_k: int = 10,
    ranking: Literal["absolute", "positive", "negative"] = "absolute",
    max_environments: Optional[int] = 12,
    mols_per_row: int = 4,
    sub_img_size: Optional[list[int]] = None,
    output_path: Optional[str] = None,
) -> dict[str, Any]:
    """Render top-k SHAP fragment environments as parent-molecule overlays.

    When to use:
        - You already have a SHAP payload and want visual context for the most
          influential fragment environments on the full molecule.
        - You need a serialization-safe response that only returns metadata and
          file paths (no inline custom image objects).

    When not to use:
        - You only need structured environment data. Use
          get_top_k_bit_environments_from_shap().
        - You need atom heatmaps for full SHAP vectors. Use plot_shap_mol().

    Workflow:
        explain_with_shap()/explain_smiles_with_shap() -> this tool ->
        show_plot(image_path)

    Behavior:
        - Creates one tile per selected environment with parent-molecule
          highlighting.
        - Saves PNG to output_path or <session>/plots/topk_parent_env_sampleN.png.
        - Returns a JSON-serializable summary dict.

    Args:
        shap_values_path: Optional SHAP payload .pkl path. Auto-selects latest
            payload when omitted.
        sample_index: 0-based row index in payload.
        top_k: Number of bits to consider.
        ranking: "absolute", "positive", or "negative".
        max_environments: Optional cap on rendered environment tiles.
        mols_per_row: Grid width.
        sub_img_size: Optional [width, height] for each tile.
        output_path: Optional explicit PNG output path.

    Returns:
        JSON-serializable dict with image_path and rendering metadata.
    """
    payload, resolved_path, shap_values = _load_shap_payload(shap_values_path)
    smiles, shap_vector, radius, n_bits = _get_shap_sample_context(
        payload,
        shap_values,
        sample_index,
    )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES at sample_index={sample_index}: {smiles!r}")

    if sub_img_size is None:
        size = (320, 240)
    else:
        if len(sub_img_size) != 2:
            raise ValueError("sub_img_size must contain exactly two integers: [width, height].")
        size = (int(sub_img_size[0]), int(sub_img_size[1]))

    bit_info = get_ecfp_morgan_generator_bit_info(smiles, radius=radius, n_bits=n_bits)
    image = render_top_k_parent_molecule_environment_highlights(
        mol=mol,
        dict_bit_info=bit_info,
        shapley_values=shap_vector,
        top_k=top_k,
        ranking=ranking,
        max_environments=max_environments,
        mols_per_row=mols_per_row,
        sub_img_size=size,
    )

    if output_path is None:
        out_dir = _get_session_logger().session_dir / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"topk_parent_env_sample{sample_index}.png"
    else:
        out_path = Path(_resolve_path(output_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(image, "save"):
        image.save(out_path)
    elif isinstance(image, (bytes, bytearray)):
        out_path.write_bytes(bytes(image))
    else:
        raise TypeError("Rendered image output is not a supported saveable type.")

    summary = {
        "shap_values_path": resolved_path,
        "sample_index": int(sample_index),
        "smiles": smiles,
        "top_k": int(top_k),
        "ranking": ranking,
        "image_path": str(out_path),
        "next_step": "Use show_plot(image_path) to display this artifact in chat.",
    }
    return summary

