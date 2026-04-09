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
from mcp.server.fastmcp import Image
from rdkit import Chem

from chemagent.datasets.featurizer import available_featurizers
from chemagent.explainability.mol_shap_draw import (
    get_atom_wise_weight_map,
    get_ecfp_morgan_generator_bit_info,
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

    Parameters
    ----------
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

        Parameters
        ----------
        X:
            Feature matrix, shape ``(n_samples, n_features)``.
        y_pred:
            Predicted class labels, shape ``(n_samples,)``.

        Returns
        -------
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

        Parameters
        ----------
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
        idx = np.linspace(0, n_samples - 1, num=max_background, dtype=int)
        return bg[idx]

    @staticmethod
    def _build_explainer(model, background,):
        model_name = type(model).__name__
        if model_name in _TREE_MODEL_NAMES:
            return shap.TreeExplainer(model)

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


def explain_with_shap(
    model_path: str,
    split_file_path: str,
    split: Literal["train", "val", "test"] = "test",
    n_bits: int = 2048,
    correct_only: bool = True,
    save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Compute per-compound, per-feature SHAP values for a trained tabular model.

    Loads the model and split from disk, predicts on the chosen partition,
    optionally filters to correctly predicted instances, computes SHAP values,
    and saves the results (SHAP matrix, SMILES, labels, fingerprint params)
    to a .pkl file for downstream visualisation.

    By default only correctly predicted instances are explained (correct_only=True).
    Pass correct_only=False to explain all instances.

    Workflow: check_training → THIS TOOL → plot_shap_mol

    Routing note:
        Use this only for tabular sklearn models (.pkl/.joblib) trained on
        fingerprint features. If the model is a GNN checkpoint (.pt/.pth),
        use explain_gnn_with_edgeshaper instead.

    Args:
        model_path: Path to .pkl model from train_model() / check_training().
        split_file_path: Path to the .pkl split file from split_dataset().
        split: Partition to explain — "test" (default), "train", or "val".
        n_bits: Bit vector size used when computing ECFP features (default 2048).
                Must match the n_bits passed to compute_features().
        correct_only: If True (default), restrict SHAP computation to correctly
                      predicted instances. If False, explain all instances.
        save_path: Output .pkl path. Defaults to <session>/results/<stem>_<split>_shap.pkl.

    Returns:
        shap_values_path, n_samples, n_samples_total, n_correct, correct_only,
        n_features, expected_value, mean_abs_shap, has_smiles, next_step.
    """

    split_data  = joblib.load(split_file_path)
    X_all       = np.array(split_data[f"{split}_features"])
    y_all       = np.array(split_data[f"{split}_labels"])
    X_train     = np.array(split_data["train_features"])  # background for KernelExplainer

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
        "mean_abs_shap":             float(np.abs(shap_values).mean()),
        "has_smiles":                smiles_key in split_data,
        "next_step": (
            f"Call plot_shap_mol('{save_path}') to visualise "
            "atom-level SHAP heatmaps for individual compounds."
        ),
    }

    
def explain_smiles_with_shap(
    model_path: str,
    smiles: list[str],
    method: str = "ECFP",
    split_file_path: Optional[str] = None,
    n_bits: int = 2048,
    radius: int = 2,
    featurizer_kwargs: Optional[dict] = None,
    save_path: Optional[str] = None,
) -> dict[str, Any]:
    """Compute SHAP values for one or more SMILES strings with a tabular model.

    Use this when you have a SMILES string from the chat UI and want to understand
    the model's prediction — no labelled split file or ground-truth label needed.

    Featurizes each SMILES with any registered fingerprint method (default ECFP),
    runs model prediction, and computes per-feature SHAP values. Saves a .pkl that
    is directly usable by plot_shap_mol() for atom-level heatmap visualisation
    (ECFP/Morgan only; other methods produce SHAP values but atom mapping is skipped).

    The labels stored in the output file are the model's own predictions
    (class 0 or 1), not ground-truth labels — filenames/summaries will reflect
    the predicted class.

    Workflow: check_training → THIS TOOL → plot_shap_mol

    Routing note:
        This tool is for tabular sklearn models (.pkl/.joblib). For GNN
        prediction explanations, use explain_gnn_with_edgeshaper.

    Args:
        model_path: Path to .pkl model from train_model() / check_training().
        smiles: List of one or more SMILES strings to explain.
                Single compound example: ["CC(=O)Oc1ccccc1C(=O)O"]
        method: Featurization method (default "ECFP"). Call list_featurizers() to
                see all available methods. Must match the method used to train the model.
        split_file_path: Optional path to a split .pkl from split_dataset(). When
            provided the training features are used as the SHAP background (required
            for non-tree models such as SVC). The feature dimension of the split
            overrides n_bits automatically.
        n_bits: Fingerprint bit-vector size (default 2048). Ignored when
            split_file_path is provided (inferred from split dimensions).
        radius: Morgan radius for ECFP (default 2 = ECFP4). Ignored by methods
            that do not accept a radius parameter.
        featurizer_kwargs: Additional method-specific keyword arguments forwarded
            to the featurizer, e.g. {"min_path": 1} for RDKitFP. n_bits and radius
            are merged in automatically and can be overridden here.
        save_path: Output .pkl path.
            Defaults to <session>/results/<model_stem>_smiles_shap.pkl.

    Returns:
        shap_values_path, n_samples, n_features, expected_value,
        predictions, mean_abs_shap, shap_sum, method, has_smiles, next_step.
    """

    if not smiles:
        raise ValueError("smiles list must contain at least one SMILES string.")

    featurizers = available_featurizers()
    if method not in featurizers:
        raise ValueError(
            f"Unknown featurizer {method!r}. "
            f"Available: {sorted(featurizers.keys())}. "
            "Call list_featurizers() for details."
        )
    fn = featurizers[method]

    #Background / infer n_bits from split file
    background: Optional[np.ndarray] = None
    if split_file_path is not None:
        split_data = joblib.load(split_file_path)
        background = np.array(split_data["train_features"])
        n_bits     = int(background.shape[1])  # authoritative source

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
    }
    if split_file_path is not None:
        save_dict["split_file_path"] = split_file_path

    if save_path is None:
        out_dir   = _get_session_logger().session_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem      = Path(model_path).stem
        save_path = str(out_dir / f"{stem}_smiles_shap.pkl")
    else:
        save_path = _resolve_path(save_path)

    joblib.dump(save_dict, save_path)

    return {
        "shap_values_path":          save_path,
        "n_samples":                 int(shap_values.shape[0]),
        "n_features":                int(shap_values.shape[1]),
        "expected_value":            float(expected_val),
        "expected_values_by_class":  expected_values_by_class.tolist(),
        "expected_values_selected":  expected_values_selected.tolist(),
        "expected_value_classes":    class_labels.tolist(),
        "expected_value_mode":       expected_value_mode,
        "prediction":                y_pred.tolist()[0],
        "mean_abs_shap":             float(np.abs(shap_values).mean()),
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
    """Render atom-level SHAP heatmaps on molecular structures.

    Reads SHAP values and SMILES from the .pkl produced by explain_with_shap()
    or explain_smiles(), maps per-bit SHAP values onto atom positions using a
    Gaussian kernel, and returns one heatmap image per requested compound.

    Workflow: explain_with_shap / explain_smiles → THIS TOOL

    Each image is also saved to <session>/plots/ and returned inline so it
    renders directly in MCP-compatible chat interfaces.

    Args:
        shap_values_path: Path to the .pkl produced by explain_with_shap().
        sample_indices: Compound indices to visualise (0-based within the split).
                        Defaults to the first 5 compounds.
        mol_size: Image dimensions [width, height] in pixels (default [400, 300]).
        cmap: Matplotlib colormap name (default "coolwarm").

    Returns:
        List starting with a summary dict (index → path/smiles/label),
        followed by inline Image objects that render directly in the chat UI.
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
        images.append(Image(path=img_path))

    return [summary, *images]
