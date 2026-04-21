"""Unit tests for SHAPExplainer routing and value normalization logic."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

import chemagent.explainability.shap_explainer as shap_module
from chemagent.explainability.shap_explainer import SHAPExplainer


class _DummyExplainer:
    def __init__(self, shap_values: Any, expected_value: Any):
        self._shap_values = shap_values
        self.expected_value = expected_value

    def shap_values(self, X):
        _ = X
        return self._shap_values


class _PickleableToyClassifier:
    def __init__(self, preds: np.ndarray, classes: np.ndarray):
        self._preds = np.asarray(preds)
        self.classes_ = np.asarray(classes)

    def predict(self, X):
        n = len(X)
        return self._preds[:n]


class _FakeToolSHAPExplainer:
    expected_values_template = np.array([0.2, 0.8], dtype=float)
    init_backgrounds: list[np.ndarray | None] = []

    def __init__(self, model, background=None):
        self.model = model
        bg = None if background is None else np.asarray(background)
        type(self).init_backgrounds.append(bg)

    @classmethod
    def reset(cls):
        cls.expected_values_template = np.array([0.2, 0.8], dtype=float)
        cls.init_backgrounds = []

    def explain_per_predicted_class(self, X: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y_pred).astype(float)
        return np.tile(np.arange(X_arr.shape[1], dtype=float), (X_arr.shape[0], 1)) + y_arr[:, None]

    @property
    def expected_values(self) -> np.ndarray:
        return np.asarray(type(self).expected_values_template, dtype=float)

    @property
    def expected_value(self) -> float:
        ev = self.expected_values
        return float(ev[1]) if ev.size == 2 else float(ev[0])

    def expected_values_for_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        y_pred_arr = np.asarray(y_pred)
        ev = self.expected_values
        if ev.size <= 2:
            return np.full(y_pred_arr.shape[0], self.expected_value, dtype=float)

        classes = np.asarray(getattr(self.model, "classes_", np.arange(ev.size)))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx = np.array([class_to_idx[p] for p in y_pred_arr], dtype=int)
        return ev[idx]


def _build_test_explainer(shap_values: Any, expected_value: Any = 0.0) -> SHAPExplainer:
    s = SHAPExplainer.__new__(SHAPExplainer)
    s._is_dnn = False
    s.model = type("Model", (), {})()
    setattr(s, "_explainer", _DummyExplainer(shap_values=shap_values, expected_value=expected_value))
    return s


def test_sample_background_downsamples_to_requested_size():
    bg = np.arange(1000, dtype=np.float32).reshape(500, 2)
    sampled = SHAPExplainer._sample_background(bg, max_background=5)
    expected = np.asarray(shap_module.shap.sample(bg, nsamples=5, random_state=0))

    assert sampled.shape == (5, 2)
    assert np.array_equal(sampled, expected)


def test_sample_background_raises_for_non_2d_input():
    bg = np.ones((8,), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected 2D background features"):
        SHAPExplainer._sample_background(bg)


def test_build_explainer_routes_tree_models_to_treeexplainer(monkeypatch):
    captured = {}

    def _fake_tree_explainer(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return "tree-explainer"

    monkeypatch.setattr(shap_module.shap, "TreeExplainer", _fake_tree_explainer)

    model = type("RandomForestClassifier", (), {})()
    background = np.zeros((10, 4), dtype=np.float32)
    out = SHAPExplainer._build_explainer(model, background)

    assert out == "tree-explainer"
    assert captured["model"] is model
    assert captured["kwargs"]["feature_perturbation"] == "interventional"
    assert np.array_equal(captured["kwargs"]["data"], background)


def test_build_explainer_dnn_requires_background():
    class _FakeDNN:
        def get_torch_model(self):
            return object()

        def as_torch_tensor(self, X):
            return X

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    model = _FakeDNN()
    with pytest.raises(ValueError, match="background data is required for DNN SHAP explainers"):
        SHAPExplainer._build_explainer(model, background=None)


def test_build_explainer_dnn_uses_deepexplainer_with_tensor_background(monkeypatch):
    captured = {}

    def _fake_deep_explainer(torch_model, bg_tensor):
        captured["torch_model"] = torch_model
        captured["bg_tensor"] = bg_tensor
        return "deep-explainer"

    monkeypatch.setattr(shap_module.shap, "DeepExplainer", _fake_deep_explainer)

    class _FakeDNN:
        def __init__(self):
            self.seen_bg = None
            self.torch_model = object()

        def get_torch_model(self):
            return self.torch_model

        def as_torch_tensor(self, X):
            self.seen_bg = np.asarray(X)
            return ("tensor", np.asarray(X))

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    model = _FakeDNN()
    background = np.arange(1024, dtype=np.float32).reshape(256, 4)
    out = SHAPExplainer._build_explainer(model, background)

    assert out == "deep-explainer"
    assert captured["torch_model"] is model.torch_model
    assert isinstance(captured["bg_tensor"], tuple)
    assert captured["bg_tensor"][0] == "tensor"
    assert model.seen_bg is not None
    assert model.seen_bg.shape == (128, 4)


def test_build_explainer_kernel_prefers_predict_proba(monkeypatch):
    captured = {}

    def _fake_kernel_explainer(predict_fn, bg):
        captured["predict_fn"] = predict_fn
        captured["bg"] = bg
        return "kernel-explainer"

    monkeypatch.setattr(shap_module.shap, "KernelExplainer", _fake_kernel_explainer)

    class _FakeModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.zeros((len(X), 2), dtype=float)

    model = _FakeModel()
    model.__class__.__name__ = "SVC"
    background = np.zeros((10, 6), dtype=np.float32)
    out = SHAPExplainer._build_explainer(model, background)

    assert out == "kernel-explainer"
    assert captured["predict_fn"].__name__ == "predict_proba"
    assert captured["bg"].shape == (10, 6)


def test_build_explainer_kernel_uses_predict_when_proba_unavailable(monkeypatch):
    captured = {}

    def _fake_kernel_explainer(predict_fn, bg):
        captured["predict_fn"] = predict_fn
        captured["bg"] = bg
        return "kernel-explainer"

    monkeypatch.setattr(shap_module.shap, "KernelExplainer", _fake_kernel_explainer)

    class _FakeModel:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    model = _FakeModel()
    model.__class__.__name__ = "SVC"
    background = np.zeros((10, 6), dtype=np.float32)
    out = SHAPExplainer._build_explainer(model, background)

    assert out == "kernel-explainer"
    assert captured["predict_fn"].__name__ == "predict"


def test_explain_binary_list_output_returns_class_1_values():
    sv0 = np.zeros((3, 5), dtype=np.float32)
    sv1 = np.ones((3, 5), dtype=np.float32)
    s = _build_test_explainer(shap_values=[sv0, sv1])

    out = s.explain(np.zeros((3, 5), dtype=np.float32))

    assert out.shape == (3, 5)
    assert np.allclose(out, 1.0)


def test_explain_binary_3d_output_returns_class_1_slice():
    sv = np.zeros((2, 4, 2), dtype=np.float32)
    sv[..., 1] = 3.5
    s = _build_test_explainer(shap_values=sv)

    out = s.explain(np.zeros((2, 4), dtype=np.float32))

    assert out.shape == (2, 4)
    assert np.allclose(out, 3.5)


def test_explain_per_predicted_class_selects_class_specific_slice():
    sv = np.zeros((3, 4, 3), dtype=np.float32)
    sv[:, :, 0] = 10.0
    sv[:, :, 1] = 20.0
    sv[:, :, 2] = 30.0

    s = _build_test_explainer(shap_values=sv)
    s.model = type("Model", (), {"classes_": np.array(["a", "b", "c"])})()

    y_pred = np.array(["c", "a", "b"])
    out = s.explain_per_predicted_class(np.zeros((3, 4), dtype=np.float32), y_pred)

    assert out.shape == (3, 4)
    assert np.allclose(out[0], 30.0)
    assert np.allclose(out[1], 10.0)
    assert np.allclose(out[2], 20.0)


def test_expected_values_rejects_none_callable_and_empty():
    s_none = _build_test_explainer(shap_values=np.zeros((2, 2)), expected_value=None)
    with pytest.raises(TypeError, match="returned None"):
        _ = s_none.expected_values

    s_callable = _build_test_explainer(shap_values=np.zeros((2, 2)), expected_value=lambda: 1.0)
    with pytest.raises(TypeError, match="callable"):
        _ = s_callable.expected_values

    s_empty = _build_test_explainer(shap_values=np.zeros((2, 2)), expected_value=[])
    with pytest.raises(TypeError, match="empty expected_value"):
        _ = s_empty.expected_values


def test_expected_values_for_predictions_raises_on_unknown_class():
    s = _build_test_explainer(
        shap_values=np.zeros((3, 4, 3), dtype=np.float32),
        expected_value=np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )
    s.model = type("Model", (), {"classes_": np.array([0, 1, 2])})()

    with pytest.raises(ValueError, match="not found in model classes"):
        s.expected_values_for_predictions(np.array([3]))


def test_from_model_path_loads_model_and_forwards_background(monkeypatch):
    fake_model = object()
    captured = {}

    def _fake_load(path):
        captured["path"] = path
        return fake_model

    def _fake_build(model, background):
        captured["model"] = model
        captured["background"] = background
        return _DummyExplainer(shap_values=np.zeros((1, 1)), expected_value=0.0)

    monkeypatch.setattr(shap_module.joblib, "load", _fake_load)
    monkeypatch.setattr(SHAPExplainer, "_build_explainer", staticmethod(_fake_build))

    background = np.ones((4, 3), dtype=np.float32)
    out = SHAPExplainer.from_model_path("dummy_model.pkl", background=background)

    assert isinstance(out, SHAPExplainer)
    assert captured["path"] == "dummy_model.pkl"
    assert captured["model"] is fake_model
    assert np.array_equal(captured["background"], background)


def test_explain_with_shap_integration_filters_correct_and_saves(tmp_path: Path, monkeypatch):
    split_path = tmp_path / "split.pkl"
    model_path = tmp_path / "model.pkl"
    out_path = tmp_path / "out_shap.pkl"

    split_data = {
        "train_features": np.array([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], dtype=float),
        "train_labels": np.array([0, 1], dtype=int),
        "test_features": np.array([[0.2, 0.0, 0.3], [0.8, 1.1, 0.1], [1.4, 0.5, 0.2]], dtype=float),
        "test_labels": np.array([1, 0, 1], dtype=int),
        "test_smiles": np.array(["CCO", "CCN", "CCC"]),
        "test_cid": np.array(["cid-1", "cid-2", "cid-3"]),
    }
    model = _PickleableToyClassifier(preds=np.array([1, 1, 1]), classes=np.array([0, 1]))
    joblib.dump(split_data, split_path)
    joblib.dump(model, model_path)

    _FakeToolSHAPExplainer.reset()
    monkeypatch.setattr(shap_module, "SHAPExplainer", _FakeToolSHAPExplainer)

    result = shap_module.explain_with_shap(
        model_path=str(model_path),
        split_file_path=str(split_path),
        split="test",
        correct_only=True,
        save_path=str(out_path),
    )

    assert result["shap_values_path"] == str(out_path)
    assert result["n_samples"] == 2
    assert result["n_samples_total"] == 3
    assert result["n_correct"] == 2
    assert result["n_features"] == 3
    assert result["expected_value_mode"] == "class_1_or_single"
    assert result["expected_values_by_class"] == [0.2, 0.8]
    assert result["expected_values_selected"] == [0.8, 0.8]
    assert result["expected_value_classes"] == [0, 1]
    assert result["has_smiles"] is True
    assert out_path.exists()

    payload = joblib.load(out_path)
    assert payload["split"] == "test"
    assert payload["correct_only"] is True
    assert payload["shap_values"].shape == (2, 3)
    assert payload["labels"].tolist() == [1, 1]
    assert payload["smiles"].tolist() == ["CCO", "CCC"]
    assert payload["cid"].tolist() == ["cid-1", "cid-3"]
    assert len(_FakeToolSHAPExplainer.init_backgrounds) == 1
    assert _FakeToolSHAPExplainer.init_backgrounds[0] is not None
    assert _FakeToolSHAPExplainer.init_backgrounds[0].shape == (2, 3)


def test_explain_with_shap_raises_when_no_correct_predictions(tmp_path: Path):
    split_path = tmp_path / "split.pkl"
    model_path = tmp_path / "model.pkl"

    split_data = {
        "train_features": np.array([[0.0, 0.1]], dtype=float),
        "train_labels": np.array([0], dtype=int),
        "test_features": np.array([[0.2, 0.3], [0.4, 0.5]], dtype=float),
        "test_labels": np.array([0, 0], dtype=int),
    }
    model = _PickleableToyClassifier(preds=np.array([1, 1]), classes=np.array([0, 1]))
    joblib.dump(split_data, split_path)
    joblib.dump(model, model_path)

    with pytest.raises(ValueError, match="No correctly predicted instances found"):
        shap_module.explain_with_shap(
            model_path=str(model_path),
            split_file_path=str(split_path),
            split="test",
            correct_only=True,
        )


def test_explain_smiles_with_shap_requires_split_file_path():
    with pytest.raises(ValueError, match="split_file_path is required"):
        shap_module.explain_smiles_with_shap(
            model_path="unused.pkl",
            smiles=["CCO"],
            method="ECFP",
            split_file_path="",
        )


def test_explain_smiles_with_shap_integration_uses_split_background_and_saves(
    tmp_path: Path,
    monkeypatch,
):
    split_path = tmp_path / "split.pkl"
    model_path = tmp_path / "model.pkl"
    out_path = tmp_path / "smiles_out.pkl"

    split_data = {
        "train_features": np.array(
            [
                [0, 1, 2, 3, 4, 5, 6],
                [1, 2, 3, 4, 5, 6, 7],
                [2, 3, 4, 5, 6, 7, 8],
            ],
            dtype=float,
        )
    }
    model = _PickleableToyClassifier(preds=np.array([2, 1]), classes=np.array([0, 1, 2]))
    joblib.dump(split_data, split_path)
    joblib.dump(model, model_path)

    featurizer_calls: dict[str, Any] = {}

    def _fake_featurizer(smiles, n_bits=2048, radius=2):
        featurizer_calls["smiles"] = list(smiles)
        featurizer_calls["n_bits"] = n_bits
        featurizer_calls["radius"] = radius
        rows = len(smiles)
        return np.tile(np.arange(n_bits, dtype=float), (rows, 1))

    monkeypatch.setattr(
        shap_module,
        "available_featurizers",
        lambda: {"ECFP": _fake_featurizer},
    )

    _FakeToolSHAPExplainer.reset()
    _FakeToolSHAPExplainer.expected_values_template = np.array([13.0, -4.0, -12.0], dtype=float)
    monkeypatch.setattr(shap_module, "SHAPExplainer", _FakeToolSHAPExplainer)

    result = shap_module.explain_smiles_with_shap(
        model_path=str(model_path),
        smiles=["CCO", "CCN"],
        method="ECFP",
        split_file_path=str(split_path),
        n_bits=2048,
        radius=3,
        save_path=str(out_path),
    )

    assert result["shap_values_path"] == str(out_path)
    assert result["n_samples"] == 2
    assert result["n_features"] == 7
    assert result["expected_value_mode"] == "predicted_class"
    assert result["expected_values_by_class"] == [13.0, -4.0, -12.0]
    assert result["expected_values_selected"] == [-12.0, -4.0]
    assert result["expected_value_classes"] == [0, 1, 2]
    assert result["prediction"] == 2
    assert result["method"] == "ECFP"
    assert result["has_smiles"] is True
    assert out_path.exists()

    assert featurizer_calls["smiles"] == ["CCO", "CCN"]
    assert featurizer_calls["n_bits"] == 7
    assert featurizer_calls["radius"] == 3

    payload = joblib.load(out_path)
    assert payload["source"] == "explain_smiles"
    assert payload["split_file_path"] == str(split_path)
    assert payload["n_bits"] == 7
    assert payload["radius"] == 3
    assert payload["labels"].tolist() == [2, 1]
    assert payload["smiles"].tolist() == ["CCO", "CCN"]
    assert len(_FakeToolSHAPExplainer.init_backgrounds) == 1
    assert _FakeToolSHAPExplainer.init_backgrounds[0] is not None
    assert _FakeToolSHAPExplainer.init_backgrounds[0].shape == (3, 7)


def _write_minimal_shap_payload(tmp_path: Path, smiles: str = "CCO", n_bits: int = 16) -> Path:
    payload_path = tmp_path / "sample_shap.pkl"
    payload = {
        "shap_values": np.linspace(-0.4, 0.6, n_bits, dtype=float).reshape(1, -1),
        "smiles": np.array([smiles]),
        "radius": 2,
        "n_bits": n_bits,
    }
    joblib.dump(payload, payload_path)
    return payload_path


def test_get_top_k_bit_environments_from_shap_returns_expected_structure(tmp_path: Path, monkeypatch):
    shap_path = _write_minimal_shap_payload(tmp_path, smiles="CCO", n_bits=16)

    monkeypatch.setattr(
        shap_module,
        "get_ecfp_morgan_generator_bit_info",
        lambda smiles, radius, n_bits: {1: [(0, 0)]},
    )
    monkeypatch.setattr(
        shap_module,
        "get_top_k_bit_environments_with_contribution",
        lambda mol, dict_bit_info, shapley_values, top_k, ranking: [
            {
                "bit": 1,
                "contribution": 0.5,
                "environments": [
                    {
                        "center_atom": 0,
                        "radius": 0,
                        "atom_indices": [0],
                        "bond_indices": [],
                        "environment_smiles": "C",
                    }
                ],
            }
        ],
    )

    result = shap_module.get_top_k_bit_environments_from_shap(
        shap_values_path=str(shap_path),
        sample_index=0,
        top_k=5,
        ranking="absolute",
    )

    assert result["sample_index"] == 0
    assert result["smiles"] == "CCO"
    assert result["n_selected_bits"] == 1
    assert result["n_environments"] == 1
    assert result["bit_environments"][0]["bit"] == 1


def test_plot_top_k_parent_molecule_environments_from_shap_saves_image(tmp_path: Path, monkeypatch):
    from PIL import Image as PILImage

    shap_path = _write_minimal_shap_payload(tmp_path, smiles="CCO", n_bits=16)

    monkeypatch.setattr(
        shap_module,
        "get_ecfp_morgan_generator_bit_info",
        lambda smiles, radius, n_bits: {1: [(0, 0)]},
    )
    monkeypatch.setattr(
        shap_module,
        "render_top_k_parent_molecule_environment_highlights",
        lambda **kwargs: PILImage.new("RGB", (20, 20), color="white"),
    )

    out = shap_module.plot_top_k_parent_molecule_environments_from_shap(
        shap_values_path=str(shap_path),
        sample_index=0,
        top_k=3,
        ranking="absolute",
    )

    assert isinstance(out, dict)
    assert Path(out["image_path"]).exists()
    # Ensure MCP can serialize this payload without custom-object handling.
    json.dumps(out)


def test_get_top_k_bit_environments_from_shap_uses_latest_payload_when_path_missing(
    tmp_path: Path,
    monkeypatch,
):
    session_dir = tmp_path / "session"
    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    latest_payload = results_dir / "latest_shap.pkl"
    joblib.dump(
        {
            "shap_values": np.linspace(-0.2, 0.8, 16, dtype=float).reshape(1, -1),
            "smiles": np.array(["CCO"]),
            "radius": 2,
            "n_bits": 16,
        },
        latest_payload,
    )

    monkeypatch.setattr(
        shap_module,
        "_get_session_logger",
        lambda: type("_Logger", (), {"session_dir": session_dir})(),
    )
    monkeypatch.setattr(
        shap_module,
        "get_ecfp_morgan_generator_bit_info",
        lambda smiles, radius, n_bits: {1: [(0, 0)]},
    )
    monkeypatch.setattr(
        shap_module,
        "get_top_k_bit_environments_with_contribution",
        lambda mol, dict_bit_info, shapley_values, top_k, ranking: [
            {
                "bit": 1,
                "contribution": 0.5,
                "environments": [
                    {
                        "center_atom": 0,
                        "radius": 0,
                        "atom_indices": [0],
                        "bond_indices": [],
                        "environment_smiles": "C",
                    }
                ],
            }
        ],
    )

    result = shap_module.get_top_k_bit_environments_from_shap(
        sample_index=0,
        top_k=5,
        ranking="absolute",
    )

    assert result["sample_index"] == 0
    assert result["smiles"] == "CCO"
    assert result["shap_values_path"] == str(latest_payload)
