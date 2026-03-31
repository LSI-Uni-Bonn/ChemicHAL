"""Tests for DNN integration in SHAPExplainer."""

from typing import Any

import numpy as np
import pytest

from chemagent.explainability.shap_explainer import SHAPExplainer
from chemagent.ml.dnn_model import DNNClassifier, DNNRegressor


def _make_binary_data(n_train: int = 32, n_test: int = 8, n_features: int = 12):
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(n_train, n_features)).astype(np.float32)
    X_test = rng.normal(size=(n_test, n_features)).astype(np.float32)
    y_train = (X_train[:, 0] + 0.5 * X_train[:, 1] > 0.0).astype(int)
    return X_train, y_train, X_test


def _make_regression_data(n_train: int = 32, n_test: int = 8, n_features: int = 12):
    rng = np.random.default_rng(11)
    X_train = rng.normal(size=(n_train, n_features)).astype(np.float32)
    X_test = rng.normal(size=(n_test, n_features)).astype(np.float32)
    y_train = (0.7 * X_train[:, 0] - 0.2 * X_train[:, 1] + 0.1).astype(np.float32)
    return X_train, y_train, X_test


def test_dnn_classifier_shap_smoke_returns_2d_values():
    X_train, y_train, X_test = _make_binary_data()
    model = DNNClassifier(
        hidden_size=16,
        n_hidden_layers=1,
        epochs=2,
        batch_size=8,
        random_seed=0,
        verbose=False,
    )
    model.fit(X_train, y_train)

    explainer = SHAPExplainer(model, background=X_train)
    y_pred = model.predict(X_test)
    shap_values = explainer.explain_per_predicted_class(X_test, y_pred)

    assert shap_values.shape == (X_test.shape[0], X_test.shape[1])
    assert np.isfinite(shap_values).all()


def test_dnn_regressor_shap_smoke_returns_2d_values():
    X_train, y_train, X_test = _make_regression_data()
    model = DNNRegressor(
        hidden_size=16,
        n_hidden_layers=1,
        epochs=2,
        batch_size=8,
        random_seed=0,
        verbose=False,
    )
    model.fit(X_train, y_train)

    explainer = SHAPExplainer(model, background=X_train)
    shap_values = explainer.explain(X_test)

    assert shap_values.shape == (X_test.shape[0], X_test.shape[1])
    assert np.isfinite(shap_values).all()


def test_dnn_explainer_requires_background():
    X_train, y_train, _ = _make_binary_data()
    model = DNNClassifier(epochs=1, random_seed=0, verbose=False)
    model.fit(X_train, y_train)

    with pytest.raises(ValueError, match="background data is required for DNN SHAP explainers"):
        SHAPExplainer(model, background=None)


class _DummyExplainer:
    def __init__(self, shap_values, expected_value):
        self._shap_values = shap_values
        self.expected_value = expected_value

    def shap_values(self, X):
        _ = X
        return self._shap_values


class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


def test_expected_value_tensor_like_is_normalized_to_float():
    s: Any = SHAPExplainer.__new__(SHAPExplainer)
    s._explainer = _DummyExplainer(shap_values=np.zeros((2, 3)), expected_value=_FakeTensor([0.2, 0.8]))

    assert isinstance(s.expected_value, float)
    assert s.expected_value == pytest.approx(0.8)


def test_expected_values_for_predictions_multiclass_uses_predicted_class_baseline():
    s: Any = SHAPExplainer.__new__(SHAPExplainer)
    s.model = type("M", (), {"classes_": np.array([0, 1, 2])})()
    s._explainer = _DummyExplainer(
        shap_values=np.zeros((4, 6, 3), dtype=np.float32),
        expected_value=np.array([13.0, -4.0, -12.0], dtype=np.float32),
    )

    y_pred = np.array([2, 0, 1, 2])
    out = s.expected_values_for_predictions(y_pred)

    assert out.shape == (4,)
    assert np.allclose(out, np.array([-12.0, 13.0, -4.0, -12.0]))


def test_expected_values_for_predictions_binary_uses_constant_class1_baseline():
    s: Any = SHAPExplainer.__new__(SHAPExplainer)
    s.model = type("M", (), {"classes_": np.array([0, 1])})()
    s._explainer = _DummyExplainer(
        shap_values=np.zeros((3, 6), dtype=np.float32),
        expected_value=np.array([0.2, 0.8], dtype=np.float32),
    )

    y_pred = np.array([0, 1, 0])
    out = s.expected_values_for_predictions(y_pred)

    assert out.shape == (3,)
    assert np.allclose(out, np.array([0.8, 0.8, 0.8]))


def test_explain_per_predicted_class_handles_single_output_3d():
    s: Any = SHAPExplainer.__new__(SHAPExplainer)
    s.model = type("M", (), {"classes_": np.array([0, 1])})()
    s._is_dnn = False
    s._explainer = _DummyExplainer(
        shap_values=np.ones((4, 6, 1), dtype=np.float32),
        expected_value=0.0,
    )

    y_pred = np.array([0, 1, 0, 1])
    out = s.explain_per_predicted_class(np.zeros((4, 6), dtype=np.float32), y_pred)

    assert out.shape == (4, 6)
    assert np.allclose(out, 1.0)
