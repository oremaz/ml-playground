"""Tests for SklearnModelWrapper base class and concrete wrappers."""
import numpy as np
import pytest
import torch
from pipelines_torch.models import (
    SklearnModelWrapper,
    SklearnRandomForestClassifierWrapper,
    SklearnRandomForestRegressorWrapper,
    XGBoostClassifierWrapper,
    XGBoostRegressorWrapper,
    LightGBMClassifierWrapper,
    LightGBMRegressorWrapper,
)


class TestSklearnModelWrapperBase:
    def test_to_returns_self(self):
        wrapper = SklearnRandomForestClassifierWrapper(n_estimators=5)
        assert wrapper.to("cpu") is wrapper
        assert wrapper.to("cuda") is wrapper

    def test_eval_and_train_are_noops(self):
        wrapper = SklearnRandomForestClassifierWrapper(n_estimators=5)
        # Should not raise
        wrapper.eval()
        wrapper.train()

    def test_to_numpy_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = SklearnModelWrapper._to_numpy(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_to_numpy_ndarray(self):
        a = np.array([1.0, 2.0, 3.0])
        result = SklearnModelWrapper._to_numpy(a)
        assert result is a  # Should return same object


class TestRandomForestClassifier:
    def test_fit_predict(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        wrapper = SklearnRandomForestClassifierWrapper(n_estimators=5, random_state=42)
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert len(preds) == len(y)

    def test_predict_proba(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        wrapper = SklearnRandomForestClassifierWrapper(n_estimators=5, random_state=42)
        wrapper.fit(X, y)
        probs = wrapper.predict_proba(X)
        assert probs.shape == (len(y), 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_call_returns_tensor(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        wrapper = SklearnRandomForestClassifierWrapper(n_estimators=5, random_state=42)
        wrapper.fit(X, y)
        result = wrapper(X)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (len(y), 2)

    def test_fit_with_tensor_input(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        wrapper = SklearnRandomForestClassifierWrapper(n_estimators=5, random_state=42)
        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)
        wrapper.fit(X_tensor, y_tensor)
        preds = wrapper.predict(X_tensor)
        assert len(preds) == len(y)

    def test_fit_with_sample_weight(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        wrapper = SklearnRandomForestClassifierWrapper(n_estimators=5, random_state=42)
        weights = np.ones(len(y))
        wrapper.fit(X, y, sample_weight=weights)
        preds = wrapper.predict(X)
        assert len(preds) == len(y)


class TestRandomForestRegressor:
    def test_fit_predict(self, synthetic_regression_data):
        X, y = synthetic_regression_data
        wrapper = SklearnRandomForestRegressorWrapper(n_estimators=5, random_state=42)
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert len(preds) == len(y)


class TestXGBoostWrappers:
    def test_classifier(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        wrapper = XGBoostClassifierWrapper(n_estimators=5, random_state=42, verbosity=0)
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert len(preds) == len(y)

    def test_regressor(self, synthetic_regression_data):
        X, y = synthetic_regression_data
        wrapper = XGBoostRegressorWrapper(n_estimators=5, random_state=42, verbosity=0)
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert len(preds) == len(y)


class TestLightGBMWrappers:
    def test_classifier(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        wrapper = LightGBMClassifierWrapper(n_estimators=5, random_state=42, verbose=-1)
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert len(preds) == len(y)

    def test_regressor(self, synthetic_regression_data):
        X, y = synthetic_regression_data
        wrapper = LightGBMRegressorWrapper(n_estimators=5, random_state=42, verbose=-1)
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert len(preds) == len(y)

    def test_regressor_returns_tensor(self, synthetic_regression_data):
        X, y = synthetic_regression_data
        wrapper = LightGBMRegressorWrapper(n_estimators=5, random_state=42, verbose=-1)
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert isinstance(preds, torch.Tensor)
