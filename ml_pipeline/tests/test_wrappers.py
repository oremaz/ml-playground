"""Tests for SklearnModelWrapper base class and concrete wrappers."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pipelines_torch.models import (
    HuggingFaceQLoRAWrapper,
    SklearnModelWrapper,
    SklearnRandomForestClassifierWrapper,
    SklearnRandomForestRegressorWrapper,
    XGBoostClassifierWrapper,
    XGBoostRegressorWrapper,
    LightGBMClassifierWrapper,
    LightGBMRegressorWrapper,
    TabFMClassifierWrapper,
    TabFMRegressorWrapper,
)


class TestHuggingFaceQLoRAWrapper:
    def test_gemma4_classifier_uses_backbone_hidden_states(self, monkeypatch):
        class DummyBackbone(torch.nn.Module):
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                batch_size, sequence_length = input_ids.shape
                hidden = torch.arange(
                    batch_size * sequence_length * 4, dtype=torch.float32
                ).reshape(batch_size, sequence_length, 4)
                return SimpleNamespace(
                    last_hidden_state=hidden,
                    hidden_states=(hidden,),
                )

        class DummyGemma4(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.model = DummyBackbone()
                self.config = SimpleNamespace()

            @property
            def device(self):
                return self.anchor.device

        dummy_model = DummyGemma4()
        monkeypatch.setattr(
            "transformers.AutoModelForMultimodalLM.from_pretrained",
            lambda *args, **kwargs: dummy_model,
        )
        config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=4), initializer_range=0.02
        )

        model = HuggingFaceQLoRAWrapper._load_gemma4_sequence_classifier(
            "dummy/gemma-4",
            config,
            {"num_labels": 2, "ignore_mismatched_sizes": True},
            num_labels=2,
            task_type="classification",
            torch_dtype=torch.float32,
        )
        output = model(
            input_ids=torch.ones((2, 3), dtype=torch.long),
            attention_mask=torch.ones((2, 3), dtype=torch.long),
            labels=torch.tensor([0, 1]),
        )

        assert output.logits.shape == (2, 2)
        assert output.loss.ndim == 0
        assert output.hidden_states[0].shape == (2, 3, 4)
        assert model.config.problem_type == "single_label_classification"

    @pytest.mark.parametrize("model_type", ["lfm2", "nanbeige"])
    def test_custom_causal_classifier_uses_backbone_hidden_states(
        self, monkeypatch, model_type
    ):
        class DummyBackbone(torch.nn.Module):
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                hidden = torch.ones((*input_ids.shape, 4), dtype=torch.float32)
                return SimpleNamespace(
                    last_hidden_state=hidden,
                    hidden_states=(hidden,),
                )

        class DummyCausalLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.model = DummyBackbone()
                self.config = SimpleNamespace(model_type=model_type)

            @property
            def device(self):
                return self.anchor.device

        monkeypatch.setattr(
            "transformers.AutoModelForCausalLM.from_pretrained",
            lambda *args, **kwargs: DummyCausalLM(),
        )
        config = SimpleNamespace(hidden_size=4, initializer_range=0.02)

        model = HuggingFaceQLoRAWrapper._load_causal_sequence_classifier(
            f"dummy/{model_type}", config, {}, 2, "classification", torch.float32
        )
        output = model(
            input_ids=torch.ones((2, 3), dtype=torch.long),
            attention_mask=torch.ones((2, 3), dtype=torch.long),
            labels=torch.tensor([0, 1]),
        )

        assert output.logits.shape == (2, 2)
        assert output.loss.ndim == 0


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

    @pytest.mark.parametrize(
        "wrapper_cls, kwargs",
        [
            (SklearnRandomForestClassifierWrapper, {"n_estimators": 5, "random_state": 42}),
            (XGBoostClassifierWrapper, {"n_estimators": 5, "random_state": 42, "verbosity": 0}),
            (LightGBMClassifierWrapper, {"n_estimators": 5, "random_state": 42, "verbose": -1}),
        ],
    )
    def test_pipeline_only_kwargs_are_ignored(self, wrapper_cls, kwargs):
        wrapper = wrapper_cls(input_dim=50_000, num_classes=2, **kwargs)
        params = wrapper.model.get_params()
        assert "input_dim" not in params
        assert "num_classes" not in params


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


class TestTabFMWrappers:
    def test_classifier_ignores_unsupported_kwargs(self, monkeypatch):
        from tabfm import tabfm_v1_0_0_pytorch

        backbone = object()
        monkeypatch.setattr(tabfm_v1_0_0_pytorch, "load", lambda **kwargs: backbone)
        wrapper = TabFMClassifierWrapper(
            device="cpu", n_estimators=2, cache_context=True
        )

        assert wrapper.model.model is backbone
        assert wrapper.model.n_estimators == 2

    def test_regressor_ignores_unsupported_kwargs(self, monkeypatch):
        from tabfm import tabfm_v1_0_0_pytorch

        backbone = object()
        monkeypatch.setattr(tabfm_v1_0_0_pytorch, "load", lambda **kwargs: backbone)
        wrapper = TabFMRegressorWrapper(
            device="cpu", n_estimators=2, cache_context=True
        )

        assert wrapper.model.model is backbone
        assert wrapper.model.n_estimators == 2
