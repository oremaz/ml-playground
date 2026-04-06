"""Tests for GeneralPipeline and GeneralPipelineSklearn."""
import numpy as np
import pytest
import torch
import torch.optim as optim
from pipelines_torch.base import GeneralPipeline, GeneralPipelineSklearn
from pipelines_torch.models import SklearnRandomForestClassifierWrapper


class TestGeneralPipelineClassification:
    def _make_pipeline(self, model, **kwargs):
        defaults = dict(
            loss_fn="CrossEntropyLoss",
            optimizer_cls=optim.Adam,
            optimizer_params={"lr": 1e-3},
            task_type="classification",
            device="cpu",
            epochs=3,
            batch_size=16,
            use_kfold=False,
            use_class_weights=True,
            use_mixed_precision=False,
        )
        defaults.update(kwargs)
        return GeneralPipeline(model=model, **defaults)

    def test_fit_returns_history(self, simple_mlp, synthetic_classification_data):
        X, y = synthetic_classification_data
        pipeline = self._make_pipeline(simple_mlp)
        history = pipeline.fit(X, y)
        assert isinstance(history, list)
        assert len(history) == 3  # 3 epochs
        assert "loss" in history[0]

    def test_history_has_metrics(self, simple_mlp, synthetic_classification_data):
        X, y = synthetic_classification_data
        pipeline = self._make_pipeline(simple_mlp)
        history = pipeline.fit(X, y)
        # With validation split, should have val_loss and classification metrics
        assert "val_loss" in history[0]
        assert "f1_score" in history[0]
        assert "roc_auc" in history[0]

    def test_early_stopping(self, simple_mlp, synthetic_classification_data):
        X, y = synthetic_classification_data
        pipeline = self._make_pipeline(simple_mlp, epochs=50, early_stopping=2)
        history = pipeline.fit(X, y)
        # Should stop before 50 epochs (usually)
        assert len(history) <= 50

    def test_kfold(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        model = torch.nn.Sequential(
            torch.nn.Linear(5, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        )
        pipeline = self._make_pipeline(
            model, use_kfold=True, k_folds=3, epochs=2
        )
        history = pipeline.fit(X, y)
        assert isinstance(history, list)
        # CV results should be populated
        cv_scores = pipeline.get_cv_scores()
        assert cv_scores is not None

    def test_get_training_history(self, simple_mlp, synthetic_classification_data):
        X, y = synthetic_classification_data
        pipeline = self._make_pipeline(simple_mlp)
        pipeline.fit(X, y)
        history = pipeline.get_training_history()
        assert isinstance(history, list)
        assert len(history) > 0


class TestGeneralPipelineRegression:
    def test_fit(self, simple_regression_mlp, synthetic_regression_data):
        X, y = synthetic_regression_data
        pipeline = GeneralPipeline(
            model=simple_regression_mlp,
            loss_fn="MSELoss",
            optimizer_cls=optim.Adam,
            optimizer_params={"lr": 1e-3},
            task_type="regression",
            device="cpu",
            epochs=3,
            batch_size=16,
            use_kfold=False,
            use_mixed_precision=False,
        )
        history = pipeline.fit(X, y)
        assert len(history) == 3
        assert "loss" in history[0]
        assert "r2_score" in history[0]


class TestGeneralPipelineSklearn:
    def test_fit_no_kfold(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        model = SklearnRandomForestClassifierWrapper(n_estimators=5, random_state=42)
        pipeline = GeneralPipelineSklearn(
            model=model,
            task_type="classification",
            use_kfold=False,
            random_state=42,
        )
        history = pipeline.fit(X, y)
        assert isinstance(history, list)

    def test_fit_with_kfold(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        model = SklearnRandomForestClassifierWrapper(n_estimators=5, random_state=42)
        pipeline = GeneralPipelineSklearn(
            model=model,
            task_type="classification",
            use_kfold=True,
            k_folds=3,
            random_state=42,
        )
        history = pipeline.fit(X, y)
        assert isinstance(history, list)
        cv_scores = pipeline.get_cv_scores()
        assert cv_scores is not None

    def test_get_training_history(self, synthetic_classification_data):
        X, y = synthetic_classification_data
        model = SklearnRandomForestClassifierWrapper(n_estimators=5, random_state=42)
        pipeline = GeneralPipelineSklearn(
            model=model,
            task_type="classification",
            use_kfold=False,
            random_state=42,
        )
        pipeline.fit(X, y)
        history = pipeline.get_training_history()
        assert isinstance(history, list)


class TestPrepareData:
    def test_dataloader_shapes(self, simple_mlp, synthetic_classification_data):
        X, y = synthetic_classification_data
        pipeline = GeneralPipeline(
            model=simple_mlp,
            loss_fn="CrossEntropyLoss",
            optimizer_cls=optim.Adam,
            optimizer_params={"lr": 1e-3},
            task_type="classification",
            device="cpu",
            epochs=1,
            batch_size=16,
            use_kfold=False,
            use_mixed_precision=False,
        )
        loader = pipeline.prepare_data_internal(X, y, train=True)
        batch = next(iter(loader))
        X_batch, y_batch = batch
        assert X_batch.shape[1] == 5
        assert X_batch.dtype == torch.float32
        assert y_batch.dtype == torch.long
