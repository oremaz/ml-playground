"""Public model paths are part of the saved-checkpoint contract."""

import io
import pickle

import joblib
import numpy as np
import torch

from ml_pipeline.pipelines_torch import hf_models, llama_models, models, tabular_models
from pipelines_torch import models as historical_models
from ml_pipeline.pipelines_torch import base, evaluation, sklearn_pipeline


def test_historical_class_paths_resolve_to_adapter_classes():
    for module in (tabular_models, hf_models, llama_models):
        for name, cls in vars(module).items():
            if not isinstance(cls, type) or cls.__module__ not in {module.__name__, models.__name__}:
                continue
            assert cls.__module__ == models.__name__
            assert getattr(models, name) is cls
            assert getattr(historical_models, name) is cls
            for path in (models.__name__, "pipelines_torch.models"):
                legacy_reference = f"c{path}\n{name}\n.".encode()
                assert pickle.loads(legacy_reference) is cls


def test_fitted_wrapper_and_torch_module_roundtrip():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    wrapper = models.LinearClassifierWrapper()
    wrapper.fit(X, np.array([0, 0, 1, 1]))
    stream = io.BytesIO()
    joblib.dump(wrapper, stream)
    stream.seek(0)
    np.testing.assert_array_equal(joblib.load(stream).predict_proba(X), wrapper.predict_proba(X))

    network = models.TorchMLPClassifier(2, hidden_dims=[4]).eval()
    stream = io.BytesIO()
    torch.save(network, stream)
    stream.seek(0)
    restored = torch.load(stream, map_location="cpu", weights_only=False)
    torch.testing.assert_close(restored(torch.from_numpy(X)), network(torch.from_numpy(X)))


def test_historical_pipeline_paths_and_fitted_predictions():
    for module in (evaluation, sklearn_pipeline):
        for name, cls in vars(module).items():
            if not isinstance(cls, type) or cls.__module__ not in {module.__name__, base.__name__}:
                continue
            assert cls.__module__ == base.__name__
            assert getattr(base, name) is cls
            for path in (base.__name__, "pipelines_torch.base"):
                assert pickle.loads(f"c{path}\n{name}\n.".encode()) is cls

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    estimator = models.LinearClassifierWrapper()
    estimator.fit(X, np.array([0, 0, 1, 1]))
    pipeline = base.GeneralPipelineSklearn(estimator.model, use_kfold=False)
    restored = pickle.loads(pickle.dumps(pipeline))
    np.testing.assert_array_equal(restored.predict_proba(X), pipeline.predict_proba(X))
