"""Context isolation, class order and persistence for the Kumo adapter."""

import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone
from sklearn.utils import get_tags

from ml_pipeline.pipelines_torch.kumo_models import KumoTabularClassifier, KumoTabularRegressor


@pytest.fixture
def backend(monkeypatch):
    calls = []

    class Table:
        @staticmethod
        def from_pandas(frame, stypes, device):
            return frame.copy()

    class Model:
        def __init__(self, **kwargs):
            self.task = kwargs["task"]

        def eval(self):
            return self

        def requires_grad_(self, value):
            return self

        def __call__(self, **kwargs):
            calls.append(kwargs)
            rows = len(kwargs["x_query"])
            if self.task == "classification":
                return SimpleNamespace(
                    columns={"numerical": ("1", "0")},
                    numerical=torch.tensor([[0.8, 0.2]]).expand(rows, -1),
                )

            class Quantiles:
                def __getitem__(self, key):
                    assert key[1] == "q500"
                    return SimpleNamespace(numerical=torch.full((rows, 1), 3.0))

            return Quantiles()

    monkeypatch.setitem(sys.modules, "sdm", SimpleNamespace(
        TableTensor=Table, infer_stypes=lambda frame: dict.fromkeys(frame.columns, "numerical"),
        models=SimpleNamespace(KumoTabular=Model), Stype=SimpleNamespace(numerical="numerical"),
        __version__="0.2.1",
    ))
    return calls


def test_training_context_class_order_batching_and_pickle(backend):
    X = pd.DataFrame({"x": [0., 1., 2., 3.], "z": [2., 0., 1., 3.]})
    model = KumoTabularClassifier(batch_size=1).fit(X, ["machine", "human", "machine", "human"])
    X.iloc[0, 0] = 999
    queries = pd.DataFrame({"x": [42., 43.], "z": [0., 0.]})
    probabilities = model.predict_proba(queries)
    np.testing.assert_allclose(probabilities, [[0.2, 0.8], [0.2, 0.8]])
    assert model.predict(queries).tolist() == ["machine", "machine"]
    assert all(call["x_context"].iloc[0, 0] == 0 for call in backend)
    assert all(42 not in call["x_context"].x.values for call in backend)
    assert all(call["y_context"].target.tolist() == [1, 0, 1, 0] for call in backend)
    assert {call["generator"].initial_seed() for call in backend} == {42}
    restored = pickle.loads(pickle.dumps(model))
    assert restored._model is None and restored._context is None
    np.testing.assert_array_equal(restored.predict_proba(queries), probabilities)
    sys.modules["sdm"].__version__ = "changed"
    with pytest.raises(ValueError, match="SDM version changed"):
        restored.predict(queries)


def test_regression_uses_median_and_context_sampling_is_seeded(backend):
    X = np.arange(40).reshape(20, 2)
    model = KumoTabularRegressor(max_context_rows=7, random_state=19).fit(X, np.arange(20))
    other = clone(model).fit(X, np.arange(20))
    np.testing.assert_array_equal(model.context_indices_, other.context_indices_)
    assert len(model.X_context_) == 7
    np.testing.assert_array_equal(model.predict(X[:3]), [3., 3., 3.])
    assert model.model_metadata_["regression_statistic"] == "median"


@pytest.mark.parametrize("change", [{"batch_size": 0}, {"size": "giant"}, {"max_context_rows": True}])
def test_invalid_configuration_is_rejected(change):
    with pytest.raises(ValueError):
        KumoTabularClassifier(**change).fit([[1], [2]], [0, 1])


def test_schema_and_unsupported_weights_are_rejected(backend):
    model = KumoTabularClassifier().fit(pd.DataFrame({"a": [1, 2], "b": [0, 1]}), [0, 1])
    assert get_tags(model).input_tags.allow_nan
    assert get_tags(model).estimator_type == "classifier"
    with pytest.raises(ValueError, match="names and order"):
        model.predict(pd.DataFrame({"b": [1], "a": [0]}))
    with pytest.raises(ValueError, match="sample weights"):
        model.fit([[1], [2]], [0, 1], sample_weight=[1, 2])
    assert model.predict_proba(pd.DataFrame({"a": [], "b": []})).shape == (0, 2)


def test_classification_context_subset_keeps_classes():
    labels = np.repeat(["alpha", "beta", "gamma"], 10)
    model = KumoTabularClassifier(max_context_rows=9).fit(np.arange(60).reshape(30, 2), labels)
    assert set(model.y_context_) == {0, 1, 2}
