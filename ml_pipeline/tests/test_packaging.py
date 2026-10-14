import importlib
import os
import subprocess
import sys

import joblib
import numpy as np

from ml_pipeline.pipelines_torch.models import SklearnRandomForestClassifierWrapper


def test_load_checkpoint_with_historical_module_name(tmp_path, monkeypatch):
    legacy = importlib.import_module("pipelines_torch.models")
    assert legacy.SklearnRandomForestClassifierWrapper is SklearnRandomForestClassifierWrapper
    model = SklearnRandomForestClassifierWrapper(n_estimators=2, random_state=42)
    features = np.array([[0.0], [0.1], [1.0], [1.1]])
    model.fit(features, np.array([0, 0, 1, 1]))
    expected = model.predict(features)
    path = tmp_path / "legacy.joblib"
    with monkeypatch.context() as context:
        context.setattr(SklearnRandomForestClassifierWrapper, "__module__", "pipelines_torch.models")
        joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(restored.predict(features), expected)
    assert type(restored) is SklearnRandomForestClassifierWrapper


def test_results_directory_can_be_configured(tmp_path):
    destination = tmp_path / "artifacts"
    environment = dict(os.environ, ML_PIPELINE_RESULTS_DIR=str(destination))
    result = subprocess.run(
        [sys.executable, "-c", "from ml_pipeline.utils.utils import RESULTS_DIR_IN, RESULTS_DIR_OUT; print(RESULTS_DIR_IN); print(RESULTS_DIR_OUT)"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == [str(destination), str(destination)]
    assert destination.is_dir()
