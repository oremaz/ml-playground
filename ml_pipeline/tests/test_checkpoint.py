"""Tests for checkpoint save/load utilities."""
import json
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import torch

from utils.utils import (
    _checkpoint_id,
    _load_index,
    _write_index_entry,
    model_exists,
    save_model,
    save_metrics,
)


@pytest.fixture
def metadata_core():
    return {
        "model_name": "TestMLP",
        "augmentation_name": "none",
        "task_type": "classification",
        "epochs": 10,
        "learning_rate": 1e-4,
        "weight_decay": None,
        "dropout": None,
        "batch_size": 32,
        "use_kfold": True,
        "k_folds": 5,
    }


class TestCheckpointId:
    def test_deterministic(self, metadata_core):
        id1 = _checkpoint_id(metadata_core)
        id2 = _checkpoint_id(metadata_core)
        assert id1 == id2

    def test_different_metadata_different_id(self, metadata_core):
        id1 = _checkpoint_id(metadata_core)
        modified = dict(metadata_core, model_name="OtherModel")
        id2 = _checkpoint_id(modified)
        assert id1 != id2

    def test_returns_12_char_hex(self, metadata_core):
        cid = _checkpoint_id(metadata_core)
        assert len(cid) == 12
        assert all(c in "0123456789abcdef" for c in cid)


class TestIndexOperations:
    def test_load_empty_index(self, tmp_path):
        with patch("utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            index = _load_index("test_exp")
            assert index == {}

    def test_write_and_load(self, tmp_path):
        with patch("utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            entry = {"checkpoint_id": "abc123", "model_name": "TestMLP"}
            _write_index_entry("test_exp", entry)
            index = _load_index("test_exp")
            assert "abc123" in index
            assert index["abc123"]["model_name"] == "TestMLP"

    def test_multiple_entries(self, tmp_path):
        with patch("utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            entry1 = {"checkpoint_id": "aaa", "model_name": "Model1"}
            entry2 = {"checkpoint_id": "bbb", "model_name": "Model2"}
            _write_index_entry("test_exp", entry1)
            _write_index_entry("test_exp", entry2)
            index = _load_index("test_exp")
            assert len(index) == 2


class TestModelExists:
    def test_not_exists(self, tmp_path, metadata_core):
        with patch("utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            assert model_exists(metadata_core, "test_exp") is False

    def test_exists_after_save(self, tmp_path, metadata_core):
        with patch("utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            model = torch.nn.Linear(5, 2)
            save_model(model, "test_exp", metadata_core)
            assert model_exists(metadata_core, "test_exp") is True

    def test_raises_without_path_start(self, metadata_core):
        with pytest.raises(ValueError, match="path_start"):
            model_exists(metadata_core, None)


class TestSaveModel:
    def test_save_torch_model(self, tmp_path, metadata_core):
        with patch("utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            model = torch.nn.Linear(5, 2)
            entry = save_model(model, "test_exp", metadata_core)
            assert "checkpoint_id" in entry
            assert entry["artifact_type"] == "pt_file"
            # Verify file exists
            artifact_path = os.path.join(str(tmp_path), "test_exp", entry["artifact_path"])
            assert os.path.exists(artifact_path)

    def test_save_sklearn_model(self, tmp_path, metadata_core):
        from pipelines_torch.models import SklearnRandomForestClassifierWrapper

        with patch("utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            model = SklearnRandomForestClassifierWrapper(n_estimators=5)
            entry = save_model(model, "test_exp", metadata_core)
            assert entry["artifact_type"] == "joblib"

    def test_dedup_index_entry(self, tmp_path, metadata_core):
        with patch("utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            model = torch.nn.Linear(5, 2)
            save_model(model, "test_exp", metadata_core)
            save_model(model, "test_exp", metadata_core)
            index = _load_index("test_exp")
            # Should only have one entry despite saving twice
            assert len(index) == 1


class TestSaveMetrics:
    def test_save_metrics(self, tmp_path):
        with patch("utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            os.makedirs(os.path.join(str(tmp_path), "test_exp"), exist_ok=True)
            history = [{"loss": 0.5, "val_loss": 0.6}, {"loss": 0.3, "val_loss": 0.4}]
            save_metrics(history, "abc123", "test_exp")
            metrics_path = os.path.join(str(tmp_path), "test_exp", "abc123_metrics.csv")
            assert os.path.exists(metrics_path)
