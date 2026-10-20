"""Tests for checkpoint save/load utilities."""
import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ml_pipeline.utils.utils import (
    _checkpoint_id,
    _load_index,
    _write_index_entry,
    load_model,
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
        with patch("ml_pipeline.utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            index = _load_index("test_exp")
            assert index == {}

    def test_write_and_load(self, tmp_path):
        with patch("ml_pipeline.utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            entry = {"checkpoint_id": "abc123", "model_name": "TestMLP"}
            _write_index_entry("test_exp", entry)
            index = _load_index("test_exp")
            assert "abc123" in index
            assert index["abc123"]["model_name"] == "TestMLP"

    def test_multiple_entries(self, tmp_path):
        with patch("ml_pipeline.utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            entry1 = {"checkpoint_id": "aaa", "model_name": "Model1"}
            entry2 = {"checkpoint_id": "bbb", "model_name": "Model2"}
            _write_index_entry("test_exp", entry1)
            _write_index_entry("test_exp", entry2)
            index = _load_index("test_exp")
            assert len(index) == 2


class TestModelExists:
    def test_not_exists(self, tmp_path, metadata_core):
        with patch("ml_pipeline.utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            assert model_exists(metadata_core, "test_exp") is False

    def test_exists_after_save(self, tmp_path, metadata_core):
        with patch("ml_pipeline.utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            model = torch.nn.Linear(5, 2)
            save_model(model, "test_exp", metadata_core)
            assert model_exists(metadata_core, "test_exp") is True

    def test_raises_without_path_start(self, metadata_core):
        with pytest.raises(ValueError, match="path_start"):
            model_exists(metadata_core, None)


class TestSaveModel:
    def test_save_torch_model(self, tmp_path, metadata_core):
        with patch("ml_pipeline.utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            model = torch.nn.Linear(5, 2)
            entry = save_model(model, "test_exp", metadata_core)
            assert "checkpoint_id" in entry
            assert entry["artifact_type"] == "pt_file"
            # Verify file exists
            artifact_path = os.path.join(str(tmp_path), "test_exp", entry["artifact_path"])
            assert os.path.exists(artifact_path)

    def test_save_sklearn_model(self, tmp_path, metadata_core):
        from ml_pipeline.pipelines_torch.models import SklearnRandomForestClassifierWrapper

        with patch("ml_pipeline.utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            model = SklearnRandomForestClassifierWrapper(n_estimators=5)
            entry = save_model(model, "test_exp", metadata_core)
            assert entry["artifact_type"] == "joblib"

    def test_save_and_load_wrapped_sklearn_model(self, tmp_path, metadata_core):
        from ml_pipeline.pipelines_torch.models import SklearnRandomForestClassifierWrapper

        X = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])

        with patch("ml_pipeline.utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            model = SklearnRandomForestClassifierWrapper(n_estimators=5, random_state=42)
            model.fit(X, y)
            entry = save_model(model, "test_exp", metadata_core)
            loaded = load_model(
                SklearnRandomForestClassifierWrapper,
                {"n_estimators": 5, "random_state": 42},
                "test_exp",
                entry["checkpoint_id"],
            )

            assert entry["artifact_type"] == "joblib"
            assert isinstance(loaded, SklearnRandomForestClassifierWrapper)
            np.testing.assert_array_equal(loaded.predict(X), model.predict(X))

    def test_save_and_load_direct_sklearn_estimator(self, tmp_path, metadata_core):
        from sklearn.linear_model import LogisticRegression

        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])

        with patch("ml_pipeline.utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            model = LogisticRegression().fit(X, y)
            entry = save_model(model, "test_exp", metadata_core)
            loaded = load_model(
                LogisticRegression,
                {},
                "test_exp",
                entry["checkpoint_id"],
            )
            artifact_path = os.path.join(str(tmp_path), "test_exp", entry["artifact_path"])

            assert entry["artifact_type"] == "joblib"
            assert os.path.exists(artifact_path)
            assert isinstance(loaded, LogisticRegression)
            np.testing.assert_array_equal(loaded.predict(X), model.predict(X))

    def test_dedup_index_entry(self, tmp_path, metadata_core):
        with patch("ml_pipeline.utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            model = torch.nn.Linear(5, 2)
            save_model(model, "test_exp", metadata_core)
            save_model(model, "test_exp", metadata_core)
            index = _load_index("test_exp")
            # Should only have one entry despite saving twice
            assert len(index) == 1


class TestSaveMetrics:
    def test_save_metrics(self, tmp_path):
        with patch("ml_pipeline.utils.utils.RESULTS_DIR_OUT", str(tmp_path)):
            os.makedirs(os.path.join(str(tmp_path), "test_exp"), exist_ok=True)
            history = [{"loss": 0.5, "val_loss": 0.6}, {"loss": 0.3, "val_loss": 0.4}]
            save_metrics(history, "abc123", "test_exp")
            metrics_path = os.path.join(str(tmp_path), "test_exp", "abc123_metrics.csv")
            assert os.path.exists(metrics_path)


@pytest.mark.parametrize("architecture", ["gemma4", "lfm2"])
def test_attached_classifier_adapter_reload_matches_saved_probabilities(tmp_path, monkeypatch, architecture):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (Gemma4Config, Gemma4ForConditionalGeneration,
                              Lfm2Config, Lfm2ForCausalLM, PreTrainedTokenizerFast)
    from ml_pipeline.pipelines_torch.models import HuggingFaceQLoRAWrapper
    from ml_pipeline.utils import utils

    torch.manual_seed(42)
    if architecture == "gemma4":
        config = Gemma4Config(text_config=dict(
            vocab_size=24, vocab_size_per_layer_input=24,
            hidden_size=16, hidden_size_per_layer_input=4, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, global_head_dim=8, layer_types=["full_attention"],
            rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 10000.0}},
        ))
        base = Gemma4ForConditionalGeneration(config)
    else:
        config = Lfm2Config(vocab_size=24, hidden_size=16, intermediate_size=32,
                           num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                           block_auto_adjust_ff_dim=False, full_attn_idxs=[0])
        base = Lfm2ForCausalLM(config)
    base_dir = tmp_path / "base"
    base.save_pretrained(base_dir)
    base.config._name_or_path = str(base_dir)
    base.name_or_path = str(base_dir)
    attached = HuggingFaceQLoRAWrapper._attach_sequence_classification_head(
        base, 16, 0.02, 2, "classification", torch.float32,
    )
    model = get_peft_model(attached, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=2, lora_alpha=2,
        target_modules=["q_proj", "v_proj"], modules_to_save=["score"],
    ))
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.01)
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "human": 2, "text": 3, "generated": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    wrapper = HuggingFaceQLoRAWrapper.__new__(HuggingFaceQLoRAWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]",
                                               padding_side="left", model_input_names=["input_ids", "attention_mask"])
    wrapper.task_type, wrapper.device, wrapper.max_seq_length = "classification", "cpu", 16
    wrapper._needs_token_type_ids = False
    expected = wrapper.predict_proba(["human text", "generated"], batch_size=2)
    directory = tmp_path / "run" / "adapter"
    wrapper.save_pretrained(str(directory))
    monkeypatch.setattr(utils, "RESULTS_DIR_OUT", str(tmp_path))
    utils._write_index_entry("run", {"checkpoint_id": "adapter", "artifact_type": "hf_dir", "artifact_path": "adapter"})
    loaded = load_model(lambda **kwargs: None, {}, "run", "adapter")
    np.testing.assert_allclose(loaded.predict_proba(["human text", "generated"], batch_size=2), expected,
                               rtol=1e-5, atol=1e-6)
