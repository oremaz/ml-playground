import numpy as np
import pytest
import torch

from ml_pipeline.pipelines_torch.base import GeneralPipeline
from ml_pipeline.utils.training_checkpoint import CheckpointPause, load_checkpoint


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Dropout(0.3), torch.nn.Linear(8, 2))
        self.calls = 0
        self.interrupt = None

    def forward(self, inputs):
        if self.training:
            self.calls += 1
            if self.interrupt is not None and self.calls == 3:
                self.interrupt()
        return self.layers(inputs)


def make(path, stop=False):
    torch.manual_seed(42)
    model = Policy()
    pipeline = GeneralPipeline(
        model=model, loss_fn="CrossEntropyLoss", optimizer_cls=torch.optim.Adam,
        optimizer_params={"lr": 0.01}, scheduler_cls=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 1, "gamma": 0.8}, epochs=2, batch_size=4,
        use_kfold=False, use_mixed_precision=False,
    )
    pipeline.restart_path = str(path)
    if stop:
        model.interrupt = lambda: setattr(pipeline._stop_request, "requested", True)
    return pipeline


def test_restart_preserves_shuffle_dropout_optimizer_best_weights_and_history(tmp_path):
    X = np.random.default_rng(13).normal(size=(48, 3)).astype(np.float32)
    y = np.arange(48) % 2
    continuous = make(tmp_path / "continuous")
    continuous.fit(X, y)
    interrupted = make(tmp_path / "restart", stop=True)
    with pytest.raises(CheckpointPause):
        interrupted.fit(X, y)
    assert load_checkpoint(tmp_path / "restart/restart.pt")["batch"] == 3
    resumed = make(tmp_path / "restart")
    resumed.fit(X, y)
    for name, value in continuous.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], value, rtol=0, atol=0)
    assert continuous.history == resumed.history
    assert continuous.scheduler.state_dict() == resumed.scheduler.state_dict()


def test_float32_input_tensor_reuses_array_storage():
    pipeline = make("unused")
    X = np.ones((8, 3), dtype=np.float32)
    y = np.arange(8, dtype=np.int64) % 2
    loader = pipeline.prepare_data_internal(X, y)
    assert loader.dataset.tensors[0].data_ptr() == X.ctypes.data
    assert loader.dataset.tensors[1].data_ptr() == y.ctypes.data


def test_qlora_wrapper_native_resume_without_model_download(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast
    from ml_pipeline.pipelines_torch.models import HuggingFaceQLoRAWrapper
    from ml_pipeline.utils import training_checkpoint as checkpoints

    def wrapper():
        torch.manual_seed(42)
        result = HuggingFaceQLoRAWrapper.__new__(HuggingFaceQLoRAWrapper)
        torch.nn.Module.__init__(result)
        tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "human": 2, "text": 3, "generated": 4}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        result.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]")
        result.model = BertForSequenceClassification(BertConfig(
            vocab_size=5, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=16, num_labels=2,
        ))
        result._is_quantized = True
        result._needs_token_type_ids = True
        result._gradient_accumulation_steps = 2
        result.max_seq_length = 16
        result.task_type, result.device = "classification", "cpu"
        return result

    X = ["human text", "generated text"] * 4
    y = np.array([1, 0] * 4)
    continuous = wrapper()
    history = continuous.fit(X, y, output_dir=str(tmp_path / "continuous"), epochs=2, batch_size=2)
    active = []
    original_stop = checkpoints.StopRequest
    original_due = checkpoints.CheckpointClock.due

    class Interrupt(original_stop):
        def __enter__(self):
            active.append(self)
            return super().__enter__()

    def due(clock):
        active[-1].requested = True
        return True

    monkeypatch.setattr(checkpoints, "StopRequest", Interrupt)
    monkeypatch.setattr(checkpoints.CheckpointClock, "due", due)
    interrupted = wrapper()
    with pytest.raises(CheckpointPause):
        interrupted.fit(X, y, output_dir=str(tmp_path / "restart"), epochs=2, batch_size=2)
    monkeypatch.setattr(checkpoints, "StopRequest", original_stop)
    monkeypatch.setattr(checkpoints.CheckpointClock, "due", original_due)
    resumed = wrapper()
    resumed_history = resumed.fit(X, y, output_dir=str(tmp_path / "restart"), epochs=2, batch_size=2)
    for name, value in continuous.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], value, rtol=0, atol=0)
    assert resumed_history == history


def test_classical_resume_skips_completed_folds_and_full_fit(tmp_path, monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from ml_pipeline.pipelines_torch.base import GeneralPipelineSklearn

    X = np.random.default_rng(7).normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    original = GeneralPipelineSklearn.internal_fit
    calls = []
    interrupt = [False]

    def fit(pipeline, *arguments, **keywords):
        calls.append(len(arguments[0]))
        result = original(pipeline, *arguments, **keywords)
        if interrupt[0]:
            pipeline._stop_request.requested = True
        return result

    monkeypatch.setattr(GeneralPipelineSklearn, 'internal_fit', fit)

    def pipeline(path):
        result = GeneralPipelineSklearn(LogisticRegression(random_state=42), k_folds=3)
        result.restart_path = str(path)
        return result

    full = pipeline(tmp_path / 'full')
    full.fit(X, y)
    assert len(calls) == 4
    calls.clear()
    interrupt[0] = True
    with pytest.raises(CheckpointPause):
        pipeline(tmp_path / 'resume').fit(X, y)
    assert len(calls) == 1
    interrupt[0] = False
    resumed = pipeline(tmp_path / 'resume')
    resumed.fit(X, y)
    assert len(calls) == 4
    np.testing.assert_array_equal(full.model.coef_, resumed.model.coef_)
    assert full.history == resumed.history
    calls.clear()
    completed = pipeline(tmp_path / 'resume')
    completed.fit(X, y)
    assert not calls
    np.testing.assert_array_equal(completed.model.coef_, full.model.coef_)
    with pytest.raises(ValueError):
        pipeline(tmp_path / 'resume').fit(X + 1, y)


def test_multimodal_classifier_syncs_nested_padding_before_batched_forward():
    from types import SimpleNamespace
    from transformers import Qwen3_5Config, Qwen3_5ForSequenceClassification
    from ml_pipeline.pipelines_torch.models import HuggingFaceQLoRAWrapper

    config = Qwen3_5Config(
        text_config=dict(vocab_size=24, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=8, layer_types=["full_attention"],
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                          "partial_rotary_factor": 1.0,
                                          "mrope_section": [1, 1, 2]}),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32,
                           num_heads=2, out_hidden_size=16, num_position_embeddings=16),
        num_labels=2,
    )
    result = HuggingFaceQLoRAWrapper.__new__(HuggingFaceQLoRAWrapper)
    torch.nn.Module.__init__(result)
    result.model = Qwen3_5ForSequenceClassification(config).eval()
    result.tokenizer = SimpleNamespace(pad_token_id=0, bos_token_id=1, eos_token_id=2)
    ids = torch.tensor([[0, 1, 3, 4], [1, 5, 6, 7]])
    with pytest.raises(ValueError, match="no padding token"):
        result.model(input_ids=ids, attention_mask=ids.ne(0))
    result._sync_special_token_ids_with_tokenizer()
    assert config.get_text_config().pad_token_id == 0
    with torch.no_grad():
        output = result.model(input_ids=ids, attention_mask=ids.ne(0))
    assert output.logits.shape == (2, 2)
    assert torch.isfinite(output.logits).all()


@pytest.mark.parametrize("output_hidden_states", [None, False, True])
def test_gemma_classifier_accepts_peft_forward_options(output_hidden_states):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    from ml_pipeline.pipelines_torch.models import HuggingFaceQLoRAWrapper

    config = Gemma4Config(text_config=dict(
        vocab_size=24, vocab_size_per_layer_input=24,
        hidden_size=16, hidden_size_per_layer_input=4, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, global_head_dim=8, layer_types=["full_attention"],
        rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 10000.0}},
    ))
    model = HuggingFaceQLoRAWrapper._attach_sequence_classification_head(
        Gemma4ForConditionalGeneration(config), 16, 0.02, 2, "classification", torch.float32,
    )
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=2, lora_alpha=2,
        target_modules=["q_proj", "v_proj"], modules_to_save=["score"],
    ))
    ids = torch.tensor([[0, 2, 3, 4], [2, 5, 6, 7]])
    output = model(input_ids=ids, attention_mask=ids.ne(0), labels=torch.tensor([0, 1]),
                   output_hidden_states=output_hidden_states, return_dict=True)
    assert output.logits.shape == (2, 2)
    assert torch.isfinite(output.loss)
    assert output.hidden_states is not None
    output.loss.backward()
    trained = dict(model.named_parameters())
    assert any("score" in name and value.grad is not None for name, value in trained.items())
    assert any("lora_B" in name and value.grad is not None and torch.isfinite(value.grad).all()
               for name, value in trained.items())


def test_attached_causal_classifier_gradient_accumulation_matches_full_batch(tmp_path):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Lfm2Config, Lfm2ForCausalLM, Trainer, TrainingArguments
    from ml_pipeline.pipelines_torch.models import HuggingFaceQLoRAWrapper

    def make():
        torch.manual_seed(42)
        config = Lfm2Config(vocab_size=24, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                            block_auto_adjust_ff_dim=False, full_attn_idxs=[0])
        model = HuggingFaceQLoRAWrapper._attach_sequence_classification_head(
            Lfm2ForCausalLM(config), 16, 0.02, 2, "classification", torch.float32,
        )
        return get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS, r=2, lora_alpha=2, lora_dropout=0,
            target_modules=["q_proj", "v_proj"], modules_to_save=["score"],
        ))

    dataset = [{"input_ids": torch.tensor([2, 3, index + 4]),
                "attention_mask": torch.ones(3, dtype=torch.long),
                "labels": torch.tensor(index % 2)} for index in range(4)]
    models = [make(), make()]
    for batch, accumulation, candidate in [(4, 1, models[0]), (2, 2, models[1])]:
        trainer = Trainer(model=candidate, train_dataset=dataset, args=TrainingArguments(
            output_dir=str(tmp_path / str(batch)), per_device_train_batch_size=batch,
            gradient_accumulation_steps=accumulation, max_steps=1, optim="sgd",
            learning_rate=0.01, lr_scheduler_type="constant", weight_decay=0,
            max_grad_norm=0, use_cpu=True, report_to="none", save_strategy="no",
            disable_tqdm=True,
        ))
        assert trainer.model_accepts_loss_kwargs is False
        trainer.train()
    for (name, expected), (_, actual) in zip(models[0].named_parameters(), models[1].named_parameters()):
        if expected.requires_grad:
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
