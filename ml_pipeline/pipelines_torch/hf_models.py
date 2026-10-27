"""QLoRA model construction, persistence and batched inference."""

import logging
import inspect
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("ml_pipeline.pipelines_torch.models")

from .hf_training import HuggingFaceTrainingMixin


class HuggingFaceQLoRAWrapper(HuggingFaceTrainingMixin, nn.Module):
    """
    Wrapper for QLoRA (Quantized LoRA) fine-tuning with HuggingFace models.
    Supports both classification and regression tasks.
    Compatible with standard encoder models (BERT family, ModernBERT) and causal /
    hybrid-architecture decoder models (Qwen3.5, Gemma 4, Llama, Mistral, etc.).
    Based on the guidelines from https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora

    Note: This wrapper inherits from nn.Module to be compatible with GeneralPipeline,
    but training is handled by HuggingFace Trainer, not by the standard PyTorch training loop.
    """

    __module__ = "ml_pipeline.pipelines_torch.models"
    # Standard attention/MLP linear names present in virtually every transformer architecture.
    # Used as a safe fallback when "all-linear" fails (e.g. hybrid MoE/SSM architectures).
    _FALLBACK_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    @staticmethod
    def _model_dtype_kwargs(torch_dtype):
        """Use the Transformers 5 name while keeping Transformers 4 compatibility."""
        try:
            import transformers

            major = int(transformers.__version__.split(".", 1)[0])
        except Exception:
            major = 4
        return {"dtype": torch_dtype} if major >= 5 else {"torch_dtype": torch_dtype}

    def __init__(self, model_name: str, tokenizer_name: Optional[str] = None,
                 num_labels: int = 2, task_type: str = 'classification', device: str = 'cpu',
                 lora_config: Optional[Dict[str, Any]] = None, quantization_config: Optional[Dict[str, Any]] = None,
                 gradient_accumulation_steps: int = 2, max_seq_length: int = 512):
        super().__init__()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
        from peft import LoraConfig, get_peft_model, TaskType
        import torch

        self.task_type = task_type
        self.device = device

        # Force GPU usage if available, raise error if not
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU is required for HuggingFaceQLoRAWrapper but CUDA is not available. "
                "Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support installed."
            )

        # Determine dtype based on GPU compute capability
        try:
            compute_capability = torch.cuda.get_device_capability()[0]
            torch_dtype = torch.bfloat16 if compute_capability >= 8 else torch.float16
            logger.info("Using GPU with compute capability %d.x - dtype: %s", compute_capability, torch_dtype)
        except Exception:
            torch_dtype = torch.float16
            logger.info("Using GPU with default dtype: %s", torch_dtype)

        # Detect whether this is a causal (decoder-only) model so we can configure
        # padding and LoRA correctly.  We inspect the HF config before loading weights.
        hf_config = AutoConfig.from_pretrained(tokenizer_name or model_name, trust_remote_code=True)
        model_arch = getattr(hf_config, "architectures", [""])[0].lower()
        self._is_causal = any(k in model_arch for k in ("causal", "gpt", "llama", "qwen", "gemma", "mistral", "falcon", "mpt"))

        # Set up tokenizer with correct padding side for the architecture
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name,
            trust_remote_code=True,
            # Causal LMs: left-pad so the classification head reads the last *real* token
            padding_side="left" if self._is_causal else "right",
        )
        # Many causal LMs ship without a pad token; reuse eos_token as pad
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info("pad_token not set: using eos_token as pad_token for causal LM.")

        if task_type == 'regression':
            num_labels = 1

        # Build BitsAndBytes quantization config (NF4 4-bit)
        bnb_config = None
        if quantization_config is None:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    # Keep classification heads unquantized: PEFT's modules_to_save
                    # cannot clone 4-bit layers (quant state is lost on copy)
                    llm_int8_skip_modules=["classifier", "score", "head"],
                )
                logger.info("4-bit NF4 quantization enabled")
            except Exception as e:
                logger.warning("Failed to setup quantization config: %s. Falling back to fp16/bf16.", e)
        elif quantization_config is not None:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(**quantization_config)
                logger.info("Custom quantization config applied")
            except Exception as e:
                logger.warning("Failed to setup custom quantization config: %s. Falling back.", e)

        model_kwargs = {
            "num_labels": num_labels,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        model_kwargs.update(self._model_dtype_kwargs(torch_dtype))
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        # Suppress warnings about newly initialised classification head weights
        model_kwargs["ignore_mismatched_sizes"] = True

        self.model = self._load_sequence_classifier(
            model_name, hf_config, model_kwargs, num_labels, task_type, torch_dtype
        )

        # Some architectures (e.g. Gemma3) require an explicit token_type_ids input
        # even for single-segment classification, and raise instead of defaulting it.
        import inspect
        self._needs_token_type_ids = "token_type_ids" in inspect.signature(self.model.forward).parameters

        # Keep Trainer's Transformers 5 special-token alignment quiet and deterministic.
        # ModernBERT's config has legacy BOS/EOS ids while its tokenizer does not expose
        # BOS/EOS tokens; for sequence classification the tokenizer should be the source
        # of truth.
        self._sync_special_token_ids_with_tokenizer()

        if bnb_config is not None:
            from peft import prepare_model_for_kbit_training
            # use_reentrant=False avoids autograd issues with causal LMs + gradient checkpointing
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

        # Auto-detect classification head module name to add to modules_to_save.
        # Common names: "classifier" (BERT-family), "score" (Llama/Mistral/Qwen causal LMs).
        _head_candidates = ["classifier", "score"]
        _head_name = next(
            (n for n in _head_candidates if hasattr(self.model, n)), None
        )
        _modules_to_save = [_head_name] if _head_name else []

        if lora_config is None:
            lora_config = {
                'lora_alpha': 16,
                'lora_dropout': 0.05,
                'r': 16,
                'bias': "none",
                'target_modules': "all-linear",
                'task_type': TaskType.SEQ_CLS,
            }
            if _modules_to_save:
                lora_config['modules_to_save'] = _modules_to_save

        # Apply LoRA: fall back to explicit module names if "all-linear" fails
        # (can happen with hybrid MoE / SSM architectures like Qwen3.5 or Delta Networks)
        lora_cfg = LoraConfig(**lora_config)
        try:
            self.model = get_peft_model(self.model, lora_cfg)
            logger.info("LoRA applied with target_modules=%s", lora_config.get('target_modules'))
        except (ValueError, RuntimeError) as e:
            logger.warning(
                "LoRA with target_modules=%s failed (%s). "
                "Retrying with fallback explicit modules: %s",
                lora_config.get('target_modules'), e, self._FALLBACK_TARGET_MODULES,
            )
            fallback_cfg = LoraConfig(
                lora_alpha=lora_config.get('lora_alpha', 16),
                lora_dropout=lora_config.get('lora_dropout', 0.05),
                r=lora_config.get('r', 16),
                bias=lora_config.get('bias', "none"),
                target_modules=self._FALLBACK_TARGET_MODULES,
                task_type=TaskType.SEQ_CLS,
                modules_to_save=_modules_to_save if _modules_to_save else None,
            )
            self.model = get_peft_model(self.model, fallback_cfg)
            logger.info("LoRA applied with fallback target_modules=%s", self._FALLBACK_TARGET_MODULES)

        self._sync_special_token_ids_with_tokenizer()
        self._default_peft_return_dict()

        self._is_quantized = bnb_config is not None
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self.max_seq_length = max_seq_length  # Default, can be overridden in fit()

    @staticmethod
    def _load_sequence_classifier(model_name, hf_config, model_kwargs, num_labels, task_type, torch_dtype):
        from transformers import AutoModelForSequenceClassification

        if hf_config.model_type == "gemma4":
            return HuggingFaceQLoRAWrapper._load_gemma4_sequence_classifier(
                model_name, hf_config, model_kwargs, num_labels, task_type, torch_dtype
            )
        if hf_config.model_type in {"lfm2", "nanbeige"}:
            return HuggingFaceQLoRAWrapper._load_causal_sequence_classifier(
                model_name, hf_config, model_kwargs, num_labels, task_type, torch_dtype
            )
        return AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)

    @staticmethod
    def _load_gemma4_sequence_classifier(
        model_name, hf_config, model_kwargs, num_labels, task_type, torch_dtype
    ):
        """Attach a trainable sequence-classification head to Gemma 4's text path."""
        from transformers import AutoModelForMultimodalLM

        model_kwargs = dict(model_kwargs)
        model_kwargs.pop("num_labels", None)
        model_kwargs.pop("ignore_mismatched_sizes", None)
        model = AutoModelForMultimodalLM.from_pretrained(model_name, **model_kwargs)
        return HuggingFaceQLoRAWrapper._attach_sequence_classification_head(
            model, hf_config.text_config.hidden_size, hf_config.initializer_range,
            num_labels, task_type, torch_dtype,
        )

    @staticmethod
    def _load_causal_sequence_classifier(
        model_name, hf_config, model_kwargs, num_labels, task_type, torch_dtype
    ):
        """Attach a classifier when Transformers has no sequence-classification auto class."""
        from transformers import AutoModelForCausalLM

        model_kwargs = dict(model_kwargs)
        model_kwargs.pop("num_labels", None)
        model_kwargs.pop("ignore_mismatched_sizes", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        return HuggingFaceQLoRAWrapper._attach_sequence_classification_head(
            model, hf_config.hidden_size, hf_config.initializer_range,
            num_labels, task_type, torch_dtype,
        )

    @staticmethod
    def _attach_sequence_classification_head(
        model, hidden_size, initializer_range, num_labels, task_type, torch_dtype
    ):
        import types
        import torch
        from torch import nn
        from transformers.modeling_outputs import SequenceClassifierOutput

        model.score = nn.Linear(
            hidden_size, num_labels, bias=False, device=model.device, dtype=torch_dtype
        )
        nn.init.normal_(model.score.weight, mean=0.0, std=initializer_range)
        model.accepts_loss_kwargs = False
        model.config.num_labels = num_labels
        model.config.problem_type = "regression" if task_type == "regression" else "single_label_classification"
        multimodal_backbone = model.model

        def sequence_classification_forward(
            self, input_ids=None, attention_mask=None, labels=None, **kwargs
        ):
            kwargs.pop("return_dict", None)
            kwargs.pop("output_hidden_states", None)
            kwargs.pop("num_items_in_batch", None)
            kwargs["use_cache"] = False
            outputs = multimodal_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
            logits = self.score(outputs.last_hidden_state[:, -1, :])
            loss = None
            if labels is not None:
                if self.config.problem_type == "regression":
                    loss = nn.functional.mse_loss(logits.squeeze(-1), labels.float())
                else:
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, self.config.num_labels), labels.view(-1)
                    )
            return SequenceClassifierOutput(
                loss=loss, logits=logits, hidden_states=outputs.hidden_states
            )

        model.forward = types.MethodType(sequence_classification_forward, model)
        return model

    def _sync_special_token_ids_with_tokenizer(self):
        for attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
            token_id = getattr(self.tokenizer, attr, None)
            if hasattr(self.model, "config"):
                setattr(self.model.config, attr, token_id)
                get_text_config = getattr(self.model.config, "get_text_config", None)
                if get_text_config is not None:
                    setattr(get_text_config(), attr, token_id)
            generation_config = getattr(self.model, "generation_config", None)
            if generation_config is not None and hasattr(generation_config, attr):
                setattr(generation_config, attr, token_id)

    def _default_peft_return_dict(self):
        """Avoid PEFT reading deprecated config.use_return_dict under Transformers 5."""
        import functools

        original_forward = self.model.forward

        @functools.wraps(original_forward)
        def forward_with_return_dict(*args, **kwargs):
            kwargs.setdefault("return_dict", True)
            return original_forward(*args, **kwargs)

        self.model.forward = forward_with_return_dict

    def to(self, device):
        self.device = device
        # Model is already on device due to device_map="auto"
        return self

    def save_pretrained(self, save_directory: str):
        """
        Save the PEFT model and tokenizer to a directory.
        This allows the model to be reloaded later.
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        # Save the PEFT adapter weights
        self.model.save_pretrained(save_directory)
        # Save the tokenizer
        self.tokenizer.save_pretrained(save_directory)
        # Save additional metadata
        import json
        metadata = {
            'task_type': self.task_type,
            'max_seq_length': getattr(self, 'max_seq_length', 512),
        }
        # Save label encoder if it exists
        if hasattr(self, 'label_encoder'):
            metadata['label_classes'] = self.label_encoder.classes_.tolist()
        with open(os.path.join(save_directory, 'wrapper_metadata.json'), 'w') as f:
            json.dump(metadata, f)
        logger.info("PEFT model, tokenizer, and metadata saved to %s", save_directory)


    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def _tokenize(self, text, **tok_kwargs):
        """Tokenize with the token_type_ids some architectures (e.g. Gemma3) require."""
        inputs = self.tokenizer(text, **tok_kwargs)
        if self._needs_token_type_ids and 'token_type_ids' not in inputs:
            inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids'])
        return inputs

    def _batched_logits(self, X, batch_size: int = 32):
        """Tokenize and forward X in padded batches, yielding float32 logits per batch.

        Dynamic padding (vs. per-example calls) is what makes inference on
        large evaluation sets (tens of thousands of rows) tractable.
        """
        for start in range(0, len(X), batch_size):
            chunk = X[start:start + batch_size]
            inputs = self._tokenize(
                chunk,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=self.max_seq_length,
            ).to(self.device)
            outputs = self.model(**inputs)
            # bf16/fp16 logits aren't numpy-convertible directly
            yield outputs.logits.float()

    def predict(self, X, batch_size: int = 32):
        """For classification: returns predicted class indices. For regression: returns predicted values."""
        import torch
        import numpy as np

        if self.task_type == 'classification':
            # Use predict_proba for consistency
            probs = self.predict_proba(X, batch_size=batch_size)
            predictions = np.argmax(probs, axis=1)

            # Diagnostic stats - check BEFORE decoding
            unique_preds_raw, counts_raw = np.unique(predictions, return_counts=True)
            logger.debug("[%s] Raw prediction indices: %s", self.__class__.__name__, dict(zip(unique_preds_raw, counts_raw)))
            if len(unique_preds_raw) == 1:
                logger.warning("All %d predictions are class index %s", len(predictions), unique_preds_raw[0])
                logger.debug("   Sample probabilities (first 3): %s", probs[:3])

            # Decode predictions if label encoder exists
            if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                try:
                    predictions = self.label_encoder.inverse_transform(predictions)
                    # Diagnostic stats after decoding
                    unique_preds_decoded, counts_decoded = np.unique(predictions, return_counts=True)
                    logger.debug("   Decoded predictions: %s", dict(zip(unique_preds_decoded, counts_decoded)))
                except Exception as e:
                    logger.warning("Label decoding failed: %s", e)

            return predictions
        else:
            # Regression
            self.model.eval()
            predictions = []

            # Handle both single text and list of texts
            if isinstance(X, str):
                X = [X]

            with torch.no_grad():
                for logits in self._batched_logits(X, batch_size=batch_size):
                    predictions.append(logits.squeeze(-1).cpu().numpy())

            result = np.concatenate(predictions) if predictions else np.array([])
            # Diagnostic stats
            logger.debug("[%s] Predictions - min: %.4f, max: %.4f, mean: %.4f, std: %.4f", self.__class__.__name__, result.min(), result.max(), result.mean(), result.std())
            return result

    def predict_proba(self, X, batch_size: int = 32):
        """
        Predict class probabilities for text inputs.
        Returns probabilities for each class (shape: [n_samples, n_classes]).
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        import torch
        import numpy as np

        self.model.eval()
        all_probs = []

        # Handle both single text and list of texts
        if isinstance(X, str):
            X = [X]

        with torch.no_grad():
            for logits in self._batched_logits(X, batch_size=batch_size):
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        # Concatenate batches into a single (n_samples, n_classes) array
        result = np.concatenate(all_probs, axis=0)

        # Diagnostic stats
        predicted_classes = np.argmax(result, axis=1)
        unique_preds, counts = np.unique(predicted_classes, return_counts=True)
        logger.debug("[%s] predict_proba class distribution: %s", self.__class__.__name__, dict(zip(unique_preds, counts)))
        if len(unique_preds) == 1:
            logger.warning("All %d probabilities lead to class %s", len(result), unique_preds[0])
            logger.debug("   Sample probabilities (first 3): %s", result[:3])
        logger.debug("   Probability stats - min: %.4f, max: %.4f, mean: %.4f", result.min(), result.max(), result.mean())

        return result

    def __call__(self, X):
        """
        Returns raw logits for compatibility with pipeline evaluation.
        Handles both single text inputs and lists of texts.
        """
        import torch
        self.model.eval()

        with torch.no_grad():
            if isinstance(X, list):
                # Handle batch
                all_logits = []
                for text in X:
                    inputs = self._tokenize(text, return_tensors='pt', truncation=True,
                                            padding='max_length', max_length=self.max_seq_length).to(self.device)
                    outputs = self.model(**inputs)
                    all_logits.append(outputs.logits)
                return torch.cat(all_logits, dim=0)
            else:
                # Single input
                inputs = self._tokenize(X, return_tensors='pt', truncation=True,
                                        padding='max_length', max_length=self.max_seq_length).to(self.device)
                outputs = self.model(**inputs)
                return outputs.logits

    def forward(self, x):
        """
        Forward pass for nn.Module compatibility.
        Note: This is not typically used for text models. Use the fit/predict methods instead.
        For compatibility with GeneralPipeline, this method is provided but should not be called directly.
        """
        # This method exists for nn.Module compatibility but isn't used in practice
        # The actual forward pass happens through the HuggingFace model via fit/predict
        raise NotImplementedError(
            "HuggingFaceQLoRAWrapper uses HuggingFace Trainer for training. "
            "Use the fit() method for training and predict()/predict_proba() for inference."
        )

