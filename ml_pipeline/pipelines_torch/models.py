import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Callable, Any, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

logger = logging.getLogger(__name__)

# --- MLPs ---
class TorchMLP(nn.Module):
    """
    Configurable MLP for classification or regression.
    """
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64], output_dim: int = 1, dropout: float = 0.3, batchnorm: bool = False):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class TorchMLPClassifier(TorchMLP):
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64], num_classes: int = 2, dropout: float = 0.3, batchnorm: bool = False):
        super().__init__(input_dim, hidden_dims, num_classes, dropout, batchnorm)

class TorchMLPRegressor(TorchMLP):
    def __init__(self, input_dim: int, output_dim: int = 1, hidden_dims: list = [256, 128, 64], dropout: float = 0.3, batchnorm: bool = False):
        super().__init__(input_dim, hidden_dims, output_dim, dropout, batchnorm)

# --- Sklearn Model Wrapper Base ---
class SklearnModelWrapper:
    """Base wrapper making any sklearn model compatible with GeneralPipeline."""

    _PIPELINE_ONLY_KWARGS = {
        "input_dim",
        "num_classes",
        "output_dim",
        "device",
        "task_type",
    }

    def __init__(self, sklearn_cls, **kwargs):
        kwargs = self._filter_sklearn_kwargs(sklearn_cls, kwargs)
        self.model = sklearn_cls(**kwargs)

    @classmethod
    def _filter_sklearn_kwargs(cls, sklearn_cls, kwargs):
        model_name = getattr(sklearn_cls, "__name__", str(sklearn_cls))
        valid_params = set(sklearn_cls().get_params().keys())
        filtered = {
            key: value
            for key, value in kwargs.items()
            if key in valid_params
        }
        ignored = sorted(set(kwargs) - set(filtered))
        unexpected = [key for key in ignored if key not in cls._PIPELINE_ONLY_KWARGS]
        if unexpected:
            logger.warning("Ignoring unsupported %s kwargs: %s", model_name, unexpected)
        return filtered

    def to(self, device):
        return self

    def eval(self):
        pass

    def train(self):
        pass

    def fit(self, X, y, *args, **kwargs):
        X, y = self._to_numpy(X), self._to_numpy(y)
        sample_weight = kwargs.get('sample_weight')
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)

    def predict(self, X):
        X = self._to_numpy(X)
        predictions = self.model.predict(X)
        logger.debug("[%s] predict shape=%s", self.__class__.__name__, np.asarray(predictions).shape)
        return predictions

    def predict_proba(self, X):
        X = self._to_numpy(X)
        probs = self.model.predict_proba(X)
        logger.debug("[%s] predict_proba shape=%s", self.__class__.__name__, np.asarray(probs).shape)
        return probs

    def __call__(self, X):
        X = self._to_numpy(X)
        if hasattr(self.model, 'predict_proba'):
            return torch.tensor(self.model.predict_proba(X), dtype=torch.float32)
        return torch.tensor(self.model.predict(X), dtype=torch.float32)

    @staticmethod
    def _to_numpy(data):
        return data.cpu().numpy() if isinstance(data, torch.Tensor) else data


class SklearnRandomForestClassifierWrapper(SklearnModelWrapper):
    def __init__(self, **kwargs):
        super().__init__(RandomForestClassifier, **kwargs)


class SklearnRandomForestRegressorWrapper(SklearnModelWrapper):
    def __init__(self, **kwargs):
        super().__init__(RandomForestRegressor, **kwargs)


class XGBoostClassifierWrapper(SklearnModelWrapper):
    def __init__(self, **kwargs):
        super().__init__(xgb.XGBClassifier, **kwargs)


class XGBoostRegressorWrapper(SklearnModelWrapper):
    def __init__(self, **kwargs):
        super().__init__(xgb.XGBRegressor, **kwargs)


class LightGBMClassifierWrapper(SklearnModelWrapper):
    def __init__(self, **kwargs):
        super().__init__(lgb.LGBMClassifier, **kwargs)


class LightGBMRegressorWrapper(SklearnModelWrapper):
    """LightGBM regressor wrapped in MultiOutputRegressor for multi-target support."""

    def __init__(self, **kwargs):
        super().__init__(lgb.LGBMRegressor, **kwargs)
        # Re-wrap the inner model with MultiOutputRegressor
        self.model = MultiOutputRegressor(estimator=self.model)

    def fit(self, X, y, *args, **kwargs):
        X, y = self._to_numpy(X), self._to_numpy(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.model.fit(X, y)

    def predict(self, X):
        X = self._to_numpy(X)
        preds = self.model.predict(X)
        logger.debug("[%s] predict shape=%s", self.__class__.__name__, np.asarray(preds).shape)
        return torch.tensor(preds, dtype=torch.float32)

class TabICLClassifierWrapper(SklearnModelWrapper):
    def __init__(self, **kwargs):
        from tabicl import TabICLClassifier
        super().__init__(TabICLClassifier, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        kwargs.pop('sample_weight', None)
        super().fit(X, y, *args, **kwargs)


class TabICLRegressorWrapper(SklearnModelWrapper):
    def __init__(self, **kwargs):
        from tabicl import TabICLRegressor
        super().__init__(TabICLRegressor, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        kwargs.pop('sample_weight', None)
        super().fit(X, y, *args, **kwargs)


class LlamaCppClassifier:
    """
    Adapter-style classifier/regressor for llama.cpp models.
    Trains a dense layer on top of frozen llama.cpp outputs for classification or regression.
    """
    def __init__(self, gguf_path, num_classes=2, task_type='classification', device="cpu"):
        from llama_cpp import Llama
        self.llama = Llama(model_path=gguf_path, n_ctx=2048, logits_all=True)
        self.task_type = task_type
        # You must know the hidden size of the model (e.g., 8192 for 27B)
        self.hidden_size = self.llama.config["hidden_size"]
        import torch
        import torch.nn as nn
        
        if task_type == 'classification':
            self.classifier = nn.Linear(self.hidden_size, num_classes).to(device)
        else:  # regression
            self.classifier = nn.Linear(self.hidden_size, 1).to(device)
        self.device = device

    def _get_logits(self, prompt):
        import torch
        # Get llama.cpp output (logits for each token)
        output = self.llama(prompt)
        # Use the last token's logits as the representation
        last_logits = torch.tensor(output["logits"][-1]).to(self.device)
        return last_logits

    def fit(self, prompts, labels, epochs=3, lr=1e-3):
        """
        Fine-tune the classifier/regressor head on top of llama.cpp outputs.
        Only the classifier head is trained.
        """
        import torch
        import torch.optim as optim
        import torch.nn as nn
        self.classifier.train()
        optimizer = optim.Adam(self.classifier.parameters(), lr=lr)
        
        if self.task_type == 'classification':
            loss_fn = nn.CrossEntropyLoss()
        else:  # regression
            loss_fn = nn.MSELoss()
            
        for epoch in range(epochs):
            total_loss = 0
            for prompt, label in zip(prompts, labels):
                logits = self._get_logits(prompt)
                pred = self.classifier(logits.unsqueeze(0))
                
                if self.task_type == 'classification':
                    loss = loss_fn(pred, torch.tensor([label]).to(self.device))
                else:  # regression
                    loss = loss_fn(pred.squeeze(), torch.tensor([label], dtype=torch.float32).to(self.device))
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info("Epoch %d loss: %.4f", epoch + 1, total_loss / len(prompts))

    def predict(self, prompts):
        """
        For classification: predict class indices.
        For regression: predict continuous values.
        """
        import torch
        self.classifier.eval()
        preds = []
        with torch.no_grad():
            for prompt in prompts:
                logits = self._get_logits(prompt)
                pred = self.classifier(logits.unsqueeze(0))
                if self.task_type == 'classification':
                    preds.append(pred.argmax(dim=-1).item())
                else:  # regression
                    preds.append(pred.squeeze().item())
        
        # Diagnostic stats
        preds_array = np.array(preds)
        if self.task_type == 'classification':
            unique_preds, counts = np.unique(preds_array, return_counts=True)
            logger.debug("[%s] Prediction distribution: %s", self.__class__.__name__, dict(zip(unique_preds, counts)))
        else:
            logger.debug("[%s] Predictions - min: %.4f, max: %.4f, mean: %.4f, std: %.4f", self.__class__.__name__, preds_array.min(), preds_array.max(), preds_array.mean(), preds_array.std())
        
        return preds

    def predict_proba(self, prompts):
        """
        Predict class probabilities for classification tasks only.
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        import torch
        self.classifier.eval()
        probs = []
        with torch.no_grad():
            for prompt in prompts:
                logits = self._get_logits(prompt)
                pred = self.classifier(logits.unsqueeze(0))
                prob = torch.softmax(pred, dim=-1).cpu().numpy()[0]
                probs.append(prob)
        
        # Diagnostic stats
        probs_array = np.array(probs)
        predicted_classes = np.argmax(probs_array, axis=1)
        unique_preds, counts = np.unique(predicted_classes, return_counts=True)
        logger.debug("[%s] predict_proba class distribution: %s", self.__class__.__name__, dict(zip(unique_preds, counts)))
        logger.debug("   Probability stats - min: %.4f, max: %.4f, mean: %.4f", probs_array.min(), probs_array.max(), probs_array.mean())
        
        return probs

class HuggingFaceQLoRAWrapper(nn.Module):
    """
    Wrapper for QLoRA (Quantized LoRA) fine-tuning with HuggingFace models.
    Supports both classification and regression tasks.
    Compatible with standard encoder models (BERT family, ModernBERT) and causal /
    hybrid-architecture decoder models (Qwen3.5, Gemma 4, Llama, Mistral, etc.).
    Based on the guidelines from https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora

    Note: This wrapper inherits from nn.Module to be compatible with GeneralPipeline,
    but training is handled by HuggingFace Trainer, not by the standard PyTorch training loop.
    """
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
                 gradient_accumulation_steps: int = 2):
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
            logger.info("pad_token not set — using eos_token as pad_token for causal LM.")

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

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)

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

        # Apply LoRA — fall back to explicit module names if "all-linear" fails
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
        self.max_seq_length = 512  # Default, can be overridden in fit()

    def _sync_special_token_ids_with_tokenizer(self):
        for attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
            token_id = getattr(self.tokenizer, attr, None)
            if hasattr(self.model, "config"):
                setattr(self.model.config, attr, token_id)
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
        
    def fit(self, X, y, *args, **kwargs):
        """
        Fine-tune the model using QLoRA.
        X: list of texts, y: labels/targets
        Returns: training_history (list of dicts with metrics per epoch)
        """
        from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback
        from datasets import Dataset
        import torch
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        
        # Convert labels to proper format - handle string labels
        if isinstance(y, (list, np.ndarray)):
            if hasattr(y, 'to_numpy'):
                y = y.to_numpy()
            y = np.array(y).flatten()
            
            # Check if labels are strings and encode them
            if y.dtype == object or isinstance(y[0], str):
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                # Store the encoder for later use in predict
                self.label_encoder = label_encoder
                logger.info("Label encoding: %s", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
            
            y = y.astype(int).tolist()
        elif hasattr(y, 'tolist'):
            y = y.tolist()
            # Check if labels are strings
            if isinstance(y[0], str):
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y).tolist()
                self.label_encoder = label_encoder
                logger.info("Label encoding: %s", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
            # Flatten if nested
            elif isinstance(y[0], (list, np.ndarray)):
                y = [int(item[0] if isinstance(item, (list, np.ndarray)) else item) for item in y]
        
        # Prepare dataset for sequence classification (not conversation format)
        dataset = Dataset.from_dict({'text': list(X), 'label': y})  # Use 'label' not 'labels' initially
        
        # Store max_seq_length for use in predict()
        self.max_seq_length = kwargs.get('max_seq_length', 512)
        
        # Tokenize the dataset - DataCollatorWithPadding expects 'label' not 'labels'
        def tokenize_fn(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples['text'], 
                truncation=True, 
                padding=False,  # Don't pad here, let DataCollator handle it
                max_length=self.max_seq_length
            )
            # Keep the label (DataCollator will rename to labels)
            tokenized['label'] = examples['label']
            return tokenized
        
        dataset = dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
        # Don't set format yet - let the data collator handle it
        
        # Determine dtype for training based on GPU capability
        try:
            compute_capability = torch.cuda.get_device_capability()[0]
            use_bf16 = compute_capability >= 8
            use_fp16 = compute_capability < 8
        except Exception:
            # Fallback for older GPUs
            use_bf16 = False
            use_fp16 = True
        
        # Disable fp16/bf16 if using quantization to avoid gradient scaling issues
        # Quantized models already use mixed precision internally
        if self._is_quantized:
            use_bf16 = False
            use_fp16 = False
        
        # Training configuration - use standard TrainingArguments for sequence classification
        num_epochs = kwargs.get('epochs', 2)
        training_args = TrainingArguments(
            output_dir=kwargs.get('output_dir', './qlora_results'),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=kwargs.get('batch_size', 1),
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', self._gradient_accumulation_steps),
            gradient_checkpointing=True,
            # use_reentrant=False is required for causal LMs and hybrid architectures
            # (Qwen3.5, Gemma 4, Llama) to avoid autograd graph issues
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch_fused",
            logging_steps=10,
            learning_rate=kwargs.get('learning_rate', 1e-4),
            fp16=use_fp16,
            bf16=use_bf16,
            max_grad_norm=0.3,
            warmup_steps=0,
            lr_scheduler_type="constant",
            report_to="none",
            save_strategy="no",
        )
        
        # Create data collator for dynamic padding
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors='pt'
        )
        
        # Create a custom callback to track training history
        training_history = []
        
        class MetricsCallback(TrainerCallback):
            def on_epoch_end(self, args, state, control, **callback_kwargs):
                # Get the latest log entry (contains loss)
                if state.log_history:
                    latest_log = state.log_history[-1]
                    epoch_metrics = {
                        'epoch': state.epoch,
                        'loss': latest_log.get('loss', 0.0),
                    }
                    training_history.append(epoch_metrics)
                    logger.info("Epoch %d/%d - Loss: %.4f", int(state.epoch), num_epochs, epoch_metrics['loss'])
        
        # Create trainer with standard Trainer (not SFTTrainer)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,  # Use processing_class instead of tokenizer
            data_collator=data_collator,
            callbacks=[MetricsCallback()],
        )
        
        # Train
        trainer.train()
        
        # Return training history for compatibility with benchmark runner
        return training_history
        
    def eval(self):
        self.model.eval()
        
    def train(self):
        self.model.train()
        
    def predict(self, X):
        """For classification: returns predicted class indices. For regression: returns predicted values."""
        import torch
        import numpy as np
        
        if self.task_type == 'classification':
            # Use predict_proba for consistency
            probs = self.predict_proba(X)
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
            # Regression - process directly one at a time
            self.model.eval()
            predictions = []
            
            # Handle both single text and list of texts
            if isinstance(X, str):
                X = [X]
            
            # Process one at a time to avoid padding token issues
            with torch.no_grad():
                for text in X:
                    inputs = self.tokenizer(
                        text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=self.max_seq_length
                    ).to(self.device)
                    
                    outputs = self.model(**inputs)
                    logits = outputs.logits.squeeze().cpu().numpy()
                    predictions.append(logits)
            
            result = np.array(predictions)
            # Diagnostic stats
            logger.debug("[%s] Predictions - min: %.4f, max: %.4f, mean: %.4f, std: %.4f", self.__class__.__name__, result.min(), result.max(), result.mean(), result.std())
            return result
                
    def predict_proba(self, X):
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
        
        # Process one at a time to avoid padding token issues
        with torch.no_grad():
            for text in X:
                # Tokenize single text (no padding needed for batch_size=1)
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.max_seq_length
                ).to(self.device)
                
                # Get predictions
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Convert to probabilities (squeeze batch dimension)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                all_probs.append(probs)
        
        # Stack all predictions into a numpy array
        result = np.stack(all_probs, axis=0)
        
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
                    inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                                            padding='max_length', max_length=self.max_seq_length).to(self.device)
                    outputs = self.model(**inputs)
                    all_logits.append(outputs.logits)
                return torch.cat(all_logits, dim=0)
            else:
                # Single input
                inputs = self.tokenizer(X, return_tensors='pt', truncation=True,
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




# --- Registry ---
CLASSIFICATION_MODEL_REGISTRY: Dict[str, Callable] = {
    "mlp_classifier": TorchMLPClassifier,
    "deep_mlp_classifier": lambda input_dim, num_classes=2: TorchMLPClassifier(input_dim, [512, 256, 128, 64], num_classes=num_classes, dropout=0.3, batchnorm=True),
    "random_forest_classifier": SklearnRandomForestClassifierWrapper,
    "xgboost_classifier": XGBoostClassifierWrapper,
    "lightgbm_classifier": LightGBMClassifierWrapper,
    "hf_qlora_classifier": lambda **kwargs: HuggingFaceQLoRAWrapper(task_type='classification', **kwargs),
    "llama_cpp_classifier": lambda **kwargs: LlamaCppClassifier(task_type='classification', **kwargs),
    "tabicl_classifier": TabICLClassifierWrapper,
}

REGRESSION_MODEL_REGISTRY: Dict[str, Callable] = {
    "mlp_regressor": TorchMLPRegressor,
    "deep_mlp_regressor": lambda input_dim, output_dim: TorchMLPRegressor(input_dim, output_dim, [512, 256, 128, 64], dropout=0.3, batchnorm=True),
    "random_forest_regressor": SklearnRandomForestRegressorWrapper,
    "xgboost_regressor": XGBoostRegressorWrapper,
    "lightgbm_regressor": LightGBMRegressorWrapper,
    "hf_qlora_regressor": lambda **kwargs: HuggingFaceQLoRAWrapper(task_type='regression', **kwargs),
    "llama_cpp_regressor": lambda **kwargs: LlamaCppClassifier(task_type='regression', **kwargs),
    "tabicl_regressor": TabICLRegressorWrapper,
}

# Combined registry for backward compatibility
MODEL_REGISTRY: Dict[str, Callable] = {
    **CLASSIFICATION_MODEL_REGISTRY,
    **REGRESSION_MODEL_REGISTRY,
}
