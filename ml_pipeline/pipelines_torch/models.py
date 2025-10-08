import torch
import torch.nn as nn
from typing import Dict, Callable, Any, Optional, Sequence
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

from third_party import load_class, resolve_repo_path

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

# --- RandomForest Wrapper ---
class SklearnRandomForestClassifierWrapper:
    """
    Wrapper for sklearn RandomForestClassifier to be compatible with GeneralPipeline.
    """
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)
    def to(self, device):
        return self  # For compatibility
    def fit(self, X, y, *args, **kwargs):
        self.model.fit(X, y)
    def eval(self):
        pass
    def train(self):
        pass
    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        return self.model.predict(X)
    def predict_proba(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        return self.model.predict_proba(X)
    def __call__(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        proba = self.model.predict_proba(X)
        return torch.tensor(proba, dtype=torch.float32)

# --- RandomForest Regressor Wrapper ---
class SklearnRandomForestRegressorWrapper:
    """
    Wrapper for sklearn RandomForestRegressor to be compatible with GeneralPipelineSklearn for regression.
    """
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)
    def to(self, device):
        return self  # For compatibility
    def fit(self, X, y, *args, **kwargs):
        self.model.fit(X, y)
    def eval(self):
        pass
    def train(self):
        pass
    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        return self.model.predict(X)
    def __call__(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        preds = self.model.predict(X)
        return torch.tensor(preds, dtype=torch.float32)

# --- XGBoost Wrappers ---
class XGBoostClassifierWrapper:
    """
    Wrapper for XGBoost Classifier to be compatible with GeneralPipeline.
    """
    def __init__(self, **kwargs):
        self.model = xgb.XGBClassifier(**kwargs)
    
    def to(self, device):
        return self  # For compatibility
    
    def fit(self, X, y, *args, **kwargs):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        sample_weight = kwargs.get('sample_weight', None)
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
    
    def eval(self):
        pass
    
    def train(self):
        pass
    
    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        return self.model.predict_proba(X)
    
    def __call__(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        proba = self.model.predict_proba(X)
        return torch.tensor(proba, dtype=torch.float32)

class XGBoostRegressorWrapper:
    """
    Wrapper for XGBoost Regressor to be compatible with GeneralPipeline.
    """
    def __init__(self, **kwargs):
        self.model = xgb.XGBRegressor(**kwargs)
    
    def to(self, device):
        return self  # For compatibility
    
    def fit(self, X, y, *args, **kwargs):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        self.model.fit(X, y)
    
    def eval(self):
        pass
    
    def train(self):
        pass
    
    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        return self.model.predict(X)
    
    def __call__(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        preds = self.model.predict(X)
        return torch.tensor(preds, dtype=torch.float32)

# --- LightGBM Wrappers ---
class LightGBMClassifierWrapper:
    """
    Wrapper for LightGBM Classifier to be compatible with GeneralPipeline.
    """
    def __init__(self, **kwargs):
        self.model = lgb.LGBMClassifier(**kwargs)
    
    def to(self, device):
        return self  # For compatibility
    
    def fit(self, X, y, *args, **kwargs):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        sample_weight = kwargs.get('sample_weight', None)
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
    
    def eval(self):
        pass
    
    def train(self):
        pass
    
    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        return self.model.predict_proba(X)
    
    def __call__(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        proba = self.model.predict_proba(X)
        return torch.tensor(proba, dtype=torch.float32)


class LightGBMRegressorWrapper:
    """
    Wrapper for LightGBM Regressor to be compatible with GeneralPipeline.
    Uses MultiOutputRegressor to handle multi-target regression.
    """
    def __init__(self, **kwargs):
        # Instantiate the base LGBMRegressor with any provided arguments
        base_estimator = lgb.LGBMRegressor(**kwargs)
        # Wrap it with MultiOutputRegressor to handle multiple outputs
        self.model = MultiOutputRegressor(estimator=base_estimator)
    
    def to(self, device):
        """
        Compatibility method for the pipeline. Does nothing for scikit-learn models.
        """
        return self
    
    def fit(self, X, y, *args, **kwargs):
        """
        Fits the model. Converts tensors to numpy arrays if necessary.
        Handles 1D labels by reshaping them to 2D for MultiOutputRegressor.
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        
        # MultiOutputRegressor requires 2D labels (n_samples, n_outputs)
        # If labels are 1D, reshape to 2D with shape (n_samples, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        self.model.fit(X, y)

    def predict(self, X):
        """
        Makes predictions. Converts input tensor to numpy and output back to tensor.
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        preds = self.model.predict(X)
        
        return torch.tensor(preds, dtype=torch.float32)

    def __call__(self, X):
        """
        Allows the instance to be called like a function, e.g., model(X).
        """
        return self.predict(X)
# --- RandomForest Regressor Wrapper ---
class SklearnRandomForestRegressorWrapper:
    """
    Wrapper for sklearn RandomForestRegressor to be compatible with GeneralPipeline for regression.
    """
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)
    def to(self, device):
        return self  # For compatibility
    def fit(self, X, y, *args, **kwargs):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        self.model.fit(X, y)
    def eval(self):
        pass
    def train(self):
        pass
    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        return self.model.predict(X)
    def __call__(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        preds = self.model.predict(X)
        return torch.tensor(preds, dtype=torch.float32)

# --- LoRA/PEFT HuggingFace Wrapper ---
class HuggingFaceLoRAWrapper:
    """
    Wrapper for any HuggingFace model with LoRA/PEFT fine-tuning.
    Supports both classification and regression tasks.
    Accepts model_name, tokenizer_name, and LoRA config.
    Note: Input must be tokenized text.
    """
    def __init__(self, model_name: str, tokenizer_name: Optional[str] = None, lora_config: Optional[Dict[str, Any]] = None, 
                 num_labels: int = 2, task_type: str = 'classification', device: str = 'cpu'):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType
        
        self.task_type = task_type
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        
        if task_type == 'regression':
            num_labels = 1
            
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        
        if lora_config is None:
            lora_config = {
                'r': 8,
                'lora_alpha': 16,
                'target_modules': ["q_proj", "v_proj"],
                'lora_dropout': 0.05,
                'bias': "none",
                'task_type': TaskType.SEQ_CLS
            }
        lora_cfg = LoraConfig(**lora_config)
        self.model = get_peft_model(self.model, lora_cfg)
        self.device = device
        self.model.to(device)
        
    def to(self, device):
        self.device = device
        self.model.to(device)
        return self
        
    def fit(self, X, y, *args, **kwargs):
        # X: list of texts, y: labels/targets
        from transformers import TrainingArguments, Trainer
        import torch
        from datasets import Dataset
        ds = Dataset.from_dict({'text': X, 'labels': y})
        def tokenize_fn(batch):
            return self.tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)
        ds = ds.map(tokenize_fn, batched=True)
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=kwargs.get('epochs', 2),
            per_device_train_batch_size=kwargs.get('batch_size', 8),
            logging_steps=10,
            report_to='none',
            fp16=True if self.device == 'cuda' else False
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=ds,
            eval_dataset=None,
            tokenizer=self.tokenizer
        )
        trainer.train()
        
    def eval(self):
        self.model.eval()
        
    def train(self):
        self.model.train()
        
    def predict(self, X):
        """For classification: returns predicted class indices. For regression: returns predicted values."""
        import torch
        inputs = self.tokenizer(X, return_tensors='pt', truncation=True, padding='max_length', max_length=256).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            if self.task_type == 'classification':
                return logits.argmax(dim=-1).cpu().numpy()
            else:  # regression
                return logits.squeeze().cpu().numpy()
                
    def predict_proba(self, X):
        """Only for classification tasks."""
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        import torch
        inputs = self.tokenizer(X, return_tensors='pt', truncation=True, padding='max_length', max_length=256).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            return probs.cpu().numpy()
        
    def __call__(self, X):
        # X: list of texts
        import torch
        inputs = self.tokenizer(X, return_tensors='pt', truncation=True, padding='max_length', max_length=256).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        return logits

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
            print(f"Epoch {epoch+1} loss: {total_loss/len(prompts):.4f}")

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
        return probs

class HuggingFaceQLoRAWrapper:
    """
    Wrapper for QLoRA (Quantized LoRA) fine-tuning with HuggingFace models.
    Supports both classification and regression tasks.
    Based on the guidelines from https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora
    """
    def __init__(self, model_name: str, tokenizer_name: Optional[str] = None, 
                 num_labels: int = 2, task_type: str = 'classification', device: str = 'cpu',
                 lora_config: Optional[Dict[str, Any]] = None, quantization_config: Optional[Dict[str, Any]] = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType
        import torch
        
        self.task_type = task_type
        self.device = device
        
        # Set up tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        
        # Adjust num_labels for regression
        if task_type == 'regression':
            num_labels = 1
            
        # Force GPU usage if available, raise error if not
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU is required for HuggingFaceQLoRAWrapper but CUDA is not available. "
                "Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support installed."
            )
        
        # Determine dtype based on GPU compute capability
        try:
            # Check compute capability for bfloat16 support (Ampere and newer = 8.0+)
            compute_capability = torch.cuda.get_device_capability()[0]
            torch_dtype = torch.bfloat16 if compute_capability >= 8 else torch.float16
            print(f"Using GPU with compute capability {compute_capability}.x - dtype: {torch_dtype}")
        except Exception:
            # Default to float16 for older GPUs
            torch_dtype = torch.float16
            print(f"Using GPU with default dtype: {torch_dtype}")
            
        # Try to set up quantization config with proper error handling
        bnb_config = None
        if quantization_config is None:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',  # NF4 quantization (better than FP4 for many models)
                    bnb_4bit_use_double_quant=True,  # Double quantization for additional memory savings
                    bnb_4bit_compute_dtype=torch_dtype,
                )
                print("4-bit quantization enabled with NF4")
            except Exception as e:
                print(f"Warning: Failed to setup quantization config: {e}")
                print("Falling back to non-quantized model loading on GPU.")
                bnb_config = None
        elif quantization_config is not None:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(**quantization_config)
                print("Custom quantization config applied")
            except Exception as e:
                print(f"Warning: Failed to setup custom quantization config: {e}")
                print("Falling back to non-quantized model loading on GPU.")
                bnb_config = None
            
        # Prepare model loading kwargs - always use GPU
        model_kwargs = {
            "num_labels": num_labels,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "device_map": "auto",  # Always use auto device mapping for GPU
        }
        
        # Add quantization config if available
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        
        # Load model with quantization or fallback
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Default LoRA config
        if lora_config is None:
            lora_config = {
                'lora_alpha': 16,
                'lora_dropout': 0.05,
                'r': 16,
                'bias': "none",
                'target_modules': "all-linear",
                'task_type': TaskType.SEQ_CLS,
                'modules_to_save': ["classifier"]  # Save classifier head
            }
        
        # Apply LoRA
        lora_cfg = LoraConfig(**lora_config)
        self.model = get_peft_model(self.model, lora_cfg)
        
    def to(self, device):
        self.device = device
        # Model is already on device due to device_map="auto"
        return self
        
    def fit(self, X, y, *args, **kwargs):
        """
        Fine-tune the model using QLoRA.
        X: list of texts, y: labels/targets
        """
        from transformers import TrainingArguments, Trainer
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
        import torch
        
        # Prepare dataset
        if self.task_type == 'classification':
            # For classification, format as conversation
            def create_conversation(text, label):
                return {
                    "messages": [
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": str(label)}
                    ]
                }
            formatted_data = [create_conversation(text, label) for text, label in zip(X, y)]
        else:
            # For regression, similar format but with continuous values
            def create_conversation(text, target):
                return {
                    "messages": [
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": str(target)}
                    ]
                }
            formatted_data = [create_conversation(text, target) for text, target in zip(X, y)]
            
        dataset = Dataset.from_list(formatted_data)
        
        # Determine dtype for training based on GPU capability
        try:
            compute_capability = torch.cuda.get_device_capability()[0]
            use_bf16 = compute_capability >= 8
            use_fp16 = compute_capability < 8
        except Exception:
            # Fallback for older GPUs
            use_bf16 = False
            use_fp16 = True
        
        # Training configuration
        args = SFTConfig(
            output_dir=kwargs.get('output_dir', './qlora_results'),
            max_seq_length=kwargs.get('max_seq_length', 512),
            packing=True,
            num_train_epochs=kwargs.get('epochs', 3),
            per_device_train_batch_size=kwargs.get('batch_size', 1),
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 4),
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            logging_steps=10,
            learning_rate=kwargs.get('learning_rate', 2e-4),
            fp16=use_fp16,
            bf16=use_bf16,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            report_to="none",
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": True,
            }
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
            processing_class=self.tokenizer
        )
        
        # Train
        trainer.train()
        
    def eval(self):
        self.model.eval()
        
    def train(self):
        self.model.train()
        
    def predict(self, X):
        """For classification: returns predicted class indices. For regression: returns predicted values."""
        import torch
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in X:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                      padding='max_length', max_length=256).to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                if self.task_type == 'classification':
                    pred = logits.argmax(dim=-1).cpu().numpy()[0]
                else:  # regression
                    pred = logits.squeeze().cpu().numpy()
                    if pred.ndim == 0:  # scalar
                        pred = pred.item()
                predictions.append(pred)
                
        return predictions
                
    def predict_proba(self, X):
        """Only for classification tasks."""
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        import torch
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for text in X:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                      padding='max_length', max_length=256).to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                probabilities.append(probs)
                
        return probabilities
        
    def __call__(self, X):
        """Returns raw logits."""
        import torch
        self.model.eval()

        with torch.no_grad():
            if isinstance(X, list):
                # Handle batch
                all_logits = []
                for text in X:
                    inputs = self.tokenizer(text, return_tensors='pt', truncation=True,
                                          padding='max_length', max_length=256).to(self.device)
                    outputs = self.model(**inputs)
                    all_logits.append(outputs.logits)
                return torch.cat(all_logits, dim=0)
            else:
                # Single input
                inputs = self.tokenizer(X, return_tensors='pt', truncation=True,
                                      padding='max_length', max_length=256).to(self.device)
                outputs = self.model(**inputs)
                return outputs.logits


class ThirdPartyTabularModel(nn.Module):
    """Generic nn.Module wrapper for third-party tabular research models."""

    def __init__(
        self,
        repo_name: str,
        class_name: str,
        *,
        repo_path: Optional[str] = None,
        env_var: Optional[str] = None,
        module_candidates: Optional[Sequence[str]] = None,
        init_args: Optional[Sequence] = None,
        init_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        init_args = tuple(init_args or ())
        init_kwargs = dict(init_kwargs or {})
        cls = load_class(
            repo_name,
            class_name,
            repo_path=repo_path,
            env_var=env_var,
            module_candidates=module_candidates,
        )
        self.inner = cls(*init_args, **init_kwargs)

    def forward(self, *args, **kwargs):  # type: ignore[override]
        return self.inner(*args, **kwargs)


class TabRWrapper(ThirdPartyTabularModel):
    """Wrapper around the TabR reference implementation (NeurIPS 2024)."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        repo_path: Optional[str] = None,
        env_var: str = "TABR_REPO",
        model_kwargs: Optional[dict] = None,
    ) -> None:
        nn.Module.__init__(self)

        from importlib.util import module_from_spec, spec_from_file_location

        repo_dir = resolve_repo_path(
            "tabular-dl-tabr", repo_path=repo_path, env_var=env_var
        )
        tabr_entry = repo_dir / "bin" / "tabr.py"
        if not tabr_entry.exists():
            raise FileNotFoundError(
                f"Unable to locate TabR entry point at '{tabr_entry}'."
            )

        spec = spec_from_file_location("third_party.tabr", tabr_entry)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create import spec for '{tabr_entry}'.")

        module = module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore[arg-type]
        except ImportError as exc:
            raise ImportError(
                "TabR dependencies are missing. Install the requirements listed in "
                "tabular-dl-tabr/environment-simple.yaml (faiss, delu, etc.)."
            ) from exc

        ModelCls = getattr(module, "Model", None)
        if ModelCls is None:
            raise ImportError("'Model' class not found in bin/tabr.py")

        kwargs = dict(model_kwargs or {})
        n_num_features = kwargs.pop("n_num_features", input_dim)
        n_bin_features = kwargs.pop("n_bin_features", 0)
        cat_cardinalities = list(kwargs.pop("cat_cardinalities", []))
        n_classes = kwargs.pop(
            "n_classes", num_classes if num_classes > 1 else None
        )

        defaults = {
            "num_embeddings": None,
            "d_main": 128,
            "d_multiplier": 2.0,
            "encoder_n_blocks": 2,
            "predictor_n_blocks": 2,
            "mixer_normalization": "auto",
            "context_dropout": 0.0,
            "dropout0": 0.1,
            "dropout1": "dropout0",
            "normalization": "LayerNorm",
            "activation": "ReLU",
            "memory_efficient": False,
            "candidate_encoding_batch_size": None,
        }
        for key, value in defaults.items():
            kwargs.setdefault(key, value)

        self.context_size = int(max(1, kwargs.pop("context_size", 32)))
        self._candidate_x = self._normalize_feature_dict(
            kwargs.pop("candidate_x", None)
        )
        self._candidate_y = kwargs.pop("candidate_y", None)
        self._is_classification = n_classes is not None

        self.inner = ModelCls(
            n_num_features=n_num_features,
            n_bin_features=n_bin_features,
            cat_cardinalities=cat_cardinalities,
            n_classes=n_classes,
            **kwargs,
        )

    @staticmethod
    def _normalize_feature_dict(
        value: Optional[Any],
    ) -> Optional[dict[str, torch.Tensor]]:
        if value is None:
            return None
        if isinstance(value, dict):
            return {k: v for k, v in value.items() if v is not None}
        if isinstance(value, torch.Tensor):
            return {"num": value}
        raise TypeError(
            "TabR candidate features must be provided as a Tensor or a dict of tensors."
        )

    def set_candidates(
        self,
        features: dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> None:
        self._candidate_x = self._normalize_feature_dict(features)
        self._candidate_y = targets

    def forward(self, x: Any) -> torch.Tensor:  # type: ignore[override]
        features = self._normalize_feature_dict(x)
        if features is None:
            raise ValueError("TabRWrapper requires feature tensors to be provided.")

        candidate_x = self._candidate_x or features
        candidate_y = self._candidate_y
        if candidate_y is None:
            representative = next(iter(candidate_x.values()))
            size = representative.shape[0]
            dtype = torch.long if self._is_classification else torch.float32
            candidate_y = torch.zeros(size, dtype=dtype, device=representative.device)

        context_size = max(1, min(self.context_size, candidate_y.shape[0]))

        return self.inner(
            x_=features,
            y=None,
            candidate_x_=candidate_x,
            candidate_y=candidate_y,
            context_size=context_size,
            is_train=False,
        )


class GrandeWrapper(ThirdPartyTabularModel):
    """Thin wrapper around the official GRANDE TensorFlow model."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        repo_path: Optional[str] = None,
        env_var: str = "GRANDE_REPO",
        model_kwargs: Optional[dict] = None,
    ) -> None:
        nn.Module.__init__(self)

        kwargs = dict(model_kwargs or {})
        params = kwargs.pop("params", {})
        args = kwargs.pop("args", {})

        params_defaults = {
            "depth": 5,
            "n_estimators": 512,
            "learning_rate_weights": 0.01,
            "learning_rate_index": 0.01,
            "learning_rate_values": 0.01,
            "learning_rate_leaf": 0.01,
            "optimizer": "adam",
            "cosine_decay_steps": 0,
            "loss": "crossentropy" if num_classes > 1 else "mse",
            "focal_loss": False,
            "temperature": 0.0,
            "from_logits": True,
            "use_class_weights": False,
            "dropout": 0.0,
            "selected_variables": 1.0,
            "data_subset_fraction": 1.0,
            "objective": "classification" if num_classes > 1 else "regression",
            "random_seed": 42,
        }
        params_defaults.update(params)

        args_defaults = {
            "epochs": 100,
            "early_stopping_epochs": 20,
            "batch_size": 64,
            "cat_idx": kwargs.pop("cat_idx", []),
            "objective": params_defaults["objective"],
            "random_seed": params_defaults["random_seed"],
            "verbose": 0,
        }
        args_defaults.update(args)

        GrandeCls = load_class(
            "grande",
            "GRANDE",
            repo_path=repo_path,
            env_var=env_var,
            module_candidates=("GRANDE.GRANDE", "GRANDE"),
        )

        params_defaults.setdefault("num_columns", list(range(input_dim)))
        params_defaults.setdefault("cat_idx", args_defaults["cat_idx"])

        self.inner = GrandeCls(params=params_defaults, args=args_defaults)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.training:
            raise RuntimeError("GrandeWrapper currently supports inference only.")

        if not isinstance(x, torch.Tensor):
            raise TypeError("GrandeWrapper expects a torch.Tensor input.")

        preds = self.inner.predict(x.detach().cpu().numpy())
        return torch.from_numpy(preds).to(x.device)


class TabMWrapper(ThirdPartyTabularModel):
    """Wrapper around the official TabM implementation."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        repo_path: Optional[str] = None,
        env_var: str = "TABM_REPO",
        model_kwargs: Optional[dict] = None,
    ) -> None:
        nn.Module.__init__(self)

        kwargs = dict(model_kwargs or {})

        tabm_cls = load_class(
            "tabm",
            "TabM",
            repo_path=repo_path,
            env_var=env_var,
            module_candidates=("tabm", "tabm.tabm"),
        )

        n_num_features = kwargs.pop("n_num_features", input_dim)
        cat_cardinalities = list(kwargs.pop("cat_cardinalities", []))
        d_out = kwargs.pop("d_out", num_classes if num_classes > 1 else None)
        num_embeddings = kwargs.pop("num_embeddings", None)

        defaults = {
            "k": 16,
            "n_blocks": 3,
            "d_block": 512,
            "dropout": 0.1,
            "activation": "ReLU",
            "arch_type": "tabm",
            "start_scaling_init": None,
        }
        for key, value in defaults.items():
            kwargs.setdefault(key, value)

        if kwargs.pop("use_make", True):
            self.inner = tabm_cls.make(
                n_num_features=n_num_features,
                cat_cardinalities=cat_cardinalities,
                d_out=d_out,
                num_embeddings=num_embeddings,
                **kwargs,
            )
        else:
            self.inner = tabm_cls(
                n_num_features=n_num_features,
                cat_cardinalities=cat_cardinalities,
                d_out=d_out,
                num_embeddings=num_embeddings,
                **kwargs,
            )


# --- Registry ---
CLASSIFICATION_MODEL_REGISTRY: Dict[str, Callable] = {
    "mlp_classifier": TorchMLPClassifier,
    "deep_mlp_classifier": lambda input_dim, num_classes=2: TorchMLPClassifier(input_dim, [512, 256, 128, 64], num_classes=num_classes, dropout=0.3, batchnorm=True),
    "random_forest_classifier": SklearnRandomForestClassifierWrapper,
    "xgboost_classifier": XGBoostClassifierWrapper,
    "lightgbm_classifier": LightGBMClassifierWrapper,
    "hf_lora_classifier": lambda **kwargs: HuggingFaceLoRAWrapper(task_type='classification', **kwargs),
    "hf_qlora_classifier": lambda **kwargs: HuggingFaceQLoRAWrapper(task_type='classification', **kwargs),
    "llama_cpp_classifier": lambda **kwargs: LlamaCppClassifier(task_type='classification', **kwargs),
    "tabr_classifier": TabRWrapper,
    "grande_classifier": GrandeWrapper,
    "tabm_classifier": TabMWrapper,
}

REGRESSION_MODEL_REGISTRY: Dict[str, Callable] = {
    "mlp_regressor": TorchMLPRegressor,
    "deep_mlp_regressor": lambda input_dim, output_dim: TorchMLPRegressor(input_dim, output_dim, [512, 256, 128, 64], dropout=0.3, batchnorm=True),
    "random_forest_regressor": SklearnRandomForestRegressorWrapper,
    "xgboost_regressor": XGBoostRegressorWrapper,
    "lightgbm_regressor": LightGBMRegressorWrapper,
    "hf_lora_regressor": lambda **kwargs: HuggingFaceLoRAWrapper(task_type='regression', **kwargs),
    "hf_qlora_regressor": lambda **kwargs: HuggingFaceQLoRAWrapper(task_type='regression', **kwargs),
    "llama_cpp_regressor": lambda **kwargs: LlamaCppClassifier(task_type='regression', **kwargs),
}

# Combined registry for backward compatibility
MODEL_REGISTRY: Dict[str, Callable] = {
    **CLASSIFICATION_MODEL_REGISTRY,
    **REGRESSION_MODEL_REGISTRY,
    # Legacy entries
    "hf_lora": lambda **kwargs: HuggingFaceLoRAWrapper(task_type='classification', **kwargs),
    "hf_qlora": lambda **kwargs: HuggingFaceQLoRAWrapper(task_type='classification', **kwargs),
}

