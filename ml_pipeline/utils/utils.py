import os
import json
import hashlib
import joblib
import torch
import pandas as pd
import numpy as np
try:
    from .kaggle_utils import _running_on_kaggle
except Exception:
    from kaggle_utils import _running_on_kaggle

if _running_on_kaggle():
    RESULTS_DIR_OUT = "/kaggle/working/results"
    RESULTS_DIR_IN = "/kaggle/input"
    RESULTS_DIR = RESULTS_DIR_OUT
else:
    RESULTS_DIR = "results"
    RESULTS_DIR_OUT = RESULTS_DIR
    RESULTS_DIR_IN = RESULTS_DIR

os.makedirs(RESULTS_DIR_OUT, exist_ok=True)

def _index_path(path_start: str) -> str:
    return os.path.join(RESULTS_DIR_OUT, path_start, "index.jsonl")


def _load_index(path_start: str) -> dict:
    index_path = _index_path(path_start)
    if not os.path.exists(index_path):
        return {}
    entries = {}
    with open(index_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entries[entry["checkpoint_id"]] = entry
    return entries


def _write_index_entry(path_start: str, entry: dict) -> None:
    index_path = _index_path(path_start)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def _checkpoint_id(metadata_core: dict) -> str:
    payload = json.dumps(metadata_core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def get_checkpoint_id(
    path_start: str,
    model_name: str,
    augmentation_name: str = "none",
    task_type: str = None,
) -> str:
    index = _load_index(path_start)
    matches = [
        v
        for v in index.values()
        if v.get("model_name") == model_name
        and v.get("augmentation_name") == augmentation_name
        and (task_type is None or v.get("task_type") == task_type)
    ]
    if not matches:
        raise FileNotFoundError("No matching checkpoint in index.jsonl.")
    if len(matches) > 1:
        raise ValueError("Multiple matches found; refine your query.")
    return matches[0]["checkpoint_id"]


def load_model_by_name(
    model_class,
    model_name: str,
    params: dict,
    path_start: str,
    augmentation_name: str = "none",
    task_type: str = None,
):
    checkpoint_id = get_checkpoint_id(
        path_start=path_start,
        model_name=model_name,
        augmentation_name=augmentation_name,
        task_type=task_type,
    )
    return load_model(model_class, params, path_start=path_start, checkpoint_id=checkpoint_id)


def model_exists(metadata_core: dict, path_start: str) -> bool:
    if path_start is None:
        raise ValueError("path_start must be provided to check model existence.")
    checkpoint_id = _checkpoint_id(metadata_core)
    index = _load_index(path_start)
    return checkpoint_id in index

def _is_sklearn_estimator(model) -> bool:
    try:
        from sklearn.base import BaseEstimator
        return isinstance(model, BaseEstimator)
    except Exception:
        return (
            hasattr(model, "fit")
            and hasattr(model, "predict")
            and hasattr(model, "get_params")
        )

def save_model(model, path_start: str, metadata_core: dict) -> dict:
    if path_start is None:
        raise ValueError("path_start must be provided to save the model.")
    base_dir = os.path.join(RESULTS_DIR_OUT, path_start)
    os.makedirs(base_dir, exist_ok=True)

    checkpoint_id = _checkpoint_id(metadata_core)

    if hasattr(model, "save_pretrained"):
        save_dir = os.path.join(base_dir, checkpoint_id)
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        if hasattr(model, "tokenizer"):
            model.tokenizer.save_pretrained(save_dir)
        artifact_type = "hf_dir"
        artifact_path = checkpoint_id
        print(f"Model saved to {save_dir}")
    elif hasattr(model, "state_dict"):
        path = os.path.join(base_dir, f"{checkpoint_id}.pt")
        torch.save(model.state_dict(), path)
        artifact_type = "pt_file"
        artifact_path = f"{checkpoint_id}.pt"
        print(f"Model state dict saved to {path}")
    elif hasattr(model, "model"):
        path = os.path.join(base_dir, f"{checkpoint_id}.pkl")
        joblib.dump(model.model, path)
        artifact_type = "joblib"
        artifact_path = f"{checkpoint_id}.pkl"
        print(f"Model saved to {path}")
    elif _is_sklearn_estimator(model):
        path = os.path.join(base_dir, f"{checkpoint_id}.pkl")
        joblib.dump(model, path)
        artifact_type = "joblib"
        artifact_path = f"{checkpoint_id}.pkl"
        print(f"Model saved to {path}")
    else:
        raise ValueError(f"Could not determine how to save model of type {type(model)}")

    entry = dict(metadata_core)
    entry.update(
        {
            "checkpoint_id": checkpoint_id,
            "artifact_type": artifact_type,
            "artifact_path": artifact_path,
        }
    )
    index = _load_index(path_start)
    if checkpoint_id not in index:
        _write_index_entry(path_start, entry)
    return entry

def load_model(
    model_class,
    params,
    path_start,
    checkpoint_id: str,
):
    """
    Load a trained model from disk using index metadata.

    Provide:
    - checkpoint_id
    """
    if path_start is None:
        raise ValueError("path_start must be provided to load the model.")
    if checkpoint_id is None:
        raise ValueError("checkpoint_id must be provided to load a model.")

    index = _load_index(path_start)
    entry = index.get(checkpoint_id)
    if entry is None:
        raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found in index.jsonl.")

    base_dir = os.path.join(RESULTS_DIR_OUT, path_start)
    found_path = os.path.join(base_dir, entry["artifact_path"])
    if not os.path.exists(found_path):
        raise FileNotFoundError(f"Checkpoint artifact not found at {found_path}")
    is_directory = entry["artifact_type"] == "hf_dir"
    
    print(f"Loading model from: {found_path}")
    
    # Load based on model type
    if is_directory:
        # HuggingFace model with PEFT/LoRA - Don't instantiate the wrapper normally
        import json
        from transformers import AutoTokenizer
        
        # Check if this is a HuggingFace PEFT model
        adapter_config_path = os.path.join(found_path, 'adapter_config.json')
        if os.path.exists(adapter_config_path):
            # This is a PEFT model - load it specially
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification
            import torch.nn as nn
            
            # Load metadata if available
            metadata_path = os.path.join(found_path, 'wrapper_metadata.json')
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"Loaded metadata: {metadata}")
            
            # Load adapter config manually to get base model name
            # Don't use PeftConfig.from_pretrained as it may have version incompatibilities
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            
            base_model_name = adapter_config.get('base_model_name_or_path')
            if not base_model_name:
                raise ValueError(f"Could not find base_model_name_or_path in {adapter_config_path}")
            
            print(f"Loading base model: {base_model_name}")
            
            # Determine task type and num_labels
            task_type = metadata.get('task_type', params.get('task_type', 'classification'))
            num_labels = params.get('num_labels', 2)
            
            # Determine dtype based on GPU availability
            if torch.cuda.is_available():
                try:
                    compute_capability = torch.cuda.get_device_capability()[0]
                    torch_dtype = torch.bfloat16 if compute_capability >= 8 else torch.float16
                except Exception:
                    torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
            
            # Load base model WITHOUT quantization (inference only)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=num_labels,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            )
            
            # Load PEFT adapters
            model_with_adapters = PeftModel.from_pretrained(base_model, found_path)
            print("✓ PEFT adapters loaded")
            
            # Now create the wrapper with the loaded model
            # Don't use the normal __init__ which would create a new model.
            # model_class is often a lambda factory (not a type) in caller notebooks,
            # so object.__new__ needs the concrete wrapper class instead.
            from pipelines_torch.models import HuggingFaceQLoRAWrapper
            model = object.__new__(HuggingFaceQLoRAWrapper)  # Create instance without calling __init__
            nn.Module.__init__(model)  # Initialize nn.Module part
            
            # Set attributes manually
            model.model = model_with_adapters
            model.tokenizer = AutoTokenizer.from_pretrained(found_path)
            model.task_type = task_type
            model.device = params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            model.max_seq_length = metadata.get('max_seq_length', 512)
            model._is_quantized = False  # Loaded model is not quantized
            import inspect as _inspect
            model._needs_token_type_ids = "token_type_ids" in _inspect.signature(base_model.forward).parameters
            # base_model.config.pad_token_id is None on a fresh from_pretrained load;
            # batched generation/inference needs it set from the tokenizer.
            model._sync_special_token_ids_with_tokenizer()
            
            # Restore label encoder if available
            if 'label_classes' in metadata:
                from sklearn.preprocessing import LabelEncoder
                model.label_encoder = LabelEncoder()
                model.label_encoder.classes_ = np.array(metadata['label_classes'])
                print(f"Restored label encoder with classes: {metadata['label_classes']}")
            
            print(f"✓ HuggingFace PEFT model loaded from {found_path}")
            
        else:
            raise FileNotFoundError(f"adapter_config.json not found in {found_path}")
    
    else:
        # Non-directory models (PyTorch .pt files or sklearn models)
        if entry["artifact_type"] == "joblib":
            loaded_model = joblib.load(found_path)
            try:
                model = model_class(**params)
            except TypeError:
                print(f"✓ Sklearn model loaded from {found_path}")
                return loaded_model

            if hasattr(model, "model"):
                model.model = loaded_model
                print(f"✓ Sklearn model loaded from {found_path}")
                return model

            print(f"✓ Sklearn model loaded from {found_path}")
            return loaded_model

        model = model_class(**params)

        if hasattr(model, "load_state_dict"):
            # PyTorch model with state_dict
            if torch.cuda.is_available():
                state_dict = torch.load(found_path)
            else:
                state_dict = torch.load(found_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print(f"✓ PyTorch model loaded from {found_path}")

        else:
            raise ValueError(f"Don't know how to load model of type {type(model)}")
    
    return model


def save_metrics(metrics, checkpoint_id, path_start):
    if path_start is not None:
        base_dir = os.path.join(RESULTS_DIR_OUT, path_start)
    else:
        raise ValueError("path_start must be provided to save metrics.")
    os.makedirs(base_dir, exist_ok=True)
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(base_dir, f"{checkpoint_id}_metrics.csv"), index=False)

def select_best_epoch(history, task_type='classification', metric=None):
    """
    Select best epoch using only the referenced metric.
    This is the centralized logic used by both GeneralPipeline and BenchmarkRunner.
    
    Args:
        history: List of dictionaries containing epoch metrics
        task_type: 'classification' or 'regression'
        metric: Optional metric name to prioritize (defaults to ROC-AUC for classification, R² for regression)
        
    Returns:
        int: Index of the best epoch
    """
    if not history:
        return 0

    # Determine metric name based on task type (allow override)
    default_metric = 'roc_auc' if task_type == 'classification' else 'r2_score'
    metric_name = metric or default_metric

    metric_values = []

    for record in history:
        value = record.get(metric_name)
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            value = None
        if value is not None:
            metric_values.append(float(value))
        else:
            metric_values.append(None)

    best_epoch = 0
    best_value = float('-inf')
    for idx, value in enumerate(metric_values):
        if value is None:
            continue
        if value > best_value:
            best_value = value
            best_epoch = idx

    return best_epoch
