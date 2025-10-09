import os
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

def save_model(model, model_name, path_start):
    if path_start is not None:
        base_dir = os.path.join(RESULTS_DIR_OUT, path_start)
    else:
        raise ValueError("path_start must be provided to save the model.")
    os.makedirs(base_dir, exist_ok=True)
    
    # Priority order for saving:
    # 1. HuggingFace models with save_pretrained (PEFT/LoRA models)
    # 2. PyTorch models with state_dict
    # 3. Sklearn models via joblib
    
    if hasattr(model, "save_pretrained"):
        # HuggingFace models (including PEFT/LoRA wrapped models)
        save_dir = os.path.join(base_dir, model_name)
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        # Also save tokenizer if available
        if hasattr(model, "tokenizer"):
            model.tokenizer.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")
    elif hasattr(model, "state_dict"):
        # PyTorch models
        path = os.path.join(base_dir, f"{model_name}.pt")
        torch.save(model.state_dict(), path)
        print(f"Model state dict saved to {path}")
    elif hasattr(model, "model"):
        # Sklearn models wrapped in a class
        path = os.path.join(base_dir, f"{model_name}.pt")
        joblib.dump(model.model, path)
        print(f"Model saved to {path}")
    else:
        print(f"Warning: Could not determine how to save model of type {type(model)}")

def load_model(model_class, model_name, params, path_start, augmentation=None):
    """
    Load a trained model from disk.
    
    Supports:
    - HuggingFace models with PEFT/LoRA adapters (saved via save_pretrained)
    - PyTorch models (saved via state_dict)
    - Sklearn models (saved via joblib)
    """
    augmentation = augmentation or 'none'
    if path_start is None:
        raise ValueError("path_start must be provided to load the model.")

    candidate_dirs = []
    if _running_on_kaggle():
        candidate_dirs.append(os.path.join(RESULTS_DIR_IN, path_start))
        candidate_dirs.append(os.path.join(RESULTS_DIR_OUT, path_start))
    else:
        candidate_dirs.append(os.path.join(RESULTS_DIR_OUT, path_start))
        candidate_dirs.append(os.path.join(RESULTS_DIR_IN, path_start))

    # Try to find the model directory or file
    model_dir_name = f"{model_name}_{augmentation}"
    model_file_name = f"{model_name}_{augmentation}.pt"
    
    found_path = None
    is_directory = False
    
    for base_dir in candidate_dirs:
        # First, check if it's a directory (HuggingFace format)
        candidate_dir = os.path.join(base_dir, model_dir_name)
        if os.path.isdir(candidate_dir):
            # Check if it contains HuggingFace model files
            try:
                dir_contents = os.listdir(candidate_dir)
                if any(f in dir_contents for f in ['adapter_config.json', 'adapter_model.safetensors', 'config.json', 'pytorch_model.bin', 'model.safetensors']):
                    found_path = candidate_dir
                    is_directory = True
                    break
            except Exception:
                continue
        
        # Then check for .pt file (PyTorch/sklearn format)
        candidate_file = os.path.join(base_dir, model_file_name)
        if os.path.exists(candidate_file):
            found_path = candidate_file
            is_directory = False
            break

    if found_path is None:
        raise FileNotFoundError(
            f"Could not find model '{model_name}_{augmentation}' in any of the candidate directories: {candidate_dirs}"
        )
    
    print(f"Loading model from: {found_path}")
    
    # Instantiate the model
    model = model_class(**params)
    
    # Load based on model type
    if is_directory:
        # HuggingFace model with PEFT/LoRA
        import json
        
        # Load metadata if available
        metadata_path = os.path.join(found_path, 'wrapper_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Loaded metadata: {metadata}")
            
            # Restore label encoder if available
            if 'label_classes' in metadata:
                from sklearn.preprocessing import LabelEncoder
                model.label_encoder = LabelEncoder()
                model.label_encoder.classes_ = np.array(metadata['label_classes'])
                print(f"Restored label encoder with classes: {metadata['label_classes']}")
            
            # Restore max_seq_length
            if 'max_seq_length' in metadata:
                model.max_seq_length = metadata['max_seq_length']
                print(f"Restored max_seq_length: {model.max_seq_length}")
        
        # Load tokenizer
        if hasattr(model, 'tokenizer'):
            from transformers import AutoTokenizer
            model.tokenizer = AutoTokenizer.from_pretrained(found_path)
            print("✓ Tokenizer loaded")
        
        # Load PEFT model
        if hasattr(model, 'model'):
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification
            
            # Get adapter config to find base model
            adapter_config_path = os.path.join(found_path, 'adapter_config.json')
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get('base_model_name_or_path')
                
                if base_model_name:
                    print(f"Loading base model: {base_model_name}")
                    # Determine if it's classification or regression
                    task_type = getattr(model, 'task_type', 'classification')
                    num_labels = params.get('num_labels', 2)
                    
                    # Determine dtype and device based on availability
                    device_map = "auto" if torch.cuda.is_available() else None
                    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                    
                    # Load base model
                    try:
                        base_model = AutoModelForSequenceClassification.from_pretrained(
                            base_model_name,
                            num_labels=num_labels,
                            torch_dtype=torch_dtype,
                            device_map=device_map,
                            low_cpu_mem_usage=True,
                        )
                        print("✓ Base model loaded")
                    except Exception as e:
                        print(f"Warning: Failed to load with quantization, trying without: {e}")
                        base_model = AutoModelForSequenceClassification.from_pretrained(
                            base_model_name,
                            num_labels=num_labels,
                        )
                    
                    # Load PEFT adapters
                    model.model = PeftModel.from_pretrained(base_model, found_path)
                    model.model.eval()
                    print("✓ PEFT adapters loaded")
                else:
                    raise ValueError(f"Could not find base_model_name_or_path in {adapter_config_path}")
            else:
                raise FileNotFoundError(f"adapter_config.json not found in {found_path}")
        
        print(f"✓ HuggingFace PEFT model loaded successfully from {found_path}")
        
    elif hasattr(model, "load_state_dict"):
        # PyTorch model
        if torch.cuda.is_available():
            state_dict = torch.load(found_path)
        else:
            state_dict = torch.load(found_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"✓ PyTorch model loaded from {found_path}")
        
    elif hasattr(model, "model"):
        # Sklearn model
        model.model = joblib.load(found_path)
        print(f"✓ Sklearn model loaded from {found_path}")
        
    else:
        raise ValueError(f"Don't know how to load model of type {type(model)}")
    
    return model

def save_metrics(metrics, model_name, phase, path_start):
    if path_start is not None:
        base_dir = os.path.join(RESULTS_DIR_OUT, path_start)
    else:
        raise ValueError("path_start must be provided to save metrics.")
    os.makedirs(base_dir, exist_ok=True)
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(base_dir, f"{model_name}_{phase}_metrics.csv"), index=False)

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
