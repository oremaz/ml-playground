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
    augmentation = augmentation or 'none'
    model_file = f"{model_name}_{augmentation}.pt"
    if path_start is None:
        raise ValueError("path_start must be provided to load the model.")

    candidate_dirs = []
    if _running_on_kaggle():
        candidate_dirs.append(os.path.join(RESULTS_DIR_IN, path_start))
        candidate_dirs.append(os.path.join(RESULTS_DIR_OUT, path_start))
    else:
        candidate_dirs.append(os.path.join(RESULTS_DIR_IN, path_start))

    path = None
    for base_dir in candidate_dirs:
        candidate_path = os.path.join(base_dir, model_file)
        if os.path.exists(candidate_path):
            path = candidate_path
            break

    if path is None:
        # Fall back to the first candidate so the load call raises a clear error
        path = os.path.join(candidate_dirs[0], model_file)
    model = model_class(**params)
    
    if hasattr(model, "load_state_dict"):
        # Only use map_location if CUDA is not available
        if torch.cuda.is_available():
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    elif hasattr(model, "model"):
        model.model = joblib.load(path)
    elif hasattr(model, "from_pretrained"):
        model = model_class.from_pretrained(os.path.join(RESULTS_DIR, model_name))
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
