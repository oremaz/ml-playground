import os
import joblib
import torch
import pandas as pd
import numpy as np
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_model(model, model_name, path_start=None):
    if path_start is not None:
        base_dir = os.path.join(RESULTS_DIR, path_start)
    else:
        base_dir = RESULTS_DIR
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{model_name}.pt")
    if hasattr(model, "state_dict"):
        torch.save(model.state_dict(), path)
    elif hasattr(model, "model"):
        joblib.dump(model.model, path)
    elif hasattr(model, "save_pretrained"):
        model.save_pretrained(os.path.join(base_dir, model_name))
    # Add more logic for LLMs if needed

def load_model(model_class, model_name, params, path_start=None, augmentation=None):
    augmentation = augmentation or 'none'
    model_file = f"{model_name}_{augmentation}.pt"
    if path_start is not None:
        path = os.path.join(RESULTS_DIR, path_start, model_file)
    else:
        path = os.path.join(RESULTS_DIR, model_file)
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

def save_metrics(metrics, model_name, phase, path_start=None):
    if path_start is not None:
        base_dir = os.path.join(RESULTS_DIR, path_start)
    else:
        base_dir = RESULTS_DIR
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

def calculate_combined_score_for_epoch(history, target_epoch, task_type='classification', metric=None):
    """
    Calculate combined score for a specific epoch using only the referenced metric.
    Used for early stopping to ensure consistency.
    
    Args:
        history: List of dictionaries containing epoch metrics
        target_epoch: The epoch to calculate score for
        task_type: 'classification' or 'regression'
        metric: Optional metric name to prioritize (defaults to ROC-AUC for classification, R² for regression)
        
    Returns:
        float: Combined score for the target epoch
    """
    if target_epoch >= len(history):
        return float('inf')
    
    # Get metric name based on task type (allow override)
    default_metric = 'roc_auc' if task_type == 'classification' else 'r2_score'
    metric_name = metric or default_metric

    try:
        value = history[target_epoch].get(metric_name)
    except IndexError:
        return float('inf')

    if isinstance(value, (float, np.floating)) and np.isnan(value):
        value = None

    if value is None:
        legacy_metric = 'f1_score' if task_type == 'classification' else 'r2_score'
        value = history[target_epoch].get(legacy_metric)

    if isinstance(value, (float, np.floating)) and np.isnan(value):
        value = None

    if value is None:
        return float('inf')

    # Lower combined score should still indicate better performance, so negate the metric
    return -float(value)
