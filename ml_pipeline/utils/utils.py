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

def load_model(model_class, model_name, params, path_start=None):
    if path_start is not None:
        path = os.path.join(RESULTS_DIR, path_start, f"{model_name}.pt")
    else:
        path = os.path.join(RESULTS_DIR, f"{model_name}.pt")
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

def select_best_epoch(history, task_type='classification'):
    """
    Select best epoch using combined score of normalized validation loss and primary metric.
    This is the centralized logic used by both GeneralPipeline and BenchmarkRunner.
    
    Args:
        history: List of dictionaries containing epoch metrics
        task_type: 'classification' or 'regression'
        
    Returns:
        int: Index of the best epoch
    """
    if not history:
        return 0
    
    # Get primary metric name based on task type
    primary_metric = 'f1_score' if task_type == 'classification' else 'r2_score'
    
    # Gather per-epoch values for validation loss and primary metric
    val_losses = [h.get('val_loss', None) for h in history]
    primary_scores = [h.get(primary_metric, None) for h in history]

    # Determine indices that have at least one recorded value
    valid_indices = [i for i, (vl, ps) in enumerate(zip(val_losses, primary_scores)) if vl is not None or ps is not None]
    if not valid_indices:
        return 0  # Fallback to first epoch

    # Build numpy arrays for normalization (preserve order relative to valid_indices)
    vl_arr = np.array([val_losses[i] if val_losses[i] is not None else np.nan for i in valid_indices], dtype=float)
    ps_arr = np.array([primary_scores[i] if primary_scores[i] is not None else np.nan for i in valid_indices], dtype=float)

    # Normalization helper: map values to [0,1]; handle constant arrays and all-nan arrays
    def _normalize(arr: np.ndarray) -> np.ndarray:
        if np.all(np.isnan(arr)):
            return np.full_like(arr, 0.5, dtype=float)
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        if mx == mn:
            # All values identical -> produce neutral score (0.0 so it doesn't bias loss)
            return np.zeros_like(arr, dtype=float)
        return (arr - mn) / (mx - mn)

    norm_loss = _normalize(vl_arr)
    norm_metric = _normalize(ps_arr)

    # For missing entries (nan) use neutral value 0.5 so they don't dominate selection
    norm_loss = np.where(np.isnan(vl_arr), 0.5, norm_loss)
    norm_metric = np.where(np.isnan(ps_arr), 0.5, norm_metric)

    # Lower is better for loss; higher is better for metric. Invert normalized metric so lower==better
    inv_norm_metric = 1.0 - norm_metric

    # Combined score: equal weight to normalized loss and inverted normalized metric
    combined_score = 0.5 * norm_loss + 0.5 * inv_norm_metric

    # Select epoch with minimal combined score. Map back to original epoch index
    best_idx_in_valid = int(np.nanargmin(combined_score))
    best_epoch = valid_indices[best_idx_in_valid]
    
    return best_epoch

def calculate_combined_score_for_epoch(history, target_epoch, task_type='classification'):
    """
    Calculate combined score for a specific epoch using the same logic as select_best_epoch.
    Used for early stopping to ensure consistency.
    
    Args:
        history: List of dictionaries containing epoch metrics
        target_epoch: The epoch to calculate score for
        task_type: 'classification' or 'regression'
        
    Returns:
        float: Combined score for the target epoch
    """
    if target_epoch >= len(history):
        return float('inf')
    
    # Get primary metric name based on task type
    primary_metric = 'f1_score' if task_type == 'classification' else 'r2_score'
    
    # Use history up to current epoch (for early stopping)
    history_subset = history[:target_epoch + 1]
    
    # Gather per-epoch values for validation loss and primary metric
    val_losses = [h.get('val_loss', None) for h in history_subset]
    primary_scores = [h.get(primary_metric, None) for h in history_subset]

    # Determine indices that have at least one recorded value
    valid_indices = [i for i, (vl, ps) in enumerate(zip(val_losses, primary_scores)) if vl is not None or ps is not None]
    if not valid_indices or target_epoch not in valid_indices:
        return float('inf')

    # Build numpy arrays for normalization (preserve order relative to valid_indices)
    vl_arr = np.array([val_losses[i] if val_losses[i] is not None else np.nan for i in valid_indices], dtype=float)
    ps_arr = np.array([primary_scores[i] if primary_scores[i] is not None else np.nan for i in valid_indices], dtype=float)

    # Same normalization logic as select_best_epoch
    def _normalize(arr: np.ndarray) -> np.ndarray:
        if np.all(np.isnan(arr)):
            return np.full_like(arr, 0.5, dtype=float)
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)
        if mx == mn:
            return np.zeros_like(arr, dtype=float)
        return (arr - mn) / (mx - mn)

    norm_loss = _normalize(vl_arr)
    norm_metric = _normalize(ps_arr)

    norm_loss = np.where(np.isnan(vl_arr), 0.5, norm_loss)
    norm_metric = np.where(np.isnan(ps_arr), 0.5, norm_metric)

    inv_norm_metric = 1.0 - norm_metric
    combined_score = 0.5 * norm_loss + 0.5 * inv_norm_metric

    # Find the target epoch in valid_indices and return its score
    try:
        target_idx_in_valid = valid_indices.index(target_epoch)
        return float(combined_score[target_idx_in_valid])
    except ValueError:
        return float('inf')
