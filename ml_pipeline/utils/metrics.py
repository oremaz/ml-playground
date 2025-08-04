from typing import Callable, Dict
import numpy as np
from sklearn.metrics import accuracy_score as sk_accuracy_score, f1_score as sk_f1_score, precision_score as sk_precision_score, recall_score as sk_recall_score, mean_squared_error as sk_mse, mean_absolute_error as sk_mae, r2_score as sk_r2

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy for classification."""
    if y_pred.ndim > 1:
        y_pred = y_pred.argmax(axis=1)
    return sk_accuracy_score(y_true, y_pred)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro F1 score for classification."""
    if y_pred.ndim > 1:
        y_pred = y_pred.argmax(axis=1)
    return sk_f1_score(y_true, y_pred, average='macro')

def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro precision for classification."""
    if y_pred.ndim > 1:
        y_pred = y_pred.argmax(axis=1)
    return sk_precision_score(y_true, y_pred, average='macro', zero_division=0)

def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro recall for classification."""
    if y_pred.ndim > 1:
        y_pred = y_pred.argmax(axis=1)
    return sk_recall_score(y_true, y_pred, average='macro', zero_division=0)

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error for regression."""
    return sk_mse(y_true, y_pred)

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error for regression."""
    return sk_mae(y_true, y_pred)

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R^2 score for regression."""
    return sk_r2(y_true, y_pred)

METRIC_REGISTRY: Dict[str, Callable] = {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "r2": r2_score,
}
