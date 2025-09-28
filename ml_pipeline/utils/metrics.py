from typing import Callable, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score as sk_accuracy_score,
    f1_score as sk_f1_score,
    precision_score as sk_precision_score,
    recall_score as sk_recall_score,
    mean_squared_error as sk_mse,
    mean_absolute_error as sk_mae,
    r2_score as sk_r2,
    roc_auc_score as sk_roc_auc_score,
    average_precision_score as sk_average_precision_score,
)
from sklearn.preprocessing import label_binarize

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


def roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute ROC-AUC score.

    Supports binary and multiclass. If `y_pred` contains per-class scores/probabilities
    (shape (n_samples, n_classes)) the function handles binary (uses column 1)
    and multiclass (uses macro average with one-vs-rest).
    """
    try:
        if y_pred.ndim > 1:
            # Binary with two columns -> use positive class scores
            if y_pred.shape[1] == 2:
                scores = y_pred[:, 1]
                return float(sk_roc_auc_score(y_true, scores))
            # Multiclass: binarize and compute macro ROC-AUC
            classes = np.unique(y_true)
            y_true_bin = label_binarize(y_true, classes=classes)
            return float(sk_roc_auc_score(y_true_bin, y_pred, average="macro", multi_class="ovr"))
        else:
            return float(sk_roc_auc_score(y_true, y_pred))
    except Exception:
        return 0.0


def pr_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute PR-AUC (average precision).

    Supports binary and multiclass. For multiclass, returns macro average across classes.
    """
    try:
        if y_pred.ndim > 1:
            if y_pred.shape[1] == 2:
                scores = y_pred[:, 1]
                return float(sk_average_precision_score(y_true, scores))
            classes = np.unique(y_true)
            y_true_bin = label_binarize(y_true, classes=classes)
            return float(sk_average_precision_score(y_true_bin, y_pred, average="macro"))
        else:
            return float(sk_average_precision_score(y_true, y_pred))
    except Exception:
        return 0.0

METRIC_REGISTRY: Dict[str, Callable] = {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "r2": r2_score,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
}
