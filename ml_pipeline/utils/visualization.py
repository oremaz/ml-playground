import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional
from sklearn.metrics import confusion_matrix
import pandas as pd

def plot_confusion_matrix(y_true: List, y_pred: List, labels: Optional[List[str]] = None, title: str = "Confusion Matrix"):
    """
    Plot a confusion matrix for classification results.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_regression_results(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Regression Results"):
    """
    Scatter plot of true vs predicted values for regression.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_metric_history(history: dict, metric: str = "loss", val_metric: Optional[str] = None, title: str = "Training History"):
    """
    Plot training/validation metric curves from a history dict.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(history[metric], label=metric)
    if val_metric and val_metric in history:
        plt.plot(history[val_metric], label=val_metric)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_augmentation_effect(metrics_before: dict, metrics_after: dict, metric: str = "accuracy", title: str = "Augmentation Effect"):
    """
    Barplot of metric before/after augmentation.
    """
    labels = ["Before", "After"]
    values = [metrics_before.get(metric, 0), metrics_after.get(metric, 0)]
    plt.figure(figsize=(4, 4))
    sns.barplot(x=labels, y=values)
    plt.ylabel(metric.capitalize())
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_metric_csv(csv_path, metric="loss", val_metric=None, title="Training History"):
    """
    Plot training/validation metric curves from a CSV file.
    """
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 4))
    plt.plot(df[metric], label=metric)
    if val_metric and val_metric in df.columns:
        plt.plot(df[val_metric], label=val_metric)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_metrics_bar(metrics_df, title="Augmentation Benchmark Metrics", save_path=None):
    """
    Plot available metrics (columns) vs augmentation method (index) as a grouped bar chart.
    metrics_df: DataFrame with index=augmentation, columns=metrics, values=scores
    """
    ax = metrics_df.plot(kind='bar', figsize=(12,6))
    plt.title(title)
    plt.ylabel("Score")
    plt.xlabel("Augmentation Method")
    plt.xticks(rotation=45)
    plt.legend(title="Metric")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_per_class_metrics(report_df: pd.DataFrame, title: str = "Per-Class Metrics"):
    """
    Plots per-class precision, recall, and f1-score from a classification report DataFrame.
    report_df: DataFrame generated from sklearn.metrics.classification_report.
    """
    # Exclude support and summary rows like 'accuracy', 'macro avg', 'weighted avg'
    plot_df = report_df.drop(['support'], axis=1)
    plot_df = plot_df.loc[~plot_df.index.isin(['accuracy', 'macro avg', 'weighted avg'])]
    
    ax = plot_df.plot(kind='bar', figsize=(12, 6), rot=0)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.show()