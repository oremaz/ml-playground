"""Shared prediction and scoring contracts for model pipelines."""

import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional, Dict
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

logger = logging.getLogger("ml_pipeline.pipelines_torch.base")


class Evaluate:
    """Base utilities shared across torch and sklearn pipelines."""

    __module__ = "ml_pipeline.pipelines_torch.base"

    def __init__(self) -> None:
        # These attributes should be set by subclasses
        self.model = None
        self.task_type = "classification"
        self.metrics = []
        self.batch_size = 32
        self.random_state = 42
        self.device = "cpu"
        self.history = []
        self.cv_results = {}
        self.use_kfold = False

    def _default_selection_metric(self) -> str:
        """Return the metric name used to pick the best epoch."""
        return "roc_auc" if self.task_type == "classification" else "r2_score"

    # ----- Cross-validation shared methods --------------------------------
    def _compute_cv_statistics(self, fold_scores: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Compute cross-validation statistics (mean +/- std) for each metric."""
        if not fold_scores:
            return {}
        all_metrics = set()
        for scores in fold_scores:
            all_metrics.update(scores.keys())
        cv_stats = {}
        for metric in all_metrics:
            metric_values = [scores.get(metric) for scores in fold_scores]
            metric_array = np.array([v if v is not None else np.nan for v in metric_values], dtype=float)
            cv_stats[metric] = {
                'mean': float(np.nanmean(metric_array)),
                'std': float(np.nanstd(metric_array)),
                'values': metric_array.tolist()
            }
        return cv_stats

    def _print_cv_results(self, cv_results: Dict[str, Dict[str, float]]):
        """Print cross-validation results."""
        if not cv_results:
            return
        primary_metric = "f1_score" if self.task_type == "classification" else "r2_score"
        logger.info("\n%s", "=" * 60)
        logger.info("CROSS-VALIDATION RESULTS")
        logger.info("%s", "=" * 60)
        if primary_metric in cv_results:
            stats = cv_results[primary_metric]
            logger.info("  %15s: %.4f +/- %.4f", primary_metric.upper(), stats['mean'], stats['std'])
            logger.info("")
        for metric, stats in sorted(cv_results.items()):
            if metric != primary_metric:
                logger.info("  %15s: %.4f +/- %.4f", metric, stats['mean'], stats['std'])
        logger.info("%s\n", "=" * 60)

    def _create_cv_metrics_summary(self, cv_results: Dict[str, Dict[str, float]], fold_scores: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Create a metrics summary compatible with save_metrics function."""
        metrics_summary = []
        summary_row = {'fold': 'CV_MEAN'}
        for metric, stats in cv_results.items():
            summary_row[metric] = stats['mean']
            summary_row[f"{metric}_std"] = stats['std']
        metrics_summary.append(summary_row)
        for fold_idx, fold_metrics in enumerate(fold_scores):
            fold_row = {'fold': f'fold_{fold_idx + 1}'}
            fold_row.update(fold_metrics)
            metrics_summary.append(fold_row)
        return metrics_summary

    def get_training_history(self) -> List[Dict[str, float]]:
        """Get training history compatible with save_metrics."""
        if hasattr(self, 'cv_results') and self.cv_results and self.use_kfold:
            return self.cv_results.get('training_history', [])
        return self.history

    def get_cv_scores(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Get cross-validation scores summary, or None if CV wasn't performed."""
        if not hasattr(self, 'cv_results') or not self.cv_results.get('cv_scores'):
            return None
        return self.cv_results['cv_scores']

    def prepare_data_internal(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        train: bool = True,
    ) -> DataLoader:
        """Subclasses must implement how raw arrays become loaders."""
        raise NotImplementedError("Subclasses must implement prepare_data_internal")

    # ----- Evaluation helpers -------------------------------------------------
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if X is None or y is None:
            return {}
        if hasattr(self, "model") and hasattr(self.model, "eval"):
            return self._evaluate_torch(X, y)
        return self._evaluate_sklearn(X, y)

    def _evaluate_torch(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        loader: DataLoader = self.prepare_data_internal(X, y, train=False)
        self.model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                xb, yb = batch
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb)
                if self.task_type == "regression":
                    outputs = logits.squeeze(-1)
                else:
                    outputs = torch.softmax(logits, dim=1)
                preds.append(outputs.cpu())
                targets.append(yb.cpu())
        preds_np = torch.cat(preds).numpy()
        targets_np = torch.cat(targets).numpy()
        results: Dict[str, float] = {}
        for metric in self.metrics:
            key = getattr(metric, "name", None) or getattr(metric, "__name__", None) or str(metric)
            try:
                results[key] = float(metric(targets_np, preds_np))
            except (ValueError, TypeError, IndexError):
                results[key] = 0.0
            except Exception as e:
                logger.warning("Unexpected error computing metric %s: %s", key, e)
                results[key] = 0.0
        return results

    def _evaluate_sklearn(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if not hasattr(self.model, "predict"):
            raise AttributeError(f"Model of type {type(self.model)} does not have a predict method.")
        y_pred = self.model.predict(X)
        results: Dict[str, float] = {}
        for metric in self.metrics:
            key = getattr(metric, "name", None) or getattr(metric, "__name__", None) or str(metric)
            try:
                results[key] = float(metric(y, y_pred))
            except (ValueError, TypeError, IndexError):
                results[key] = 0.0
            except Exception as e:
                logger.warning("Unexpected error computing metric %s: %s", key, e)
                results[key] = 0.0
        return results

    # ----- Prediction helpers -------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self, "model") and hasattr(self.model, "eval"):
            return self._predict_torch(X)
        return self._predict_sklearn(X)

    def _predict_torch(self, X: np.ndarray) -> np.ndarray:
        loader: DataLoader = self.prepare_data_internal(X, train=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                xb = batch[0] if isinstance(batch, (list, tuple)) else batch
                xb = xb.to(self.device)
                logits = self.model(xb)
                if self.task_type == "regression":
                    preds.append(logits.squeeze(-1).cpu())
                else:
                    if logits.ndim == 1:
                        logits = logits.unsqueeze(-1)
                    if logits.shape[1] == 1:
                        probs = torch.sigmoid(logits).squeeze(-1)
                        preds.append((probs > 0.5).long().cpu())
                    else:
                        preds.append(torch.softmax(logits, dim=1).argmax(dim=1).cpu())
        return torch.cat(preds).numpy()

    def _predict_sklearn(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        raise AttributeError(f"Model of type {type(self.model)} does not have a predict method.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        if hasattr(self, "model") and hasattr(self.model, "eval"):
            return self._predict_proba_torch(X)
        return self._predict_proba_sklearn(X)

    def _predict_proba_torch(self, X: np.ndarray) -> np.ndarray:
        loader: DataLoader = self.prepare_data_internal(X, train=False)
        self.model.eval()
        probs_list = []
        with torch.no_grad():
            for batch in loader:
                xb = batch[0] if isinstance(batch, (list, tuple)) else batch
                xb = xb.to(self.device)
                logits = self.model(xb)
                if logits.ndim == 1:
                    logits = logits.unsqueeze(-1)
                if logits.shape[1] == 1:
                    pos = torch.sigmoid(logits).squeeze(-1)
                    probs = torch.stack((1 - pos, pos), dim=1)
                else:
                    probs = torch.softmax(logits, dim=1)
                probs_list.append(probs.cpu())
        return torch.cat(probs_list).numpy()

    def _predict_proba_sklearn(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise ValueError("Model does not support predict_proba")


class ComputeScore:
    """Utility scoring helpers shared by torch and sklearn pipelines."""

    __module__ = "ml_pipeline.pipelines_torch.base"

    @staticmethod
    def f1(targets: np.ndarray, preds: np.ndarray, task_type: str) -> Optional[float]:
        if task_type != "classification":
            return None
        try:
            targets_array = np.asarray(targets)
            preds_array = np.asarray(preds)
            average = "binary" if len(np.unique(targets_array)) == 2 else "macro"
            if preds_array.ndim > 1:
                preds_processed = np.argmax(preds_array, axis=1)
            else:
                if len(np.unique(targets_array)) == 2:
                    preds_processed = (preds_array >= 0.5).astype(int)
                else:
                    preds_processed = preds_array
            return float(f1_score(targets_array, preds_processed, average=average, zero_division=0))
        except (ValueError, IndexError) as e:
            logger.debug("f1 computation failed: %s", e)
            return None
        except Exception as e:
            logger.warning("Unexpected error in f1 computation: %s", e)
            return None

    @staticmethod
    def r2(targets: np.ndarray, preds: np.ndarray, task_type: str) -> Optional[float]:
        if task_type != "regression":
            return None
        try:
            from sklearn.metrics import r2_score
            return float(r2_score(targets, preds))
        except (ValueError, TypeError) as e:
            logger.debug("r2 computation failed: %s", e)
            return None
        except Exception as e:
            logger.warning("Unexpected error in r2 computation: %s", e)
            return None

    @staticmethod
    def roc_auc(targets: np.ndarray, preds: np.ndarray, task_type: str) -> Optional[float]:
        if task_type != "classification":
            return None
        try:
            y_true = np.asarray(targets)
            y_scores = np.asarray(preds)
            if y_scores.ndim == 1:
                return float(roc_auc_score(y_true, y_scores))
            if y_scores.ndim == 2:
                if y_scores.shape[1] == 1:
                    return float(roc_auc_score(y_true, y_scores[:, 0]))
                if y_scores.shape[1] == 2:
                    return float(roc_auc_score(y_true, y_scores[:, 1]))
                return float(roc_auc_score(y_true, y_scores, multi_class="ovr", average="macro"))
            return None
        except (ValueError, IndexError) as e:
            logger.debug("roc_auc computation failed: %s", e)
            return None
        except Exception as e:
            logger.warning("Unexpected error in roc_auc computation: %s", e)
            return None

    @staticmethod
    def pr_auc(targets: np.ndarray, preds: np.ndarray, task_type: str) -> Optional[float]:
        if task_type != "classification":
            return None
        try:
            y_true = np.asarray(targets)
            y_scores = np.asarray(preds)
            if y_scores.ndim == 1:
                return float(average_precision_score(y_true, y_scores))
            if y_scores.ndim == 2:
                if y_scores.shape[1] == 1:
                    return float(average_precision_score(y_true, y_scores[:, 0]))
                if y_scores.shape[1] == 2:
                    return float(average_precision_score(y_true, y_scores[:, 1]))
                classes = np.arange(y_scores.shape[1])
                y_true_bin = label_binarize(y_true, classes=classes)
                return float(average_precision_score(y_true_bin, y_scores, average="macro"))
            return None
        except (ValueError, IndexError) as e:
            logger.debug("pr_auc computation failed: %s", e)
            return None
        except Exception as e:
            logger.warning("Unexpected error in pr_auc computation: %s", e)
            return None


class SimplePredictor(Evaluate):
    """
    Lightweight class for making predictions with trained models.
    Minimal setup required - just provide the model and basic info.
    """

    __module__ = "ml_pipeline.pipelines_torch.base"

    def __init__(self, model, task_type: str = "classification", device: str = "cpu", batch_size: int = 32):
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.device = device
        self.batch_size = batch_size
        self.random_state = 42
        self.metrics = []

        # Move PyTorch models to device
        if hasattr(model, 'to'):
            self.model = model.to(device)

    def prepare_data_internal(self, X: np.ndarray, y: Optional[np.ndarray] = None, train: bool = True) -> DataLoader:
        """Prepare data for PyTorch models, return None for sklearn models."""
        if hasattr(self, 'device') and hasattr(self.model, 'eval'):
            # PyTorch model - create DataLoader
            X_tensor = torch.as_tensor(X, dtype=torch.float32)
            if y is not None:
                y_tensor = torch.as_tensor(y, dtype=torch.float32 if self.task_type == "regression" else torch.long)
                dataset = TensorDataset(X_tensor, y_tensor)
            else:
                dataset = TensorDataset(X_tensor)

            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,  # No need to shuffle for prediction
                num_workers=0,
                drop_last=False
            )
        else:
            # sklearn model - return None (data passed directly)
            return None

