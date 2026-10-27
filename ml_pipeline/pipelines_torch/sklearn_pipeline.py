"""Classical fitting, cross-validation and fold restart state."""

import logging
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold
from ml_pipeline.utils.data import train_val_test_split

logger = logging.getLogger("ml_pipeline.pipelines_torch.base")


from .evaluation import Evaluate, ComputeScore

class GeneralPipelineSklearn(Evaluate):
    """
    General pipeline for scikit-learn models supporting classification and regression.
    """

    __module__ = "ml_pipeline.pipelines_torch.base"
    def __init__(
        self,
        model,
        metrics=None,
        task_type="classification",
        use_class_weights=False,
        augmentations=None,
        max_factor=2.0,  # Data augmentation factor (majority_count / final_minority_count)
        random_state=42,  # Random seed for reproducibility
        use_kfold=True,  # Whether to use k-fold cross-validation
        k_folds=5,  # Number of folds for k-fold CV
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.metrics = metrics or []
        self.task_type = task_type
        self.use_class_weights = use_class_weights
        self.augmentations = augmentations
        self.max_factor = max_factor
        self.random_state = random_state
        self.use_kfold = use_kfold
        self.k_folds = k_folds
        self.history = []
        self.cv_results = {}  # Store cross-validation results
        self.class_weights = None

    def prepare_data_internal(self, X: np.ndarray, y: Optional[np.ndarray] = None, train: bool = True) -> DataLoader:
        """Sklearn models don't use DataLoaders, so this returns a dummy implementation."""
        # For sklearn models, we don't need DataLoaders, but we implement this for compatibility
        # The actual data handling is done directly in the evaluation methods
        return None

    def _kfold_fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation for scikit-learn models.

        Since sklearn models don't have weights to average, this approach:
        1. K-fold CV to get unbiased estimate of model performance
        2. Train final model on full dataset
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.base import clone

        # Use StratifiedKFold for classification to preserve class ratios
        if self.task_type == "classification":
            kfold = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
            cv_iterator = kfold.split(X, y)
        else:
            kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
            cv_iterator = kfold.split(X)

        from pathlib import Path
        from ml_pipeline.utils.training_checkpoint import (
            CheckpointPause, atomic_save, data_fingerprint, load_checkpoint, restore_rng, rng_state,
        )
        restart_path = getattr(self, "restart_path", None)
        restart_file = Path(restart_path) / "restart.pt" if restart_path else None
        context = {"run": getattr(self, "restart_context", None), "k_folds": self.k_folds,
                   "seed": self.random_state, "data_fingerprint": data_fingerprint(X, y)}
        restart = load_checkpoint(restart_file, context) if restart_file else None
        fold_scores, next_fold, completed = [], 0, False
        if restart is not None:
            self.model, fold_scores = restart["model"], restart["fold_scores"]
            next_fold, completed = restart["next_fold"], restart["completed"]
            restore_rng(restart["rng"])

        def save(fold, completed=False):
            if restart_file is not None:
                atomic_save({"config": context, "model": self.model, "fold_scores": fold_scores,
                             "next_fold": fold, "completed": completed, "rng": rng_state()}, restart_file)
        save(next_fold, completed)
        logger.info("Starting %d-fold cross-validation for sklearn model...", self.k_folds)

        for fold, (train_idx, val_idx) in enumerate(cv_iterator):
            if fold < next_fold:
                continue
            logger.info("Training fold %d/%d", fold + 1, self.k_folds)

            # Split data for this fold
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            fold_model = self.model

            # Train on this fold (with augmentations if provided)
            self.internal_fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val, fold_model)

            # Evaluate on validation set
            y_pred = fold_model.predict(X_fold_val)

            # Compute metrics
            fold_metrics = {}
            if self.task_type == "classification":
                fold_metrics['f1_score'] = ComputeScore.f1(y_fold_val, y_pred, self.task_type)

                prob_preds = None
                if hasattr(fold_model, 'predict_proba'):
                    try:
                        prob_preds = fold_model.predict_proba(X_fold_val)
                    except Exception:
                        prob_preds = None
                if prob_preds is not None:
                    fold_metrics['roc_auc'] = ComputeScore.roc_auc(y_fold_val, prob_preds, self.task_type)
                    fold_metrics['pr_auc'] = ComputeScore.pr_auc(y_fold_val, prob_preds, self.task_type)
                else:
                    fold_metrics['roc_auc'] = None
                    fold_metrics['pr_auc'] = None
            else:  # regression
                fold_metrics['r2_score'] = ComputeScore.r2(y_fold_val, y_pred, self.task_type)

            # Compute additional metrics if provided
            for metric in self.metrics:
                metric_key = getattr(metric, 'name', None) or getattr(metric, '__name__', None) or str(metric)
                try:
                    fold_metrics[metric_key] = metric(y_fold_val, y_pred)
                except (ValueError, TypeError, IndexError):
                    fold_metrics[metric_key] = 0.0
                except Exception as e:
                    logger.warning("Unexpected error computing metric %s in fold: %s", metric_key, e)
                    fold_metrics[metric_key] = 0.0

            fold_scores.append(fold_metrics)
            save(fold + 1)
            if self._stop_request.requested:
                raise CheckpointPause("Completed classical validation folds saved")
            # Print fold results
            primary_metric = "f1_score" if self.task_type == "classification" else "r2_score"
            metric_value = fold_metrics.get(primary_metric)
            if metric_value is not None:
                logger.info("  Fold %d %s: %.4f", fold + 1, primary_metric, metric_value)

        # Calculate cross-validation statistics
        cv_results = self._compute_cv_statistics(fold_scores)
        self._print_cv_results(cv_results)

        # Train final model on full dataset
        logger.info("Training final model on full dataset...")
        if not completed:
            self.internal_fit(X, y, None, None, self.model)
            save(self.k_folds, completed=True)
            if self._stop_request.requested:
                raise CheckpointPause("Completed classical full-data fit saved")

        logger.info("Final model trained successfully on full dataset!")

        return {
            'cv_scores': cv_results,
            'ensemble_method': 'full_dataset_retrain',
            'n_folds': self.k_folds,
            'fold_individual_scores': fold_scores,
            'training_history': self._create_cv_metrics_summary(cv_results, fold_scores)
        }

    def internal_fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray], y_val: Optional[np.ndarray], model=None):
        """
        Internal method for training a sklearn model on a single fold or dataset.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (unused for sklearn models)
            y_val: Validation targets (unused for sklearn models)
            model: Model to train (defaults to self.model)
        """
        if model is None:
            model = self.model
        # Apply augmentations if provided
        if self.augmentations is not None:
            X_train, y_train = self.augmentations(X_train, y_train, max_factor=self.max_factor, random_state=self.random_state)
        # Compute and set class weights for supported models
        sample_weight = None
        if self.use_class_weights and self.task_type == "classification":
            unique_classes = np.unique(y_train)
            if len(unique_classes) > 1:
                from sklearn.utils.class_weight import compute_class_weight
                self.class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
                # For XGBoost and LightGBM, use sample_weight
                model_name = type(model).__name__.lower()
                is_xgb = 'xgb' in model_name or 'xgboost' in model_name
                is_lgbm = 'lgbm' in model_name or 'lightgbm' in model_name
                if is_xgb or is_lgbm:
                    # Compute sample weights for each instance
                    class_weight_dict = dict(zip(unique_classes, self.class_weights))
                    sample_weight = np.array([class_weight_dict[label] for label in y_train])
                else:
                    # Try to set class_weight if model supports it
                    if hasattr(model, 'class_weight') or 'class_weight' in getattr(model, 'get_params', lambda: {})():
                        try:
                            model.set_params(class_weight=dict(zip(unique_classes, self.class_weights)))
                        except Exception:
                            try:
                                model.set_params(class_weight='balanced')
                            except Exception:
                                pass
        # Fit the sklearn model
        if sample_weight is not None:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)

    def _evaluate(self, X, y):
        if X is None or y is None:
            return {}
        # Support wrappers that use .model.predict
        if hasattr(self.model, 'predict'):
            y_pred = self.model.predict(X)
        else:
            raise AttributeError(f"Model of type {type(self.model)} does not have a predict method.")
        results = {}
        for metric in self.metrics:
            metric_key = getattr(metric, 'name', None) or getattr(metric, '__name__', None) or str(metric)
            results[metric_key] = metric(y, y_pred)
        return results

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Lightweight wrapper exposing sklearn evaluation for the benchmark runner."""
        return self._evaluate(X, y)

    def fit(self, X, y):
        from ml_pipeline.utils.training_checkpoint import StopRequest
        with StopRequest() as self._stop_request:
            return self._fit_restart(X, y)

    def _fit_restart(self, X, y):
        """
        Train the scikit-learn model with optional k-fold cross-validation.

        Args:
            X: Training features
            y: Training targets

        Returns:
            For backward compatibility with save_metrics, always returns training history.
            Access CV results via get_cv_scores() method if needed.
        """
        if self.use_kfold:
            logger.info("Starting %d-fold cross-validation for sklearn model", self.k_folds)
            results = self._kfold_fit(X, y)
            self.cv_results = results  # Store for later access

            # Return training history for compatibility with save_metrics
            return results.get('training_history', [])
        else:
            # Single train/validation split
            if self.task_type == "classification":
                X_train, X_val, _, y_train, y_val, _ = train_val_test_split(
                    X, y, val_size=0.2, test_size=0, stratify=y, random_state=self.random_state
                )
            else:
                X_train, X_val, _, y_train, y_val, _ = train_val_test_split(
                    X, y, val_size=0.2, test_size=0, random_state=self.random_state
                )

            # Train the model
            self.internal_fit(X_train, y_train, X_val, y_val)

            # Evaluate on validation set
            val_metrics = self.evaluate(X_val, y_val)

            # Create simple history for compatibility with save_metrics
            history = [{
                'epoch': 1,
                'status': 'completed',
                'model_type': 'sklearn',
                **val_metrics
            }]

            self.history = history
            return history

