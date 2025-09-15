import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Callable, List, Optional, Dict, Any, Tuple, Union
import numpy as np
import random
from collections.abc import Sized
from typing import cast
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import sys
import os
from utils.utils import select_best_epoch, calculate_combined_score_for_epoch


class GeneralPipelineSklearn:
    """
    General pipeline for scikit-learn models supporting classification and regression.
    """
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

    def _compute_f1_score(self, targets: np.ndarray, preds: np.ndarray) -> float:
        """Compute F1 score for classification tasks."""
        if self.task_type != "classification":
            return 0.0
        
        # Use macro average for multi-class, binary for binary classification
        average = 'binary' if len(np.unique(targets)) == 2 else 'macro'
        
        try:
            return f1_score(targets, preds, average=average, zero_division=0)
        except:
            return 0.0

    def _compute_r2_score(self, targets: np.ndarray, preds: np.ndarray) -> float:
        """Compute R² score for regression tasks."""
        if self.task_type != "regression":
            return 0.0
        
        try:
            from sklearn.metrics import r2_score
            return r2_score(targets, preds)
        except:
            return 0.0

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
        
        fold_scores = []
        
        print(f"Starting {self.k_folds}-fold cross-validation for sklearn model...")
        
        for fold, (train_idx, val_idx) in enumerate(cv_iterator):
            print(f"Training fold {fold + 1}/{self.k_folds}")
            
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
                fold_metrics['f1_score'] = self._compute_f1_score(y_fold_val, y_pred)
            else:  # regression
                fold_metrics['r2_score'] = self._compute_r2_score(y_fold_val, y_pred)
            
            # Compute additional metrics if provided
            for metric in self.metrics:
                metric_key = getattr(metric, 'name', None) or getattr(metric, '__name__', None) or str(metric)
                try:
                    fold_metrics[metric_key] = metric(y_fold_val, y_pred)
                except:
                    fold_metrics[metric_key] = 0.0
            
            fold_scores.append(fold_metrics)
            
            # Print fold results
            primary_metric = "f1_score" if self.task_type == "classification" else "r2_score"
            if primary_metric in fold_metrics:
                print(f"  Fold {fold + 1} {primary_metric}: {fold_metrics[primary_metric]:.4f}")
        
        # Calculate cross-validation statistics
        cv_results = self._compute_cv_statistics(fold_scores)
        self._print_cv_results(cv_results)
        
        # Train final model on full dataset
        print(f"\nTraining final model on full dataset...")
        self.internal_fit(X, y, None, None, self.model)
        
        print("✅ Final model trained successfully on full dataset!")
        
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
    
    def _compute_cv_statistics(self, fold_scores: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Compute cross-validation statistics (mean ± std) for each metric."""
        if not fold_scores:
            return {}
        
        # Get all metrics
        all_metrics = set()
        for scores in fold_scores:
            all_metrics.update(scores.keys())
        
        cv_stats = {}
        for metric in all_metrics:
            values = [scores.get(metric, 0.0) for scores in fold_scores]
            cv_stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values
            }
        
        return cv_stats
    
    def _print_cv_results(self, cv_results: Dict[str, Dict[str, float]]):
        """Print cross-validation results following scikit-learn format."""
        if not cv_results:
            return
        
        primary_metric = "f1_score" if self.task_type == "classification" else "r2_score"
        
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")
        
        # Print primary metric first
        if primary_metric in cv_results:
            stats = cv_results[primary_metric]
            print(f"🎯 {primary_metric.upper():15s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print()
        
        # Print other metrics
        for metric, stats in sorted(cv_results.items()):
            if metric != primary_metric:
                print(f"{metric:15s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        print(f"{'='*60}\n")
    
    def _create_cv_metrics_summary(self, cv_results: Dict[str, Dict[str, float]], fold_scores: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Create a metrics summary compatible with save_metrics function.
        """
        metrics_summary = []
        
        # Add overall CV summary as first row
        summary_row = {'fold': 'CV_MEAN'}
        for metric, stats in cv_results.items():
            summary_row[metric] = stats['mean']
            summary_row[f"{metric}_std"] = stats['std']
        metrics_summary.append(summary_row)
        
        # Add individual fold results
        for fold_idx, fold_metrics in enumerate(fold_scores):
            fold_row = {'fold': f'fold_{fold_idx + 1}'}
            fold_row.update(fold_metrics)
            metrics_summary.append(fold_row)
        
        return metrics_summary

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

    def evaluate(self, X, y):
        return self._evaluate(X, y)

    def predict(self, X):
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            raise AttributeError(f"Model of type {type(self.model)} does not have a predict method.")

    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError('Model does not support predict_proba')

    def fit(self, X, y):
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
            print(f"Starting {self.k_folds}-fold cross-validation for sklearn model")
            results = self._kfold_fit(X, y)
            self.cv_results = results  # Store for later access
            
            # Return training history for compatibility with save_metrics
            return results.get('training_history', [])
        else:
            # Single train/validation split
            import sys
            import os
            # Add the parent directory to path to import from utils
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from utils.data import train_val_test_split
            
            # Use train_val_test_split function
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
    
    def get_training_history(self) -> List[Dict[str, float]]:
        """
        Get training history in format compatible with save_metrics function.
        
        Returns:
            List of dictionaries with epoch-by-epoch or fold-by-fold metrics
        """
        if hasattr(self, 'cv_results') and self.cv_results and self.use_kfold:
            # Return CV-compatible format
            return self.cv_results.get('training_history', [])
        else:
            # Return original history
            return self.history

    def get_cv_scores(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Get cross-validation scores summary.
        
        Returns:
            Dictionary with mean, std, and individual values for each metric,
            or None if CV wasn't performed.
        """
        if not hasattr(self, 'cv_results') or not self.cv_results.get('cv_scores'):
            return None
        
        return self.cv_results['cv_scores']
        
class GeneralPipeline:
    """
    General pipeline for PyTorch models supporting classification and regression.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Union[str, Callable],  # Can be string name or callable
        optimizer_cls: Callable,
        optimizer_params: dict,
        scheduler_cls: Optional[Callable] = None,
        scheduler_params: Optional[dict] = None,
        augmentations: Optional[Callable] = None,
        metrics: Optional[List[Callable]] = None,
        task_type: str = "classification",
        device: str = "cpu",
        epochs: int = 10,
        batch_size: int = 32,
        early_stopping: Optional[int] = None,
        use_class_weights: bool = True,
        save_path: Optional[str] = None,  # Path to save model weights
        save_to_hf: bool = False,  # Whether to save to HuggingFace Hub
        hf_repo_name: Optional[str] = None,  # HuggingFace repository name
        hf_token: Optional[str] = None,  # HuggingFace token
        dropout: Optional[float] = None,  # Dropout rate for models
        use_kfold: bool = True,  # Whether to use k-fold cross-validation
        k_folds: int = 5,  # Number of folds for k-fold CV
        max_factor: float = 2.0,  # Data augmentation factor (majority_count / final_minority_count)
        random_state: int = 42,  # Random seed for reproducibility
    ):
        # If dropout is provided and model supports it, set it
        if dropout is not None and hasattr(model, 'net'):
            # For nn.Sequential models, set dropout for all Dropout layers
            for layer in model.net:
                if isinstance(layer, torch.nn.Dropout):
                    layer.p = dropout
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)
        if scheduler_cls:
            if scheduler_params is None:
                scheduler_params = {}
            self.scheduler = scheduler_cls(self.optimizer, **scheduler_params)
        else:
            self.scheduler = None
        self.augmentations = augmentations
        self.metrics = metrics or []
        self.task_type = task_type
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.use_class_weights = use_class_weights
        self.class_weights = None
        self.save_path = save_path
        self.save_to_hf = save_to_hf
        self.hf_repo_name = hf_repo_name
        self.hf_token = hf_token
        self.use_kfold = use_kfold
        self.k_folds = k_folds
        self.max_factor = max_factor
        self.random_state = random_state
        self.history = []
        self.train_losses = []
        self.val_losses = []
        self.cv_results = {}  # Store cross-validation results
        
        # Set random seeds for reproducibility
        self._set_seeds()

    def _set_seeds(self):
        """Set random seed for all libraries to ensure reproducibility."""
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _compute_f1_score(self, targets: np.ndarray, preds: np.ndarray) -> float:
        """Compute F1 score for classification tasks."""
        if self.task_type != "classification":
            return 0.0
        
        # Convert probabilities to predictions if necessary
        if preds.ndim > 1 and preds.shape[1] > 1:
            preds = np.argmax(preds, axis=1)
        
        # Use macro average for multi-class, binary for binary classification
        average = 'binary' if len(np.unique(targets)) == 2 else 'macro'
        
        try:
            return f1_score(targets, preds, average=average, zero_division=0)
        except:
            return 0.0

    def _compute_r2_score(self, targets: np.ndarray, preds: np.ndarray) -> float:
        """Compute R² score for regression tasks."""
        if self.task_type != "regression":
            return 0.0
        
        try:
            from sklearn.metrics import r2_score
            return r2_score(targets, preds)
        except:
            return 0.0

    def _compute_class_weights(self, y: np.ndarray) -> Optional[torch.Tensor]:
        """Compute class weights for imbalanced classification."""
        if not self.use_class_weights or self.task_type != "classification":
            return None
        
        # Get unique classes and their counts
        unique_classes = np.unique(y)
        if len(unique_classes) <= 1:
            return None  # No imbalance if only one class
        
        # Compute balanced class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y
        )
        
        # Convert to tensor and move to device
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        print(f"Computed class weights: {dict(zip(unique_classes, class_weights))}")
        return weights_tensor

    def _create_loss_function(self) -> Callable:
        """Create loss function with class weights if applicable."""
        import torch.nn as nn
        
        # If loss_fn is a string, instantiate it
        if isinstance(self.loss_fn, str):
            try:
                loss_class = getattr(nn, self.loss_fn)
                if self.task_type == "classification" and self.class_weights is not None:
                    # Try to create with weights
                    try:
                        return loss_class(weight=self.class_weights)
                    except TypeError:
                        # If weight parameter not supported, create without weights
                        print(f"Warning: {self.loss_fn} does not support weight parameter, using without weights")
                        return loss_class()
                else:
                    return loss_class()
            except AttributeError:
                raise ValueError(f"Loss function '{self.loss_fn}' not found in torch.nn")
        else:
            # If loss_fn is already a callable
            if self.task_type == "classification" and self.class_weights is not None:
                # If it's CrossEntropyLoss, create new instance with weights
                if isinstance(self.loss_fn, nn.CrossEntropyLoss):
                    return nn.CrossEntropyLoss(weight=self.class_weights)
                else:
                    # For other loss functions, return original
                    print(f"Warning: Loss function {type(self.loss_fn).__name__} does not support weight parameter")
                    return self.loss_fn
            else:
                return self.loss_fn

    def _kfold_fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation with weight averaging ensemble.
        
        Instead of final retraining, this approach:
        1. K-fold CV to get unbiased estimate of model performance
        2. Save best weights from each fold based on validation performance
        3. Average the weights to create final ensemble model
        
        This maintains validation criteria for best weights while avoiding 
        loss of cross-validation benefits from final retraining.
        """
        from sklearn.model_selection import StratifiedKFold
        
        # Use StratifiedKFold for classification to preserve class ratios
        if self.task_type == "classification":
            kfold = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
            cv_iterator = kfold.split(X, y)
        else:
            kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
            cv_iterator = kfold.split(X)
        
        fold_scores = []
        fold_weights = []  # Store best weights from each fold
        
        print(f"Starting {self.k_folds}-fold cross-validation with weight averaging ensemble...")
        
        for fold, (train_idx, val_idx) in enumerate(cv_iterator):
            print(f"Training fold {fold + 1}/{self.k_folds}")
            
            # Reset seeds before each fold for consistency
            self._set_seeds()
            
            # Split data for this fold
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Apply augmentations only to training portion of this fold
            if self.augmentations is not None:
                X_fold_train, y_fold_train = self.augmentations(X_fold_train, y_fold_train, 
                                                              max_factor=self.max_factor, 
                                                              random_state=self.random_state)
            
            # Reset model for each fold
            self._reset_model()
            
            # Train on this fold (validation set is used for early stopping)
            self._single_fit_internal(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
            
            # Save best weights from this fold (model already contains best weights from training)
            fold_weights.append(self._get_model_weights())
            
            # Evaluate final performance on validation set
            fold_metrics = self._evaluate_final_performance(X_fold_val, y_fold_val)
            fold_scores.append(fold_metrics)
            
            # Print fold results
            primary_metric = "f1_score" if self.task_type == "classification" else "r2_score"
            if primary_metric in fold_metrics:
                print(f"  Fold {fold + 1} {primary_metric}: {fold_metrics[primary_metric]:.4f}")
        
        # Calculate cross-validation statistics
        cv_results = self._compute_cv_statistics(fold_scores)
        self._print_cv_results(cv_results)
        
        # Create ensemble model by averaging weights from all folds
        print(f"\nCreating ensemble model by averaging weights from {self.k_folds} folds...")
        averaged_weights = self._average_model_weights(fold_weights)
        self._set_model_weights(averaged_weights)
        
        print("✅ Ensemble model created successfully!")
        print("   Model now contains averaged weights from all cross-validation folds")
        print("   This preserves validation criteria while creating a robust ensemble\n")
        
        return {
            'cv_scores': cv_results,
            'ensemble_method': 'weight_averaging',
            'n_folds_averaged': len(fold_weights),
            'fold_individual_scores': fold_scores,
            'training_history': self._create_cv_metrics_summary(cv_results, fold_scores)  # Add for compatibility
        }
    
    def _create_cv_metrics_summary(self, cv_results: Dict[str, Dict[str, float]], fold_scores: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        Create a metrics summary compatible with save_metrics function.
        
        This converts CV results into a format that can be saved as a DataFrame,
        providing both summary statistics and individual fold results.
        """
        metrics_summary = []
        
        # Add overall CV summary as first row
        summary_row = {'fold': 'CV_MEAN'}
        for metric, stats in cv_results.items():
            summary_row[metric] = stats['mean']
            summary_row[f"{metric}_std"] = stats['std']
        metrics_summary.append(summary_row)
        
        # Add individual fold results
        for fold_idx, fold_metrics in enumerate(fold_scores):
            fold_row = {'fold': f'fold_{fold_idx + 1}'}
            fold_row.update(fold_metrics)
            metrics_summary.append(fold_row)
        
        return metrics_summary
    
    def _evaluate_final_performance(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Evaluate final performance on validation set after training is complete."""
        self.model.eval()
        val_loader = self.prepare_data(X_val, y_val, train=False)
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                Xb, yb = batch
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                outputs = self.model(Xb)
                
                if self.task_type == "regression":
                    outputs = outputs.squeeze()
                else:
                    outputs = torch.softmax(outputs, dim=1)
                
                all_preds.append(outputs.cpu())
                all_targets.append(yb.cpu())
        
        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        
        # Compute primary metrics
        metrics = {}
        
        if self.task_type == "classification":
            metrics['f1_score'] = self._compute_f1_score(targets, preds)
        else:  # regression
            metrics['r2_score'] = self._compute_r2_score(targets, preds)
        
        # Compute additional metrics if provided
        for metric in self.metrics:
            metric_key = getattr(metric, 'name', None) or getattr(metric, '__name__', None) or str(metric)
            try:
                metrics[metric_key] = metric(targets, preds)
            except:
                metrics[metric_key] = 0.0
        
        return metrics
    
    def _compute_cv_statistics(self, fold_scores: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Compute cross-validation statistics (mean ± std) for each metric."""
        if not fold_scores:
            return {}
        
        # Get all metrics
        all_metrics = set()
        for scores in fold_scores:
            all_metrics.update(scores.keys())
        
        cv_stats = {}
        for metric in all_metrics:
            values = [scores.get(metric, 0.0) for scores in fold_scores]
            cv_stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values
            }
        
        return cv_stats
    
    def _print_cv_results(self, cv_results: Dict[str, Dict[str, float]]):
        """Print cross-validation results following scikit-learn format."""
        if not cv_results:
            return
        
        primary_metric = "f1_score" if self.task_type == "classification" else "r2_score"
        
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")
        
        # Print primary metric first
        if primary_metric in cv_results:
            stats = cv_results[primary_metric]
            print(f"🎯 {primary_metric.upper():15s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print()
        
        # Print other metrics
        for metric, stats in sorted(cv_results.items()):
            if metric != primary_metric:
                print(f"{metric:15s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        print(f"{'='*60}\n")

    def _get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get a deep copy of current model weights."""
        return {name: param.clone().detach().cpu() for name, param in self.model.state_dict().items()}
    
    def _set_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Set model weights from a weights dictionary."""
        # Move weights to correct device and load into model
        state_dict = {name: weight.to(self.device) for name, weight in weights.items()}
        self.model.load_state_dict(state_dict)
    
    def _average_model_weights(self, fold_weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Average model weights from multiple folds.
        
        Args:
            fold_weights: List of weight dictionaries from each fold
            
        Returns:
            Dictionary of averaged weights
        """
        if not fold_weights:
            raise ValueError("No fold weights provided for averaging")
        
        averaged_weights = {}
        param_names = list(fold_weights[0].keys())
        
        averaged_count = 0
        non_averaged_count = 0
        
        # Average each parameter across all folds
        for param_name in param_names:
            # Get all parameter tensors for this parameter name
            param_tensors = [fold_weights[i][param_name] for i in range(len(fold_weights))]
            first_tensor = param_tensors[0]
            
            # Check if all tensors have the same shape and dtype
            if all(t.shape == first_tensor.shape and t.dtype == first_tensor.dtype for t in param_tensors):
                
                # Handle different data types
                if first_tensor.dtype.is_floating_point:
                    # For floating point tensors, compute mean directly
                    param_stack = torch.stack(param_tensors)
                    averaged_weights[param_name] = torch.mean(param_stack, dim=0)
                    averaged_count += 1
                    
                elif first_tensor.dtype in [torch.int64, torch.int32, torch.long, torch.int]:
                    # For integer tensors, convert to float, average, then round back to int
                    float_tensors = [t.float() for t in param_tensors]
                    param_stack = torch.stack(float_tensors)
                    averaged_weights[param_name] = torch.mean(param_stack, dim=0)
                    averaged_count += 1
                                                
        print(f"  Successfully averaged {averaged_count}/{len(param_names)} parameters across {len(fold_weights)} folds")
        if non_averaged_count > 0:
            print(f"  Warning: {non_averaged_count} parameters could not be averaged and used first fold values")
        
        return averaged_weights

    def _select_best_epoch(self) -> int:
        """
        Select best epoch using centralized logic from utils.utils.
        This ensures consistency across all components.
        """
        return select_best_epoch(self.history, self.task_type)

    def _calculate_combined_score_for_epoch(self, target_epoch: int) -> float:
        """
        Calculate combined score for a specific epoch using centralized logic from utils.utils.
        Used for early stopping to ensure consistency.
        """
        return calculate_combined_score_for_epoch(self.history, target_epoch, self.task_type)

    def _reset_model(self):
        """Reset model weights and optimizer for k-fold training."""
        # Re-initialize model weights
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        self.model.apply(init_weights)
        
        # Reset optimizer
        self.optimizer = self.optimizer.__class__(self.model.parameters(), **self.optimizer.defaults)
        
        # Reset scheduler if exists
        if self.scheduler:
            scheduler_class = self.scheduler.__class__
            scheduler_params = {}
            if hasattr(self.scheduler, 'state_dict'):
                # Try to get original parameters from scheduler
                if hasattr(self.scheduler, 'last_epoch'):
                    scheduler_params = {}
            self.scheduler = scheduler_class(self.optimizer, **scheduler_params)

    def prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, train: bool = True) -> DataLoader:
        """Prepare data for training/evaluation. Augmentations handled separately."""
        return self.prepare_data_internal(X, y, train)
    
    def prepare_data_internal(self, X: np.ndarray, y: Optional[np.ndarray] = None, train: bool = True) -> DataLoader:
        """Internal data preparation without augmentations (assumes already applied)."""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            y_tensor = torch.tensor(y, dtype=torch.float32 if self.task_type == "regression" else torch.long)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)
        
        # Create generator for reproducible shuffling
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        
        loader: DataLoader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=train,
            generator=generator if train else None,
            num_workers=0 , # Ensure deterministic behavior
            drop_last=True if train else False
        )
        assert isinstance(loader.dataset, Sized), "Dataset must be Sized for len() to work."
        return loader

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model with optional k-fold cross-validation.
        
        Args:
            X: Training features
            y: Training targets
        
        Returns:
            For backward compatibility with save_metrics, always returns training history.
            Access CV results via get_cv_scores() method if needed.
        """
        if self.use_kfold:
            print(f"Starting {self.k_folds}-fold cross-validation with weight averaging ensemble")
            results = self._kfold_fit(X, y)
            self.cv_results = results  # Store for later access
            
            # Return training history for compatibility with save_metrics
            return results.get('training_history', [])
        else:
            # Single train/validation split
            import sys
            import os
            # Add the parent directory to path to import from utils
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from utils.data import train_val_test_split
            
            # Use train_val_test_split function
            if self.task_type == "classification":
                X_train, X_val, _, y_train, y_val, _ = train_val_test_split(
                    X, y, val_size=0.2, test_size=0, stratify=y, random_state=self.random_state
                )
            else:
                X_train, X_val, _, y_train, y_val, _ = train_val_test_split(
                    X, y, val_size=0.2, test_size=0, random_state=self.random_state
                )
            
            # Apply augmentations if provided
            if self.augmentations is not None:
                X_train, y_train = self.augmentations(X_train, y_train,
                                                    max_factor=self.max_factor,
                                                    random_state=self.random_state)
            
            return self._single_fit_internal(X_train, y_train, X_val, y_val)

    def _single_fit_internal(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Internal single training run (original fit logic) - assumes augmentations already applied."""
        # Compute class weights from training data (after augmentation)
        self.class_weights = self._compute_class_weights(y_train)
        
        # Create loss function with class weights
        loss_fn = self._create_loss_function()
        
        train_loader: DataLoader = self.prepare_data_internal(X_train, y_train, train=True)
        val_loader: Optional[DataLoader] = self.prepare_data_internal(X_val, y_val, train=False) if X_val is not None and y_val is not None else None
        
        best_weights = None
        patience = 0
        best_epoch = 0  # Track which epoch was selected as best
        epoch_weights = []  # Store weights from each epoch for post-training selection
        best_combined_score = float('inf')  # For combined score early stopping
        
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                Xb, yb = batch
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(Xb)
                if self.task_type == "regression":
                    outputs = outputs.squeeze()
                loss = loss_fn(outputs, yb)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * Xb.size(0)
            train_loss /= len(cast(TensorDataset, train_loader.dataset))
            self.train_losses.append(train_loss)
            
            val_loss = None
            metrics_dict = {}
            current_f1 = 0.0
            current_r2 = 0.0
            
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                all_preds = []
                all_targets = []
                with torch.no_grad():
                    for batch in val_loader:
                        Xb, yb = batch
                        Xb, yb = Xb.to(self.device), yb.to(self.device)
                        outputs = self.model(Xb)
                        if self.task_type == "regression":
                            outputs = outputs.squeeze()
                            all_preds.append(outputs.cpu())
                            loss = loss_fn(outputs, yb)
                        else:
                            # For classification: use raw logits for loss, probabilities for F1
                            loss = loss_fn(outputs, yb)  # Loss uses raw logits
                            probs = torch.softmax(outputs, dim=1)  # Convert to probabilities for F1
                            all_preds.append(probs.cpu())
                        all_targets.append(yb.cpu())
                        val_loss += loss.item() * Xb.size(0)
                val_loss /= len(cast(TensorDataset, val_loader.dataset))
                self.val_losses.append(val_loss)
                preds = torch.cat(all_preds).numpy()
                targets = torch.cat(all_targets).numpy()
                
                # Compute F1 score for classification tasks and R² for regression
                if self.task_type == "classification":
                    current_f1 = self._compute_f1_score(targets, preds)
                    current_r2 = 0.0  # Not needed for classification
                else:  # regression
                    current_r2 = self._compute_r2_score(targets, preds)
                    current_f1 = 0.0  # Not needed for regression
                
                for metric in self.metrics:
                    metric_key = getattr(metric, 'name', None) or getattr(metric, '__name__', None) or str(metric)
                    metrics_dict[metric_key] = metric(targets, preds)
                
                # Store raw scores in history - selection logic will be applied post-training
                # This ensures consistency between training and BenchmarkRunner selection
                
                # Early stopping based on combined score (consistent with final selection)
                if self.early_stopping and len(self.history) >= 2:  # Need at least 2 epochs for comparison
                    # Calculate combined score for current epoch using same logic as _select_best_epoch
                    current_combined_score = self._calculate_combined_score_for_epoch(len(self.history) - 1)
                    
                    if len(self.history) == 1:
                        best_combined_score = current_combined_score
                        patience = 0
                    else:
                        if current_combined_score < best_combined_score:
                            best_combined_score = current_combined_score
                            patience = 0
                        else:
                            patience += 1
                    
                    if patience >= self.early_stopping:
                        print(f"Early stopping at epoch {epoch+1} due to combined score plateau")
                        break
            
            # Print progress
            progress_str = f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                progress_str += f", Val Loss: {val_loss:.4f}"
            if self.task_type == "classification" and current_f1 > 0:
                progress_str += f", F1: {current_f1:.4f}"
            elif self.task_type == "regression" and current_r2 != 0:
                progress_str += f", R²: {current_r2:.4f}"
            print(progress_str)
            
            if self.scheduler:
                self.scheduler.step()
            
            # Save all metrics for this epoch
            epoch_metrics = {"loss": train_loss, "val_loss": val_loss}
            if self.task_type == "classification":
                epoch_metrics["f1_score"] = current_f1
            else:  # regression
                epoch_metrics["r2_score"] = current_r2
            epoch_metrics.update(metrics_dict)
            self.history.append(epoch_metrics)
            
            # Store weights from this epoch for potential selection later
            epoch_weights.append({name: param.clone().detach().cpu() for name, param in self.model.state_dict().items()})
        
        # Post-training: Apply selection logic to determine best epoch and weights
        if self.history and epoch_weights:
            best_epoch = self._select_best_epoch()
            print(f"Selected best epoch: {best_epoch + 1} based on combined score")
            
            # Use weights from the selected best epoch
            if best_epoch < len(epoch_weights):
                best_weights = epoch_weights[best_epoch]
            else:
                best_weights = epoch_weights[-1]  # Fallback to last epoch
        
        # Restore best weights after training
        if best_weights is not None:
            # Move weights to correct device and load into model  
            state_dict = {name: weight.to(self.device) for name, weight in best_weights.items()}
            self.model.load_state_dict(state_dict)
        
        # Save model after training if requested
        self.save_model_after_training()
        
        # Return full history (list of dicts)
        return self.history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        loader: DataLoader = self.prepare_data_internal(X, y, train=False)
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in loader:
                Xb, yb = batch
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                outputs = self.model(Xb)
                if self.task_type == "regression":
                    outputs = outputs.squeeze()
                else:
                    outputs = torch.softmax(outputs, dim=1)
                all_preds.append(outputs.cpu())
                all_targets.append(yb.cpu())
        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        results = {}
        for metric in self.metrics:
            results[metric.__name__] = metric(targets, preds)
        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        loader: DataLoader = self.prepare_data_internal(X, train=False)
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                Xb = batch[0].to(self.device)
                outputs = self.model(Xb)
                if self.task_type == "regression":
                    outputs = outputs.squeeze()
                else:
                    outputs = torch.softmax(outputs, dim=1)
                    outputs = outputs.argmax(dim=1)
                all_preds.append(outputs.cpu())
        return torch.cat(all_preds).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks.")
        loader: DataLoader = self.prepare_data_internal(X, train=False)
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                Xb = batch[0].to(self.device)
                outputs = self.model(Xb)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu())
        return torch.cat(all_probs).numpy()
    
    def get_training_history(self) -> List[Dict[str, float]]:
        """
        Get training history in format compatible with save_metrics function.
        
        Returns:
            List of dictionaries with epoch-by-epoch or fold-by-fold metrics
        """
        if hasattr(self, 'cv_results') and self.cv_results and self.use_kfold:
            # Return CV-compatible format
            return self.cv_results.get('training_history', [])
        else:
            # Return original epoch-by-epoch history
            return self.history

    def get_cv_scores(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Get cross-validation scores summary.
        
        Returns:
            Dictionary with mean, std, and individual values for each metric,
            or None if CV wasn't performed.
        """
        if not hasattr(self, 'cv_results') or not self.cv_results.get('cv_scores'):
            return None
        
        return self.cv_results['cv_scores']

    def save_model(self, path: Optional[str] = None) -> None:
        """Save model weights to a file."""
        save_path = path or self.save_path
        if save_path is None:
            raise ValueError("No save path provided. Set save_path in __init__ or pass path parameter.")
        
        # If save_path is a directory, create a default filename
        if os.path.isdir(save_path) or not save_path.endswith('.pt'):
            model_name = getattr(self.model, '__class__', type(self.model)).__name__
            filename = f"{model_name}_model.pt"
            save_path = os.path.join(save_path, filename)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'class_weights': self.class_weights,
            'task_type': self.task_type,
            'history': self.history,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'device': self.device,
        }, save_path)
        print(f"Model saved to: {save_path}")

    def load_model(self, path: str) -> None:
        """Load model weights from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.class_weights = checkpoint.get('class_weights')
        self.history = checkpoint.get('history', {"train_loss": [], "val_loss": [], "metrics": []})
        
        print(f"Model loaded from: {path}")

    def save_to_huggingface(self, repo_name: Optional[str] = None, token: Optional[str] = None) -> None:
        """Save model to HuggingFace Hub."""
        if not self.save_to_hf:
            print("save_to_hf is False. Set save_to_hf=True in __init__ to enable HuggingFace saving.")
            return
        
        repo_name = repo_name or self.hf_repo_name
        token = token or self.hf_token
        
        if repo_name is None:
            raise ValueError("No HuggingFace repository name provided. Set hf_repo_name in __init__ or pass repo_name parameter.")
        
        if token is None:
            import os
            token = os.getenv("HF_TOKEN")
            if token is None:
                raise ValueError("No HuggingFace token provided. Set hf_token in __init__, pass token parameter, or set HF_TOKEN environment variable.")
        
        try:
            from huggingface_hub import HfApi
            
            # Create a temporary directory for saving
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save model state dict
                model_path = os.path.join(temp_dir, "pytorch_model.bin")
                torch.save(self.model.state_dict(), model_path)
                
                # Create config file
                config = {
                    "model_type": "custom",
                    "task_type": self.task_type,
                    "model_class": self.model.__class__.__name__,
                }
                
                config_path = os.path.join(temp_dir, "config.json")
                import json
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                # Upload to HuggingFace Hub
                api = HfApi(token=token)
                api.upload_folder(
                    folder_path=temp_dir,
                    repo_id=repo_name,
                    repo_type="model"
                )
                
            print(f"Model uploaded to HuggingFace Hub: {repo_name}")
            
        except ImportError as e:
            print(f"Error: {e}")
            print("Install required packages: pip install transformers huggingface_hub")
        except Exception as e:
            print(f"Error uploading to HuggingFace Hub: {e}")

    def save_model_after_training(self) -> None:
        """Save model after training is complete."""
        if self.save_path:
            self.save_model()
        
        if self.save_to_hf:
            self.save_to_huggingface()
