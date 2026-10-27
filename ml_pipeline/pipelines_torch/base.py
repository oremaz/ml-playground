"""PyTorch training and stable public pipeline imports."""

import logging
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Callable, List, Optional, Dict, Any, Tuple, Union
import numpy as np
import random
from functools import partial
from collections.abc import Sized
from typing import cast
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from ml_pipeline.utils.utils import select_best_epoch
from ml_pipeline.utils.data import train_val_test_split

logger = logging.getLogger("ml_pipeline.pipelines_torch.base")


from .evaluation import Evaluate, ComputeScore, SimplePredictor
from .sklearn_pipeline import GeneralPipelineSklearn


class GeneralPipeline(Evaluate):
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
        use_mixed_precision: bool = True,  # Use automatic mixed precision for faster training
    ):
        super().__init__()
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
        self.use_mixed_precision = use_mixed_precision
        
        # Initialize mixed precision training
        self.scaler = None
        self.autocast = None
        if self.use_mixed_precision and device != "cpu":
            device_str = str(device)
            device_type = "cuda" if device_str.startswith("cuda") else None
            if device_type is not None:
                try:
                    self.scaler = torch.amp.GradScaler(device=device_type)
                    self.autocast = partial(torch.amp.autocast, device_type=device_type)
                except (TypeError, AttributeError):
                    # Older PyTorch releases do not accept the device_type kwarg or torch.amp is missing
                    if device_type == "cuda" and hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
                        self.scaler = torch.cuda.amp.GradScaler()
                        self.autocast = torch.cuda.amp.autocast
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
        logger.info("Computed class weights: %s", dict(zip(unique_classes, class_weights)))
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
                        logger.warning("%s does not support weight parameter, using without weights", self.loss_fn)
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
                    logger.warning("Loss function %s does not support weight parameter", type(self.loss_fn).__name__)
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
        
        logger.info("Starting %d-fold cross-validation with weight averaging ensemble...", self.k_folds)

        for fold, (train_idx, val_idx) in enumerate(cv_iterator):
            logger.info("Training fold %d/%d", fold + 1, self.k_folds)
            
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
            metric_value = fold_metrics.get(primary_metric)
            if metric_value is not None:
                logger.info("  Fold %d %s: %.4f", fold + 1, primary_metric, metric_value)
        
        # Calculate cross-validation statistics
        cv_results = self._compute_cv_statistics(fold_scores)
        self._print_cv_results(cv_results)
        
        # Create ensemble model by averaging weights from all folds
        logger.info("Creating ensemble model by averaging weights from %d folds...", self.k_folds)
        averaged_weights = self._average_model_weights(fold_weights)
        self._set_model_weights(averaged_weights)

        logger.info("Ensemble model created successfully (averaged weights from all CV folds)")
        
        return {
            'cv_scores': cv_results,
            'ensemble_method': 'weight_averaging',
            'n_folds_averaged': len(fold_weights),
            'fold_individual_scores': fold_scores,
            'training_history': self._create_cv_metrics_summary(cv_results, fold_scores)  # Add for compatibility
        }
    
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
            metrics['f1_score'] = ComputeScore.f1(targets, preds, self.task_type)
            metrics['roc_auc'] = ComputeScore.roc_auc(targets, preds, self.task_type)
            metrics['pr_auc'] = ComputeScore.pr_auc(targets, preds, self.task_type)
        else:  # regression
            metrics['r2_score'] = ComputeScore.r2(targets, preds, self.task_type)
        
        # Compute additional metrics if provided
        for metric in self.metrics:
            metric_key = getattr(metric, 'name', None) or getattr(metric, '__name__', None) or str(metric)
            try:
                metrics[metric_key] = metric(targets, preds)
            except (ValueError, TypeError, IndexError):
                metrics[metric_key] = 0.0
            except Exception as e:
                logger.warning("Unexpected error computing metric %s: %s", metric_key, e)
                metrics[metric_key] = 0.0
        
        return metrics
    
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
                                                
        logger.info("  Successfully averaged %d/%d parameters across %d folds", averaged_count, len(param_names), len(fold_weights))
        if non_averaged_count > 0:
            logger.warning("  %d parameters could not be averaged and used first fold values", non_averaged_count)
        
        return averaged_weights

    def _select_best_epoch(self) -> int:
        """
        Select best epoch using centralized logic from ml_pipeline.utils.utils.
        This ensures consistency across all components.
        """
        return select_best_epoch(self.history, self.task_type, metric=self._default_selection_metric())

    def _get_current_selection_metric(self, epoch_metrics: Dict[str, Any]) -> Optional[float]:
        """Extract the current selection metric value from epoch metrics."""
        metric_name = self._default_selection_metric()
        value = epoch_metrics.get(metric_name)
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        return float(value)

    @staticmethod
    def _is_metric_improved(current: float, best: float) -> bool:
        """Check if current metric is better than best (higher is better)."""
        return current > best

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
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        if y is not None:
            y_tensor = torch.as_tensor(y, dtype=torch.float32 if self.task_type == "regression" else torch.long)
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
        # Check if the model has a custom fit method (e.g., HuggingFace models)
        # We detect this by checking if the model's class directly defines a fit method
        # (not inherited from nn.Module, which doesn't have fit)
        model_class = type(self.model)
        has_custom_fit = (
            'fit' in model_class.__dict__ and 
            callable(getattr(self.model, 'fit'))
        )
        
        if has_custom_fit:
            # For models with custom fit methods (like HuggingFace wrappers),
            # delegate training to the model itself
            logger.info("Using model's custom training method")
            
            # Pass through relevant training parameters
            fit_kwargs = {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.optimizer.param_groups[0]['lr'] if hasattr(self, 'optimizer') else 1e-4,
                **({'output_dir': self.restart_path} if getattr(self, 'restart_path', None) else {}),
            }
            
            # Call the model's fit method
            training_history = self.model.fit(X, y, **fit_kwargs)
            
            # Store the history
            if isinstance(training_history, list):
                self.history = training_history
            else:
                # If the model doesn't return history, create a minimal one
                self.history = [{'loss': 0.0}]
            
            return self.history
        
        # Standard PyTorch training loop for regular nn.Module models
        if self.use_kfold:
            logger.info("Starting %d-fold cross-validation with weight averaging ensemble", self.k_folds)
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
            
            # Apply augmentations if provided
            if self.augmentations is not None:
                X_train, y_train = self.augmentations(X_train, y_train,
                                                    max_factor=self.max_factor,
                                                    random_state=self.random_state)
            
            return self._single_fit_internal(X_train, y_train, X_val, y_val)

    def _single_fit_internal(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        from ml_pipeline.utils.training_checkpoint import StopRequest
        with StopRequest() as self._stop_request:
            return self._single_fit_restart(X_train, y_train, X_val, y_val)

    def _single_fit_restart(self, X_train, y_train, X_val=None, y_val=None):
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
        best_metric_value = None  # Track best metric for live weight selection
        best_val_loss = float('inf')  # For validation loss early stopping
        
        from pathlib import Path
        from ml_pipeline.utils.training_checkpoint import (
            CheckpointClock, CheckpointPause, atomic_save, data_fingerprint,
            load_checkpoint, restore_rng, rng_state,
        )
        restart_path = getattr(self, "restart_path", None)
        restart_file = Path(restart_path) / "restart.pt" if restart_path else None
        context = {"run": getattr(self, "restart_context", None), "epochs": self.epochs,
                   "batch_size": self.batch_size, "seed": self.random_state,
                   "shape": X_train.shape, "labels": y_train.tolist(),
                   "validation_labels": y_val.tolist() if y_val is not None else None,
                   "data_fingerprint": data_fingerprint(X_train, y_train, X_val, y_val)}
        restart = load_checkpoint(restart_file, context) if restart_file else None
        epoch_start, batch_start, train_loss = 0, 0, 0.0
        epoch_loader_rng = train_loader.generator.get_state()
        if restart is not None:
            self.model.load_state_dict(restart["model"])
            self.optimizer.load_state_dict(restart["optimizer"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(restart["scheduler"])
            if self.scaler is not None:
                self.scaler.load_state_dict(restart["scaler"])
            self.history, self.train_losses, self.val_losses = restart["history"], restart["train_losses"], restart["val_losses"]
            best_weights, best_epoch = restart["best_weights"], restart["best_epoch"]
            best_metric_value, best_val_loss, patience = restart["best_metric_value"], restart["best_val_loss"], restart["patience"]
            epoch_start, batch_start, train_loss = restart["epoch"], restart["batch"], restart["train_loss"]
            epoch_loader_rng = restart["loader_rng"]
            train_loader.generator.set_state(epoch_loader_rng)
            restore_rng(restart["rng"])
        clock = CheckpointClock()

        def save(epoch, batch, loss, loader_rng):
            if restart_file is not None:
                atomic_save({
                    "config": context, "model": {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()},
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                    "scaler": self.scaler.state_dict() if self.scaler else None,
                    "epoch": epoch, "batch": batch, "train_loss": loss, "loader_rng": loader_rng,
                    "history": self.history, "train_losses": self.train_losses, "val_losses": self.val_losses,
                    "best_weights": best_weights, "best_epoch": best_epoch,
                    "best_metric_value": best_metric_value, "best_val_loss": best_val_loss, "patience": patience,
                    "rng": rng_state(),
                }, restart_file)
                clock.saved()

        save(epoch_start, batch_start, train_loss, epoch_loader_rng)
        for epoch in range(epoch_start, self.epochs):
            self.model.train()
            if not batch_start:
                train_loss = 0.0
                epoch_loader_rng = train_loader.generator.get_state()
            iterator = iter(train_loader)
            for _ in range(batch_start):
                next(iterator)
            if restart is not None and batch_start:
                restore_rng(restart["rng"])
            for batch_index, batch in enumerate(iterator, start=batch_start):
                Xb, yb = batch
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                
                # Use mixed precision if available
                if self.scaler is not None and self.autocast is not None:
                    with self.autocast():
                        outputs = self.model(Xb)
                        if self.task_type == "regression":
                            outputs = outputs.squeeze()
                        loss = loss_fn(outputs, yb)
                    if not torch.isfinite(loss).all():
                        raise FloatingPointError(f"Nonfinite training loss at epoch {epoch + 1}, batch {batch_index + 1}; last safe restart checkpoint retained")
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(Xb)
                    if self.task_type == "regression":
                        outputs = outputs.squeeze()
                    loss = loss_fn(outputs, yb)
                    if not torch.isfinite(loss).all():
                        raise FloatingPointError(f"Nonfinite training loss at epoch {epoch + 1}, batch {batch_index + 1}; last safe restart checkpoint retained")
                    loss.backward()
                    self.optimizer.step()
                    
                train_loss += loss.item() * Xb.size(0)
                if clock.due() or self._stop_request.requested:
                    save(epoch, batch_index + 1, train_loss, epoch_loader_rng)
                    if self._stop_request.requested:
                        raise CheckpointPause("PyTorch optimizer, scaler, RNG and minibatch progress saved")
            save(epoch, len(train_loader), train_loss, epoch_loader_rng)
            train_loss /= len(cast(TensorDataset, train_loader.dataset))
            self.train_losses.append(train_loss)
            
            val_loss = None
            extra_metrics: Dict[str, float] = {}
            current_f1: Optional[float] = None
            current_r2: Optional[float] = None
            current_roc_auc: Optional[float] = None
            current_pr_auc: Optional[float] = None
            
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
                        if not torch.isfinite(loss).all():
                            raise FloatingPointError("Nonfinite validation loss; last safe restart checkpoint retained")
                        val_loss += loss.item() * Xb.size(0)
                val_loss /= len(cast(TensorDataset, val_loader.dataset))
                self.val_losses.append(val_loss)
                preds = torch.cat(all_preds).numpy()
                targets = torch.cat(all_targets).numpy()
                
                # Compute metrics for validation set
                if self.task_type == "classification":
                    current_f1 = ComputeScore.f1(targets, preds, self.task_type)
                    current_roc_auc = ComputeScore.roc_auc(targets, preds, self.task_type)
                    current_pr_auc = ComputeScore.pr_auc(targets, preds, self.task_type)
                    current_r2 = None  # Not needed for classification
                else:  # regression
                    current_r2 = ComputeScore.r2(targets, preds, self.task_type)
                    current_f1 = None  # Not needed for regression

                for metric in self.metrics:
                    metric_key = getattr(metric, 'name', None) or getattr(metric, '__name__', None) or str(metric)
                    try:
                        extra_metrics[metric_key] = metric(targets, preds)
                    except (ValueError, TypeError, IndexError):
                        extra_metrics[metric_key] = 0.0
                    except Exception as e:
                        logger.warning("Unexpected error computing metric %s: %s", metric_key, e)
                        extra_metrics[metric_key] = 0.0
                
                # Store raw scores in history - selection logic will be applied post-training
                # This ensures consistency between training and BenchmarkRunner selection
                
                # Early stopping based on validation loss
                if self.early_stopping:
                    if val_loss is not None and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience = 0
                    else:
                        patience += 1
                    
                    if patience >= self.early_stopping:
                        logger.info("Early stopping at epoch %d due to validation loss plateau", epoch + 1)
                        break
            
            # Print progress
            progress_str = f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                progress_str += f", Val Loss: {val_loss:.4f}"
            if self.task_type == "classification":
                if current_f1 is not None:
                    progress_str += f", F1: {current_f1:.4f}"
                if current_roc_auc is not None:
                    progress_str += f", ROC-AUC: {current_roc_auc:.4f}"
                if current_pr_auc is not None:
                    progress_str += f", PR-AUC: {current_pr_auc:.4f}"
            elif self.task_type == "regression" and current_r2 is not None:
                progress_str += f", R²: {current_r2:.4f}"
            logger.info(progress_str)
            
            if self.scheduler:
                self.scheduler.step()
            
            # Save all metrics for this epoch
            epoch_metrics = {"loss": train_loss, "val_loss": val_loss}
            if self.task_type == "classification":
                epoch_metrics["f1_score"] = current_f1
                epoch_metrics["roc_auc"] = current_roc_auc
                epoch_metrics["pr_auc"] = current_pr_auc
            else:  # regression
                epoch_metrics["r2_score"] = current_r2
            epoch_metrics.update(extra_metrics)
            self.history.append(epoch_metrics)
            
            # Live best-weight tracking: keep only the best weights to avoid O(epochs) memory
            current_metric_value = self._get_current_selection_metric(epoch_metrics)
            if current_metric_value is not None:
                is_better = (best_metric_value is None) or self._is_metric_improved(current_metric_value, best_metric_value)
                if is_better:
                    best_metric_value = current_metric_value
                    best_epoch = epoch
                    best_weights = {name: param.clone().detach().cpu() for name, param in self.model.state_dict().items()}
            elif best_weights is None:
                # No metric available yet (e.g., no val set); keep latest weights
                best_weights = {name: param.clone().detach().cpu() for name, param in self.model.state_dict().items()}
                best_epoch = epoch

            batch_start, restart = 0, None
            save(epoch + 1, 0, 0.0, train_loader.generator.get_state())
            if self._stop_request.requested:
                raise CheckpointPause("PyTorch epoch and validation state saved")

        if best_weights is not None:
            logger.info("Selected best epoch: %d based on selection metric", best_epoch + 1)
        
        # Restore best weights after training
        if best_weights is not None:
            # Move weights to correct device and load into model  
            state_dict = {name: weight.to(self.device) for name, weight in best_weights.items()}
            self.model.load_state_dict(state_dict)
        
        # Save model after training if requested
        self.save_model_after_training()

        # Release GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Return full history (list of dicts)
        return self.history

    def save_to_huggingface(self, repo_name: Optional[str] = None, token: Optional[str] = None) -> None:
        """Save model to HuggingFace Hub."""
        if not self.save_to_hf:
            logger.warning("save_to_hf is False. Set save_to_hf=True in __init__ to enable HuggingFace saving.")
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
                
            logger.info("Model uploaded to HuggingFace Hub: %s", repo_name)

        except ImportError as e:
            logger.error("Missing dependency: %s. Install: pip install transformers huggingface_hub", e)
        except Exception as e:
            logger.error("Error uploading to HuggingFace Hub: %s", e)

    def save_model_after_training(self) -> None:
        """Save model after training is complete."""
        
        if self.save_to_hf:
            self.save_to_huggingface()
