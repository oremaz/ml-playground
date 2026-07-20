from typing import List, Dict, Any, Optional, Union
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import torch
import logging
from pipelines_torch.base import GeneralPipeline
from utils.utils import save_model, save_metrics, model_exists

logger = logging.getLogger(__name__)

def set_all_seeds(seed: int):
    """Set random seed for all libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class BenchmarkRunner:
    """
    Runs grid search over models, augmentations, and metrics, collects results in a DataFrame.
    """
    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        augmentations: List[Optional[Any]],
        metrics: Optional[List[Any]] = None,
        task_type: str = "classification",
        device: str = "cpu",
        epochs: Union[int, List[int], Dict[str, int]] = 10,
        batch_size: int = 32,
        early_stopping: Optional[int] = None,
        use_class_weights: bool = True,
        save_to_hf: bool = False,  # Whether to save to HuggingFace Hub
        hf_repo_name: Optional[str] = None,  # HuggingFace repository name
        hf_token: Optional[str] = None,  # HuggingFace token
        dropout: Optional[float] = None,  # Dropout rate for models
        weight_decay: Optional[float] = None,  # L2 regularization for optimizer
        learning_rate: Optional[float] = None,  # Learning rate for optimizer
        use_kfold: bool = True,  # Whether to use k-fold cross-validation
        k_folds: int = 5,  # Number of folds for k-fold CV
        path_start: Optional[str] = None,  # Subfolder within results directory for organizing experiments
        max_factor: float = 2.0,  # Data augmentation factor (majority_count / final_minority_count)
        random_state: int = 42,  # Random seed for reproducibility
        use_mixed_precision: bool = True,  # Use automatic mixed precision for faster training
        use_wandb: bool = False,  # Log per-run training history to Weights & Biases
        wandb_project: Optional[str] = None,  # W&B project (defaults to utils.wandb_utils.DEFAULT_PROJECT)
    ):
        self.model_configs = model_configs
        self.augmentations = augmentations
        self.metrics = metrics
        self.task_type = task_type
        self.device = device
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.use_class_weights = use_class_weights
        self.save_to_hf = save_to_hf
        self.hf_repo_name = hf_repo_name
        self.hf_token = hf_token
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.use_kfold = use_kfold
        self.k_folds = k_folds
        self.path_start = path_start
        self.max_factor = max_factor
        self.random_state = random_state
        self.use_mixed_precision = use_mixed_precision
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        # Normalize epoch configuration across models
        self._epochs_scalar: Optional[int] = None
        self._epochs_sequence: Optional[List[int]] = None
        self._epochs_map: Optional[Dict[str, int]] = None

        if isinstance(epochs, int):
            self._epochs_scalar = epochs
        elif isinstance(epochs, (list, tuple)):
            if len(epochs) != len(model_configs):
                raise ValueError(
                    "When providing epochs as a list/tuple, its length must match model_configs."
                )
            self._epochs_sequence = list(epochs)
        elif isinstance(epochs, dict):
            model_names = {cfg["name"] for cfg in model_configs}
            epoch_names = set(epochs.keys())
            if model_names != epoch_names:
                missing = model_names - epoch_names
                extra = epoch_names - model_names
                details = []
                if missing:
                    details.append(f"missing entries for: {sorted(missing)}")
                if extra:
                    details.append(f"unexpected entries: {sorted(extra)}")
                message = "; ".join(details) if details else ""
                raise ValueError(
                    "Epochs dict keys must exactly match model names in model_configs." + (
                        f" {message}" if message else ""
                    )
                )
            self._epochs_map = dict(epochs)
        else:
            raise TypeError("epochs must be int, list/tuple of ints, or dict mapping model names to ints")

        # Set all random seeds for reproducibility
        set_all_seeds(self.random_state)

    
    def run(self, X, y) -> pd.DataFrame:
        """
        Run benchmark with given data.
        
        Args:
            X: Complete feature dataset
            y: Complete target dataset
            
        Returns:
            DataFrame with benchmark results
        """
        self.save_path = None
        if self.path_start:
            self.save_path = f"results/{self.path_start}"
            # Create the path if it doesn't exist
            import os
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path, exist_ok=True)
        for model_idx, model_cfg in enumerate(tqdm(self.model_configs, desc="Models")):
            model_name = model_cfg["name"]
            model_class = model_cfg["class"]
            model_params = model_cfg.get("params", {}).copy()
            # Add dropout to model params if specified
            if self.dropout is not None:
                model_params["dropout"] = self.dropout

            epochs = self._resolve_epochs(model_name, model_idx)
            for aug in tqdm(self.augmentations, desc="Augmentations", leave=False):
                if isinstance(aug, (tuple, list)) and len(aug) == 2:
                    aug_name, aug_fn = aug
                else:
                    aug_fn = aug
                    aug_name = aug.__name__ if aug is not None else "none"
                
                # Check if model already exists
                lr_value = self.learning_rate if self.learning_rate is not None else 1e-4
                metadata_core = {
                    "model_name": model_name,
                    "augmentation_name": aug_name,
                    "task_type": self.task_type,
                    "epochs": epochs,
                    "learning_rate": lr_value,
                    "weight_decay": self.weight_decay,
                    "dropout": self.dropout,
                    "batch_size": self.batch_size,
                    "use_kfold": self.use_kfold,
                    "k_folds": self.k_folds,
                }
                if model_exists(metadata_core, path_start=self.path_start):
                    logger.info("Skipping %s | %s (checkpoint already exists)", model_name, aug_name)
                    continue
                
                logger.info("Running Model: %s | Augmentation: %s", model_name, aug_name)
                # Print class distribution for classification tasks
                if self.task_type == "classification":
                    unique, counts = np.unique(y, return_counts=True)
                    logger.info("Class distribution: %s", dict(zip(unique, counts)))
                # Instantiate model
                model = model_class(**model_params)
                # Choose loss and optimizer
                import torch.optim as optim
                import torch.nn as nn
                if self.task_type == "classification":
                    loss_fn = "CrossEntropyLoss"
                else:
                    loss_fn = "MSELoss"
                optimizer_cls = optim.Adam
                optimizer_params = {"lr": lr_value}
                if self.weight_decay is not None:
                    optimizer_params["weight_decay"] = self.weight_decay
                # Dynamically select pipeline type
                from pipelines_torch.base import GeneralPipelineSklearn
                # Some registry entries expose factories (e.g., lambdas) instead of raw classes,
                # so we inspect the instantiated model rather than the constructor.
                is_torch_model = isinstance(model, torch.nn.Module)
                if is_torch_model:
                    pipeline = GeneralPipeline(
                        model=model,
                        loss_fn=loss_fn,
                        optimizer_cls=optimizer_cls,
                        optimizer_params=optimizer_params,
                        scheduler_cls=None,
                        scheduler_params=None,
                        augmentations=aug_fn,
                        metrics=self.metrics,
                        task_type=self.task_type,
                        device=self.device,
                        epochs=epochs,
                        batch_size=self.batch_size,
                        early_stopping=self.early_stopping,
                        use_class_weights=self.use_class_weights,
                        save_path=self.save_path,
                        save_to_hf=self.save_to_hf,
                        hf_repo_name=self.hf_repo_name,
                        hf_token=self.hf_token,
                        dropout=self.dropout,
                        use_kfold=self.use_kfold,
                        k_folds=self.k_folds,
                        max_factor=self.max_factor,
                        random_state=self.random_state,
                        use_mixed_precision=self.use_mixed_precision,
                    )
                else:
                    pipeline = GeneralPipelineSklearn(
                        model=model,
                        metrics=self.metrics,
                        task_type=self.task_type,
                        use_class_weights=self.use_class_weights,
                        augmentations=aug_fn,
                        max_factor=self.max_factor,
                        random_state=self.random_state,
                        use_kfold=self.use_kfold,
                        k_folds=self.k_folds,
                    )
                wandb_run = None
                if self.use_wandb:
                    from utils.wandb_utils import init_wandb_run
                    wandb_run = init_wandb_run(
                        project=self.wandb_project,
                        name=f"{self.path_start or 'bench'}-{model_name}-{aug_name}",
                        config=metadata_core,
                        group=self.path_start,
                    )
                # Train with complete data - pipeline will handle augmentations and splits internally
                training_history = pipeline.fit(X, y)
                # Print class weights if available
                if self.task_type == "classification" and hasattr(pipeline, 'class_weights') and pipeline.class_weights is not None:
                    logger.info("Computed class weights: %s", pipeline.class_weights)
                
                # Save model weights if specified
                entry = save_model(pipeline.model, path_start=self.path_start, metadata_core=metadata_core)
                
                # Save metrics
                save_metrics(training_history, entry["checkpoint_id"], path_start=self.path_start)

                if wandb_run is not None:
                    from utils.wandb_utils import log_history, log_summary, finish_run
                    log_history(wandb_run, training_history)
                    wandb_run.summary["checkpoint_id"] = entry["checkpoint_id"]
                    cv_scores = pipeline.get_cv_scores() if hasattr(pipeline, "get_cv_scores") else None
                    if cv_scores:
                        log_summary(wandb_run, {f"cv_{m}_mean": s["mean"] for m, s in cv_scores.items()})
                    finish_run(wandb_run)
                

    def _resolve_epochs(self, model_name: str, model_idx: int) -> int:
        """Return the epoch count for a given model based on the configured scheme."""
        if self._epochs_scalar is not None:
            epochs = self._epochs_scalar
        elif self._epochs_sequence is not None:
            epochs = self._epochs_sequence[model_idx]
        elif self._epochs_map is not None:
            epochs = self._epochs_map[model_name]
        else:
            raise RuntimeError("Epoch configuration is not initialized correctly.")

        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(f"Epochs must be a positive int for model '{model_name}', got {epochs}.")

        return epochs
