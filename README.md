
# Food Prediction & Benchmarking Suite (`food_preds`)

Comprehensive machine learning experimentation and benchmarking suite for food-related prediction tasks, including recipe difficulty, meal type, nutrient regression, and cooking time estimation. This directory provides modular pipelines, advanced data augmentation, model registries, and reproducible workflows for robust ML experimentation.

---

## Features

### 🧪 End-to-End ML Benchmarking
- **Notebook-driven workflow**: Full ML lifecycle in `benchmark_results.ipynb` (data prep, augmentation, model selection, training, evaluation, visualization)
- **Automated grid search**: Systematic benchmarking of models and augmentations with cross-validation and result aggregation
- **Reproducibility**: Deterministic splits, seed control, and experiment tracking

### 🔄 Advanced Data Augmentation
- **Tabular augmentation**: SMOTE (all variants), Mixup, hybrid samplers, and cleaning (see `data_augmentation/augmentations.py`)
- **Text augmentation**: LLM-based (OpenAI, Gemini, HuggingFace) and classical (synonym, backtranslation, EDA) methods (`text_aug.py`)
- **Image augmentation**: Classical (flip, rotate, noise) and Albumentations-based pipelines (`image_aug.py`)
- **Registry pattern**: Easily add new augmentation methods; configure via registry and YAML/dict
- **Batch and single-sample APIs**: Consistent interface for all augmentation types

### 🏗️ Modular Model Pipelines
- **PyTorch & sklearn pipelines**: Unified `GeneralPipeline` and `GeneralPipelineSklearn` for all model types
- **Model registry**: Plug-and-play support for MLP, XGBoost, LightGBM, RandomForest, HuggingFace LoRA/QLoRA, Llama.cpp, and more (`models.py`)
- **K-fold CV & ensembling**: Built-in cross-validation, weight averaging, and metrics tracking
- **Flexible metrics**: Custom and standard metrics for classification/regression

### 📊 Utilities & Visualization
- **Data loading/splitting**: Robust, stratified, and reproducible splits (`utils/data.py`)
- **Metrics**: Accuracy, F1, R², MSE, and more (`utils/metrics.py`)
- **Visualization**: Training curves, confusion matrices, and result plots (`utils/visualization.py`)
- **Model persistence**: Save/load models, HuggingFace Hub integration

---

## Directory Structure

```
food_preds/
├── benchmark_results.ipynb         # End-to-end ML workflow notebook
├── recipes_df*.csv                 # Datasets (main and test splits)
├── data_augmentation/              # All augmentation modules and docs
│   ├── augmentation_example.py     # Example script for all augmentations
│   ├── AUGMENTATION_README.md      # Detailed augmentation documentation
│   ├── augmentations.py            # Tabular augmentation (SMOTE, Mixup, etc.)
│   ├── image_aug.py                # Image augmentation (classical, Albumentations)
│   ├── text_aug.py                 # Text augmentation (LLM, classical)
├── pipelines_torch/                # Model pipelines and wrappers
│   ├── __init__.py
│   ├── base.py                     # GeneralPipeline, GeneralPipelineSklearn
│   ├── benchmark.py                # BenchmarkRunner (grid search, result aggregation)
│   ├── models.py                   # Model registry and wrappers
├── results/                        # Experiment results (CSV, per-task)
│   ├── ...                         # Results for difficulty, meal type, nutrients, time, etc.
├── utils/                          # Utilities for data, metrics, reproducibility
│   ├── data.py
│   ├── metrics.py
│   ├── utils.py
│   ├── visualization.py
```

---


## Architecture & Workflow

### 1. Data Preparation
- Load and preprocess datasets (`recipes_df.csv`, etc.)
- Use `utils/data.py` for robust train/val/test splits (stratified, reproducible)

### 2. Data Augmentation
- Select augmentation via registry (see `AUGMENTATION_README.md`)
- Tabular: SMOTE (regular, borderline, kmeans, SVM), Mixup, hybrid samplers, cleaning
- Text: LLM-based (OpenAI, Gemini, HuggingFace), classical (synonym, EDA, backtranslation)
- Image: Classical and Albumentations pipelines
- Configure augmentations via dict/YAML or registry name

### 3. Model Selection & Pipeline
- Choose model from registry (`models.py`): MLP, XGBoost, LightGBM, RandomForest, HuggingFace LoRA/QLoRA, Llama.cpp, etc.
- Use `GeneralPipeline` (PyTorch) or `GeneralPipelineSklearn` (sklearn) for unified training, CV, and evaluation
- All pipelines support augmentations, metrics, k-fold CV, and model saving/loading

### 4. Benchmarking & Evaluation
- Use `BenchmarkRunner` (`benchmark.py`) for grid search over models/augmentations/metrics
- Results saved as CSV in `results/` and visualized in notebook
- Full workflow in `benchmark_results.ipynb` (see for usage examples)

### 5. Visualization & Analysis
- Plot training curves, confusion matrices, and metric comparisons (`utils/visualization.py`)
- Analyze results in notebook or programmatically

---

## Notebook Workflow Details (`benchmark_results.ipynb`)

The notebook provides a full, reproducible ML workflow for all major food prediction tasks. Key steps and experiments include:

### Tasks Covered
- **Recipe Difficulty Classification**: Predicts difficulty level (e.g., Easy, More effort, A challenge)
- **Meal Type Classification**: Predicts meal category (breakfast, lunch, dinner, snack, dessert)
- **Nutrient Regression**: Predicts nutritional values (e.g., calories, protein, fat, carbs)
- **Total Time Prediction**: Both regression (minutes) and classification (time bins)

### Models Trained
- **MLP Classifier/Regressor**: Deep and shallow variants, with/without batchnorm and dropout
- **XGBoost**: For both classification and regression
- **LightGBM**: For both classification and regression
- **RandomForest**: For both classification and regression
- **(Optionally) HuggingFace LoRA/QLoRA**: For text-based tasks (if enabled)

### Augmentation Benchmarks
- **Tabular**: SMOTE (regular, borderline, kmeans, SVM), Mixup, hybrid samplers, cleaning
- **Text**: LLM-based (Gemini, OpenAI, HuggingFace) and classical (synonym, EDA, backtranslation)
- **Image**: Classical and Albumentations pipelines (if image tasks present)

### Workflow Steps
1. **Data loading and preprocessing**: Import CSVs, clean, encode, and split data
2. **Augmentation benchmarking**: Systematically apply and compare all augmentation methods for each task
3. **Model training and evaluation**: For each task, train all models (with/without augmentation), using k-fold CV and grid search
4. **Hyperparameter tuning**: Use Optuna for selected models/tasks to optimize key hyperparameters
5. **Result aggregation**: Save all metrics and training histories to CSV in `results/`, aggregate for comparison
6. **Visualization**: Plot training curves, confusion matrices, and metric comparisons for all experiments

### Outputs
- All experiment results are saved as CSVs in `results/` (per-task, per-model, per-augmentation)
- Visualizations and summary tables are generated in the notebook for easy comparison
- Best models and configurations are saved for downstream use (e.g., in `nut_agent`)

See the notebook for code examples, experiment details, and result interpretation.

---

## Usage

### 1. Run the Benchmark Notebook

Open and execute `benchmark_results.ipynb` for a full ML workflow:

1. Data loading and preprocessing
2. Augmentation benchmarking (tabular, text, image)
3. Model training and evaluation (with/without augmentation)
4. Hyperparameter tuning (Optuna)
5. Result aggregation and visualization

### 2. Programmatic Usage

```python
from pipelines_torch.models import MODEL_REGISTRY
from pipelines_torch.base import GeneralPipeline
from data_augmentation.augmentations import AUGMENTATION_REGISTRY

# Select model and augmentation
model = MODEL_REGISTRY['mlp_classifier'](input_dim=..., num_classes=...)
augmentation = AUGMENTATION_REGISTRY['borderline_smote']

# Build pipeline
pipeline = GeneralPipeline(
  model=model,
  loss_fn='CrossEntropyLoss',
  optimizer_cls=...,  # e.g., torch.optim.Adam
  optimizer_params={'lr': 1e-3},
  augmentations=augmentation,
  metrics=[...],
  task_type='classification',
  ...
)

# Train and evaluate
pipeline.fit(X_train, y_train)
results = pipeline.evaluate(X_test, y_test)
```

### 3. Adding New Models or Augmentations

- **Augmentation**: Add function/class to `data_augmentation/`, register in `AUGMENTATION_REGISTRY`
- **Model**: Add wrapper/class to `pipelines_torch/models.py`, register in `MODEL_REGISTRY`
- **Pipeline**: Extend `GeneralPipeline` or `GeneralPipelineSklearn` as needed

---

## Configuration & Extensibility

### Augmentation
- Configure via registry name or pass custom callable
- All augmentations support batch/single APIs and random state for reproducibility
- See `AUGMENTATION_README.md` for full documentation and best practices

### Models
- Registry-based: add new models by registering in `MODEL_REGISTRY`
- Supports PyTorch, sklearn, XGBoost, LightGBM, HuggingFace, Llama.cpp
- Custom models can be added with minimal boilerplate

### Pipelines
- Unified interface for PyTorch and sklearn models
- Built-in k-fold CV, early stopping, weight averaging, and metrics tracking
- Save/load models locally or to HuggingFace Hub

### Results & Visualization
- All results saved as CSV in `results/`
- Visualization utilities for training curves, confusion matrices, and metric plots

---

## Best Practices & Tips

- Use stratified splits and set random seeds for reproducibility
- Benchmark multiple augmentations and models for robust results
- Use Optuna or grid search for hyperparameter tuning
- Leverage the registry pattern for easy extensibility
- See `AUGMENTATION_README.md` and notebook for advanced usage examples

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add new models, augmentations, or utilities
4. Add tests and update documentation
5. Submit a pull request

---

## License

This directory is part of the larger food prediction system. See the main project license for details.

### 5. Data Files
- **`recipes_df.csv`**: Main dataset containing recipe information.
- **`recipes_df_test.csv`** and **`recipes_df_test_bis.csv`**: Test datasets for evaluating model performance.

## Usage

### Running Benchmarks
To run benchmarks for different models and configurations, use the `BenchmarkRunner` class in `pipelines_torch/benchmark.py`. Example usage:

```python
from pipelines_torch.benchmark import BenchmarkRunner

runner = BenchmarkRunner(
    model_configs=[...],
    augmentations=[...],
    metrics=[...],
    task_type="classification",
    device="cpu",
    epochs=10,
    batch_size=32
)
runner.run()
```

### Data Augmentation
Use the scripts in `data_augmentation/` to augment your datasets. For example:

```python
from data_augmentation.text_aug import augment_text
augmented_text = augment_text("Sample recipe description")
```

### Utilities
- **Data Loading**: Use `load_csv` from `utils/data.py` to load datasets.
- **Metrics**: Use functions in `utils/metrics.py` to calculate evaluation metrics.

## Future Enhancements
- Add more augmentation techniques for diverse data types.
- Expand benchmarking to include additional model architectures.
- Improve documentation for individual scripts and modules.