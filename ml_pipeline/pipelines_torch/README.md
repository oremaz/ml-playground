# Model Pipelines & Registry (`pipelines_torch`)

This module provides a unified, extensible framework for training, evaluating, and benchmarking machine learning models for food-related prediction tasks. It supports both PyTorch and scikit-learn models, advanced cross-validation, model registries, and seamless integration with data augmentation modules.

---

## Features

### 🏗️ Unified Model Pipelines
- **GeneralPipeline**: PyTorch-based pipeline for classification/regression, k-fold CV, augmentation, metrics, and model saving/loading.
- **GeneralPipelineSklearn**: Sklearn-compatible pipeline for classical models, with augmentation and CV support.
- **BenchmarkRunner**: Automated grid search over models, augmentations, and metrics; result aggregation and CSV export.

### 🧩 Model Registry
- **Plug-and-play models**: Register and instantiate models by name (MLP, XGBoost, LightGBM, RandomForest, HuggingFace LoRA/QLoRA, Llama.cpp, etc.).
- **Vision models**: Comprehensive computer vision models from lightweight CNNs to foundation models (ResNet, ViT, CLIP, Qwen2-VL)
- **Custom wrappers**: All models are wrapped for compatibility with the pipeline interface.
- **Easy extensibility**: Add new models by registering in `MODEL_REGISTRY`.

### 🔄 Cross-Validation & Ensembling
- **K-fold CV**: Built-in support for k-fold cross-validation, with weight averaging for PyTorch models.
- **Metrics tracking**: Track and aggregate metrics (F1, R², etc.) across folds.
- **Early stopping**: Optional early stopping based on validation loss or metric.

### 💾 Model Persistence
- **Save/load**: Save model weights and training history locally or to HuggingFace Hub.
- **Reproducibility**: Deterministic splits, seed control, and experiment tracking.

---

## Directory Structure

```
pipelines_torch/
├── __init__.py
├── base.py           # GeneralPipeline, GeneralPipelineSklearn
├── benchmark.py      # BenchmarkRunner (grid search, result aggregation)
├── models.py         # Model registry and wrappers (tabular/text models)
├── vision_models.py  # Computer vision model registry (CNNs, ViTs, CLIP, multimodal)
```

---

## Model Registry

The model registry provides easy access to a comprehensive collection of models spanning different architectures and modalities. Models are organized by task type and complexity level.

### Classification Models
- `mlp_classifier`, `deep_mlp_classifier`: PyTorch MLPs (configurable depth, batchnorm, dropout)
- `random_forest_classifier`: Sklearn RandomForest
- `xgboost_classifier`: XGBoost
- `lightgbm_classifier`: LightGBM
- `hf_lora_classifier`, `hf_qlora_classifier`: HuggingFace LoRA/QLoRA (text)
- `llama_cpp_classifier`: Llama.cpp adapter (text)

### Regression Models
- `mlp_regressor`, `deep_mlp_regressor`: PyTorch MLPs
- `random_forest_regressor`: Sklearn RandomForest
- `xgboost_regressor`: XGBoost
- `lightgbm_regressor`: LightGBM
- `hf_lora_regressor`, `hf_qlora_regressor`: HuggingFace LoRA/QLoRA (text)
- `llama_cpp_regressor`: Llama.cpp adapter (text)

### Vision Models (`vision_models.py`)
Specialized computer vision models for image classification and analysis tasks:

#### Lightweight CNNs
- `simple_cnn`: Fast, minimal CNN for quick experiments (3→16→32 conv layers, ~64 final dense)
- `adaptive_cnn`: Modern CNN with GELU activations, dropout regularization, and adaptive pooling for any input size (32→64→128 conv layers)

#### Advanced Architectures
- `residual_cnn`: Small ResNet-inspired network with residual connections and batch normalization
- `resnet50`: Transfer learning with ResNet-50 backbone (ImageNet pretrained by default, auto-resizes input to 224x224)
- `vision_transformer`: ViT-B/16 Vision Transformer (ImageNet pretrained by default, auto-resizes input to 224x224)

#### Multimodal & Foundation Models
- `clip_classifier`: Fine-tuned CLIP vision encoder with frozen features + trainable linear head (auto-resizes input to 224x224)
- `qwen2_vl_qlora`: QLoRA fine-tuning for Qwen 2.5 Vision-Language model (4-bit quantization, LoRA adapters)

#### Usage Example
```python
from pipelines_torch.vision_models import get_model

# For small images (e.g., 32x32 CIFAR-like data)
model = get_model('adaptive_cnn', num_classes=3)  # Handles any input size

# For ImageNet-style transfer learning (auto-resizes to 224x224)
model = get_model('resnet50', num_classes=3, pretrained=True)

# For Vision Transformer (auto-resizes to 224x224)
model = get_model('vision_transformer', num_classes=3, pretrained=True)

# All models automatically handle input size mismatches through interpolation
```

---

## Usage Examples

### 1. Select and Instantiate a Model

```python
# Tabular/text models
from pipelines_torch.models import MODEL_REGISTRY
model = MODEL_REGISTRY['mlp_classifier'](input_dim=..., num_classes=...)

# Vision models
from pipelines_torch.vision_models import get_model
vision_model = get_model('resnet50', num_classes=3, pretrained=True)
```

### 2. Build a Pipeline

```python
from pipelines_torch.base import GeneralPipeline
pipeline = GeneralPipeline(
    model=model,
    loss_fn='CrossEntropyLoss',
    optimizer_cls=...,  # e.g., torch.optim.Adam
    optimizer_params={'lr': 1e-3},
    augmentations=...,  # e.g., from AUGMENTATION_REGISTRY
    metrics=[...],
    task_type='classification',
    ...
)
```

### 3. Train and Evaluate

```python
pipeline.fit(X_train, y_train)
results = pipeline.evaluate(X_test, y_test)
```

### 4. Benchmark Multiple Models/Augmentations

```python
from pipelines_torch.benchmark import BenchmarkRunner
runner = BenchmarkRunner(
    model_configs=[...],
    augmentations=[...],
    metrics=[...],
    ...
)
results_df = runner.run(X, y)
```

---

## Extending the Registry

1. Implement your model or wrapper in `models.py`.
2. Register it in `MODEL_REGISTRY` with a unique key.
3. Use it in pipelines or benchmarks by name.

---

## Best Practices

- Use k-fold CV for robust evaluation.
- Register all new models for easy experimentation.
- Save models and metrics for reproducibility.
- Integrate with augmentation modules for best results.