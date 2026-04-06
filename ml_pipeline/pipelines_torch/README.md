# Model Pipelines & Registry (`pipelines_torch`)

This module provides a unified, extensible framework for training, evaluating, and benchmarking machine learning models across vision and tabular domains. It now includes lightweight integrations with several official 2024–2025 research repositories, semi-supervised training utilities, and advanced tabular augmentation strategies.

---

## Features

### 🏗️ Unified Model Pipelines
- **GeneralPipeline**: PyTorch-based pipeline for classification/regression, k-fold CV, augmentation, metrics, and model saving/loading.
- **GeneralPipelineSklearn**: Sklearn-compatible pipeline for classical models, with augmentation and CV support.
- **BenchmarkRunner**: Automated grid search over models, augmentations, and metrics; result aggregation and CSV export.

### 🧩 Model Registry
- **Plug-and-play models**: Register and instantiate models by name (`MODEL_REGISTRY`).
- **Vision models**: Lightweight CNNs, transfer-learning backbones, and timm-based vision models.
- **Tabular models**: Classic ML baselines and transformer-style MLPs.

### 🧪 Semi-Supervised Learning Utilities
- **Vision SSL** (`ss_vision_models.py`): Implements Pseudo-Label, Pi-Model, Mean Teacher, and CDMAD debiasing while keeping hooks compatible with official repositories.
- **Tabular SSL** (`ss_models.py`): Wraps any registry model with the same reusable SSL algorithms (pseudo-labeling or Mean Teacher) plus optional tabular-specific augmentations.


### 🔄 Cross-Validation & Ensembling
- **K-fold CV**: Built-in support for k-fold cross-validation with optional weight averaging.
- **Metrics tracking**: Track and aggregate metrics (F1, R², etc.) across folds.
- **Early stopping**: Optional early stopping based on validation loss or a selected metric.

### 💾 Model Persistence
- **Save/load**: Save model weights and training history locally or to HuggingFace Hub.
- **Reproducibility**: Deterministic splits, seed control, and experiment tracking.

---

## Directory Structure

```
pipelines_torch/
├── __init__.py
├── base.py             # GeneralPipeline, GeneralPipelineSklearn
├── benchmark.py        # BenchmarkRunner (grid search, result aggregation)
├── models.py           # Model registry, wrappers, and third-party adapters for tabular/text
├── vision_models.py    # Vision model registry (CNNs, CLIP, FatFormer, DiffusionFake, etc.)
├── ss_models.py        # Tabular semi-supervised wrappers
├── ss_vision_models.py # Vision semi-supervised algorithms & hooks
├── ssl_algorithms.py   # Shared implementations of pseudo-label, Pi-Model, and Mean Teacher
└── ...
```

---

## Model Registry Overview

Models are registered under descriptive keys. The following tables highlight the most commonly used entries (see source files for the exhaustive list).

### Classification Models (`models.py`)
- `mlp_classifier`, `deep_mlp_classifier`: PyTorch MLPs (configurable depth, batchnorm, dropout)
- `random_forest_classifier`, `xgboost_classifier`, `lightgbm_classifier`: Classical baselines
- `tabicl_classifier`: TabICL tabular foundation model (sklearn-compatible, `pip install tabicl`)
- `hf_lora_classifier`, `hf_qlora_classifier`, `llama_cpp_classifier`: Text-oriented adapters

### Regression Models (`models.py`)
- `mlp_regressor`, `deep_mlp_regressor`: PyTorch MLPs
- `random_forest_regressor`, `xgboost_regressor`, `lightgbm_regressor`
- `tabicl_regressor`: TabICL tabular foundation model for regression
- `hf_lora_regressor`, `hf_qlora_regressor`, `llama_cpp_regressor`

### Vision Models (`vision_models.py`)
- `simple_cnn`, `adaptive_cnn`, `residual_cnn`: Lightweight CNN baselines
- `resnet50`: Transfer learning backbone
- `clip_classifier`, `dinov3_classifier`, `qwen2_vl_qlora`: Multimodal and large-model adapters
- `timm_dinov2_vit_small`, `timm_dinov2_vit_base`, `timm_dinov2_vit_large`: DINOv2 self-supervised ViT backbones (via timm)
- `timm_convnextv2_tiny`, `timm_efficientnetv2_s`, `timm_vit_base_patch16`, `timm_vit_mae_base`: Other timm backbones

Use `get_model("model_name", ...)` to instantiate any vision model; tabular/text models are accessed via `MODEL_REGISTRY`.

---

## Semi-Supervised Modules

### Vision (`ss_vision_models.py`)
Provides loss wrappers that can sit on top of any vision backbone:
- `PseudoLabel`, `PiModel`, `MeanTeacher`: Classical SSL baselines implemented once in `ssl_algorithms.py` and reused across vision and tabular pipelines.
- Hooks return `(supervised_loss, unsupervised_loss, logs)` so you can integrate them into existing training loops.

### Tabular (`ss_models.py`)
- `SemiSupervisedTabular`: Wraps any registry model, reusing the shared pseudo-label or Mean Teacher implementations with optional Gaussian-noise augmentation hooks.
- Designed to operate with the augmentation utilities in `augmentations.py` for minority-class oversampling or consistency regularization.

---

## Usage Examples

### 1. Select and Instantiate a Model

```python
# Tabular/text models
from pipelines_torch.models import MODEL_REGISTRY
model = MODEL_REGISTRY['tabr_classifier'](input_dim=..., num_classes=..., init_kwargs={'k': 32})

# Vision models
from pipelines_torch.vision_models import get_model
vision_model = get_model('resnet50', num_classes=2)
```

### 2. Build a Pipeline

```python
from pipelines_torch.base import GeneralPipeline
pipeline = GeneralPipeline(
    model=model,
    loss_fn='CrossEntropyLoss',
    optimizer_cls=torch.optim.Adam,
    optimizer_params={'lr': 1e-3},
    augmentations=None,
    metrics=[...],
    task_type='classification',
)
```

### 3. Train with Semi-Supervised Helpers

```python
from pipelines_torch.ss_models import SemiSupervisedTabular
ssl_wrapper = SemiSupervisedTabular(model, num_classes=2, use_mean_teacher=True)
loss, logs = ssl_wrapper.step((xb_l, yb_l), (xb_u, yb_u), epoch)
```

### 4. Benchmark Multiple Models/Augmentations

```python
from pipelines_torch.benchmark import BenchmarkRunner
runner = BenchmarkRunner(
    model_configs=[...],
    augmentations=[MGS_GRF_Augmentor()],
    metrics=[...],
)
results_df = runner.run(X, y)
```

---

## Extending the Registry

1. Implement your model or wrapper in `models.py` or `vision_models.py`.
2. Register it in `MODEL_REGISTRY` with a unique key.
3. Use it in pipelines, SSL wrappers, or benchmarks by name.

---

## Best Practices

- Use k-fold CV for robust evaluation and track metrics consistently.
- Register all new models and augmentations to keep experiments reproducible.
- Integrate with semi-supervised utilities when working with partially labeled datasets.
- Save models, logs, and augmentation parameters for later audits.
