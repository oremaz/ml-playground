# Model Pipelines & Registry (`pipelines_torch`)

This module provides a unified, extensible framework for training, evaluating, and benchmarking machine learning models across vision and tabular domains. It now includes lightweight integrations with several official 2024–2025 research repositories, semi-supervised training utilities, and advanced tabular augmentation strategies.

---

## Features

### 🏗️ Unified Model Pipelines
- **GeneralPipeline**: PyTorch-based pipeline for classification/regression, k-fold CV, augmentation, metrics, and model saving/loading.
- **GeneralPipelineSklearn**: Sklearn-compatible pipeline for classical models, with augmentation and CV support.
- **BenchmarkRunner**: Automated grid search over models, augmentations, and metrics; result aggregation and CSV export.

### 🧩 Model Registry & Third-Party Hooks
- **Plug-and-play models**: Register and instantiate models by name (`MODEL_REGISTRY`).
- **Vision models**: Lightweight CNNs, transfer-learning backbones, and official research detectors (FatFormer, DiffusionFake) via wrappers around their public GitHub repos.
- **Tabular models**: Classic ML baselines, transformer-style MLPs, and official ICLR models (TabR, GRANDE, TabM) through thin adapters.
- **Third-party loader**: `third_party/__init__.py` provides `load_class` helpers so you can keep upstream repositories as git submodules or point to local clones using environment variables (e.g., `FATFORMER_REPO`, `DIFFUSIONFAKE_REPO`).

### 🧪 Semi-Supervised Learning Utilities
- **Vision SSL** (`ss_vision_models.py`): Implements Pseudo-Label, Pi-Model, Mean Teacher, and CDMAD debiasing while keeping hooks compatible with official repositories.
- **Tabular SSL** (`ss_models.py`): General-purpose pseudo-labeling + EMA framework that can wrap any registry model and plug in tabular augmentations.

### 🧬 Advanced Augmentations (`augmentations.py`)
- **Official research integrations**: Wrappers for MGS-GRF (Artefactory, 2025) and TabEBM (NeurIPS 2024) sourced directly from their repositories.
- **Recent SMOTE variants**: Custom implementations of Simplicial SMOTE (2025) and MEB-SMOTE (2024) for handling mixed continuous/categorical data.

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
├── augmentations.py    # Tabular augmentation strategies (MGS-GRF, TabEBM, SMOTE variants)
└── ...
```

---

## Model Registry Overview

Models are registered under descriptive keys. The following tables highlight the most commonly used entries (see source files for the exhaustive list).

### Classification Models (`models.py`)
- `mlp_classifier`, `deep_mlp_classifier`: PyTorch MLPs (configurable depth, batchnorm, dropout)
- `random_forest_classifier`, `xgboost_classifier`, `lightgbm_classifier`: Classical baselines
- `tabr_classifier`: Wrapper around the official Yandex TabR implementation (ICLR 2024)
- `grande_classifier`: Wrapper for the official GRANDE differentiable tree ensemble (ICLR 2024)
- `tabm_classifier`: Wrapper around the official TabM multi-head ensemble (ICLR 2025)
- `hf_lora_classifier`, `hf_qlora_classifier`, `llama_cpp_classifier`: Text-oriented adapters

### Regression Models (`models.py`)
- `mlp_regressor`, `deep_mlp_regressor`: PyTorch MLPs
- `random_forest_regressor`, `xgboost_regressor`, `lightgbm_regressor`
- `hf_lora_regressor`, `hf_qlora_regressor`, `llama_cpp_regressor`

### Vision Models (`vision_models.py`)
- `simple_cnn`, `adaptive_cnn`, `residual_cnn`: Lightweight CNN baselines
- `resnet50`, `vision_transformer`: Transfer learning/ViT backbones
- `clip_classifier`, `qwen2_vl_qlora`: Multimodal and large-model adapters
- `fatformer_official`: Wrapper loading the CVPR 2024 FatFormer implementation via `third_party`
- `diffusionfake_official`: Wrapper loading the NeurIPS 2024 DiffusionFake detector via `third_party`

Use `get_model("model_name", ...)` to instantiate any vision model; tabular/text models are accessed via `MODEL_REGISTRY`.

---

## Semi-Supervised Modules

### Vision (`ss_vision_models.py`)
Provides loss wrappers that can sit on top of any vision backbone:
- `PseudoLabel`, `PiModel`, `MeanTeacher`: Classical SSL baselines implemented from scratch in PyTorch.
- `CDMADHook`: Debias pseudo-labels using the official CDMAD repository or a faithful fallback implementation.
- Hooks return `(supervised_loss, unsupervised_loss, logs)` so you can integrate them into existing training loops.

### Tabular (`ss_models.py`)
- `SemiSupervisedTabular`: Wraps any registry model, performs pseudo-labeling with optional EMA teacher updates, and exposes augmentation hooks.
- Designed to operate with the augmentation utilities in `augmentations.py` for minority-class oversampling or consistency regularization.

---

## Tabular Augmentations (`augmentations.py`)
- `MGS_GRF_Augmentor`: Calls the official mixed-type oversampling implementation (requires cloning `artefactory/mgs-grf` or setting `MGS_GRF_REPO`).
- `TabEBMGenerator`: Interfaces with the official TabEBM energy-based sampler (set `TABEBM_REPO` if the repo lives elsewhere).
- `SimplicialSMOTE`, `MEBSMOTE`: Recent SMOTE variants implemented in-house for datasets where official code is unavailable.

Each class operates on NumPy arrays and can be injected into pipelines or SSL routines as needed.

---

## Working with Third-Party Repositories

The project expects official research code to live under `third_party/<RepoName>` (typically as git submodules). Utilities in `third_party/__init__.py` make it easy to load classes/functions without modifying the upstream code.

```python
from third_party import load_class
FatFormer = load_class("FatFormer", "FatFormerModel", env_var="FATFORMER_REPO")
```

You can override locations using environment variables (e.g., `export FATFORMER_REPO=/path/to/FatFormer`). If a repository is missing, the loader raises a helpful error listing searched paths.

Run `git submodule update --init --recursive` (or clone the referenced repos manually) before instantiating any of the official-model adapters.

---

## Usage Examples

### 1. Select and Instantiate a Model

```python
# Tabular/text models
from pipelines_torch.models import MODEL_REGISTRY
model = MODEL_REGISTRY['tabr_classifier'](input_dim=..., num_classes=..., init_kwargs={'k': 32})

# Vision models (including official research detectors)
from pipelines_torch.vision_models import get_model
vision_model = get_model('fatformer_official', num_classes=2)
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
3. (Optional) If pulling from an external repo, add a loader that leverages `third_party.load_class`.
4. Use it in pipelines, SSL wrappers, or benchmarks by name.

---

## Best Practices

- Keep third-party repositories up to date via `git submodule update --init --recursive`.
- Use k-fold CV for robust evaluation and track metrics consistently.
- Register all new models and augmentations to keep experiments reproducible.
- Integrate with semi-supervised utilities when working with partially labeled datasets.
- Save models, logs, and augmentation parameters for later audits.
