# ML Playground: Reproducible Experiments and Benchmarking

This repository is a workspace for benchmarking ML pipelines across vision, tabular, and text tasks. The goal is to compare model families and augmentation strategies with a shared registry and consistent evaluation.


The reusable Python package is `ml_pipeline`. [NutriCoach](https://github.com/oremaz/nutricoach) and [AI Content Robustness](https://github.com/oremaz/ai-content-robustness) install it as a pinned dependency.

## Installation and quickstart

```bash
uv venv
uv pip install -e ".[dev]"
PYTHONPATH=. uv run python -m pytest ml_pipeline/tests -q
```

Install the optional experiment dependencies with `uv pip install -r requirements.txt`, or the CUDA 12.4 environment with `uv pip install -r requirements_gpu.in`. On macOS, XGBoost and LightGBM require an OpenMP runtime such as `libomp`.

```python
from ml_pipeline.pipelines_torch.benchmark import BenchmarkRunner
from ml_pipeline.utils.utils import load_model_by_name
```

Launch the benchmark UI from the repository root after installing research dependencies:

```bash
PYTHONPATH=. uv run streamlit run ml_pipeline/benchmark_app.py
```

Historical `pipelines_torch`, `utils`, and `data_augmentation` imports remain available as compatibility modules for saved checkpoints and older notebooks. New code should use `ml_pipeline` imports.

The framework writes checkpoints to `results/` relative to the working directory. Set `ML_PIPELINE_RESULTS_DIR` to override both checkpoint reading and writing; installed code never assumes a sibling checkout. On Kaggle, the existing input/output defaults remain unchanged.

---

## Repository Overview

### Core Modules
- **`pipelines_torch/`**: The core training/evaluation framework: shared pipeline classes, model registries for vision/tabular/text (including the TabICLv2 and Google TabFM tabular foundation models), and a benchmark runner to compare models and log metrics. See [pipeline documentation](ml_pipeline/pipelines_torch/README.md). The repository requirements install the pinned TabFM release and its PyTorch backend directly from the official repository.
- **`data_augmentation/`**: Augmentation registries and implementations for text, image, and tabular data (classical methods plus SMOTE-style samplers and LLM-based text transforms). See [augmentation documentation](ml_pipeline/data_augmentation/README.md).
- **`utils/`**: Shared helpers used by notebooks and pipelines:
  - **`utils/data.py`**: CSV loading, embedding parsing, splits, label encoding, class balance stats, meal type filtering.
  - **`utils/metrics.py`**: Metric registry for classification/regression, plus ROC-AUC and PR-AUC wrappers.
  - **`utils/visualization.py`**: Confusion matrices, regression plots, metric histories, bar charts, per-class reports.
  - **`utils/kaggle_utils.py`**: Kaggle/local dataset resolution and downloads.
- **`utils/ml_pipeline.utils.py`**: Results directory handling and save/load for PyTorch, sklearn, and HF/PEFT models.

### Logging

All pipeline, model wrapper, and benchmark diagnostics use Python's `logging` module. To see debug output (prediction distributions, metric details):

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Weights & Biases

Experiment tracking is available through `utils/wandb_utils.py` and is wired into `BenchmarkRunner`:

```python
BenchmarkRunner(..., use_wandb=True, wandb_project="bench-research-ml")
```

Each (model, augmentation) combination becomes one W&B run (grouped by `path_start`) with the per-epoch training history, CV summary statistics, and the checkpoint id. Notebooks that train outside `BenchmarkRunner` call `init_wandb_run` / `log_summary` / `finish_run` directly. Logging degrades gracefully: with no API key (env or `~/.netrc`) runs are written in offline mode, and a missing `wandb` package only produces a warning.

### Experiments & Notebooks
- **`benchmark_app.py`**: **Streamlit UI** for interactive model benchmarking. Upload CSV datasets, select models/augmentations from registries, configure training parameters (epochs, batch size, learning rate), and visualize results in real-time. Supports classification and regression tasks with automatic metric computation.
- **`bench-tabular-stroke.ipynb`**: Rare early recurrent-stroke ranking from the fully anonymous, openly licensed International Stroke Trial table (19,435 rows, about 4% positive). The concise progression is dataset and objective, a 1991-1994/1995/1996 temporal split, first-pass preprocessing and SMOTE ablations, then classical models, a CUDA MLP, TabICLv2, TabFM, and the official TabPFN classifier. Selection prioritizes recall, precision, and lift inside a fixed 10% monitoring budget, with PR-AUC and Brier score as supporting criteria. A validation-selected percentile-rank ensemble is the single improvement experiment. The refactored notebook requires a new GPU run before updated results are published.

Detection training and evaluation notebooks now live in [AI Content Robustness](https://github.com/oremaz/ai-content-robustness/tree/main/ai_content_detector/notebooks). Recipe training and benchmarking live in [NutriCoach](https://github.com/oremaz/nutricoach). Generic model wrappers and checkpoint utilities remain here.

The stroke notebook is `ml_pipeline/bench-tabular-stroke.ipynb`. Its refactored experiments require a new GPU run before updated results are published.

### Data Artifacts
- **`ml_pipeline/data/`** (gitignored): local experiment datasets. Generic Kaggle download utilities remain in `ml_pipeline.utils.kaggle_utils`; detection datasets are managed by AI Content Robustness.
  - `ml_pipeline/data/tabular_stroke/IST_corrected.csv` is downloaded directly from the University of Edinburgh DataShare record and verified against its SHA-256 checksum by the stroke notebook.

### Testing
- **`tests/`**: unit tests covering scoring (`ComputeScore`), sklearn wrappers, PyTorch and sklearn pipelines (single-split and k-fold), checkpoint save/load round-trips, and `BenchmarkRunner` epoch normalization + smoke runs.

```bash
PYTHONPATH=. uv run python -m pytest ml_pipeline/tests -q
```

---

## Checkpoints & Results

Each benchmark run writes artifacts under `results/<run>/` and records a lightweight index:

- `index.jsonl`: one JSON line per checkpoint, containing:
  - `checkpoint_id` (deterministic SHA-1 hash)
  - `model_name`, `augmentation_name`, `task_type`
  - key hyperparams (epochs, lr, weight_decay, dropout, batch_size, kfold config)
  - `artifact_type` and `artifact_path`
- `*_metrics.csv`: training history keyed by `checkpoint_id`

Checkpoint artifacts are saved as:
- `results/<run>/<checkpoint_id>.pt` for PyTorch state dicts
- `results/<run>/<checkpoint_id>.pkl` for sklearn joblib, including both `SklearnModelWrapper` instances and direct estimators such as `LogisticRegression`
- `results/<run>/<checkpoint_id>/` for HuggingFace `save_pretrained`

### Loading a checkpoint

Use `ml_pipeline.utils.utils.load_model` with a `checkpoint_id` from `index.jsonl`.
For sklearn joblib artifacts, `load_model` returns the direct estimator when the checkpoint was saved from a raw sklearn model, or restores the inner `.model` when loading into a wrapper class.

For CPU validation on shared VMs, set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2` before running the tests.

TabFM wrapper tests require the optional `tabular` dependencies and otherwise skip. These tests mock model loading and do not download foundation-model weights.

## Split validation (2026-10-14)

The complete CPU suite passed: **78 passed**. This includes historical checkpoint loading through compatibility imports, checkpoint round trips, scoring, model wrappers, training pipelines, and benchmarking. TabFM loading is mocked; no foundation-model weights are downloaded by those tests.

A wheel was built and installed into a separate validation interpreter. The declared NumPy, SciPy, scikit-learn, pandas, and LightGBM versions were installed for this run. GPU experiments and the refactored stroke notebook were not executed during the migration.
