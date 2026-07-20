# ML Pipeline Research Workspace

This folder is the working area for benchmarking ML pipelines across vision, tabular, and text tasks. The goal is to compare model families and augmentation strategies with a shared registry and consistent evaluation.

---

## Repository Overview

### Core Modules
- **`pipelines_torch/`**: The core training/evaluation framework: shared pipeline classes, model registries for vision/tabular/text (including the TabICLv2 and Google TabFM tabular foundation models), and a benchmark runner to compare models and log metrics. See `pipelines_torch/README.md`. TabFM is installed from a local clone of https://github.com/google-research/tabfm (`pip install -e .`, PyTorch backend).
- **`data_augmentation/`**: Augmentation registries and implementations for text, image, and tabular data (classical methods plus SMOTE-style samplers and LLM-based text transforms). See `data_augmentation/README.md`.
- **`utils/`**: Shared helpers used by notebooks and pipelines:
  - **`utils/data.py`**: CSV loading, embedding parsing, splits, label encoding, class balance stats, meal type filtering.
  - **`utils/metrics.py`**: Metric registry for classification/regression, plus ROC-AUC and PR-AUC wrappers.
  - **`utils/visualization.py`**: Confusion matrices, regression plots, metric histories, bar charts, per-class reports.
  - **`utils/kaggle_utils.py`**: Kaggle/local dataset resolution and downloads.
- **`utils/utils.py`**: Results directory handling and save/load for PyTorch, sklearn, and HF/PEFT models.

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
- **`bench-imai.ipynb`**: Trains and benchmarks image classifiers that discriminate AI-generated images from real ones on CIFAKE across the vision registry backbones, evaluates on the CIFAKE test split, and reuses the saved weights to check cross-dataset generalization on the "Shoes: AI vs Real" Kaggle dataset.
- **`bench-imai-artifact.ipynb`**: Same detection task on a local subset of the multi-generator ArtiFact dataset (200x200 inputs, five selected backbones), with a leak-free 20% hold-out carved out before training and cross-dataset checks on CIFAKE and the shoes dataset.
- **`bench-aitextdetect.ipynb`**: AI-generated text detection on MAGE: TF-IDF baselines (logreg, RF, XGBoost, LightGBM) plus QLoRA fine-tunes (ModernBERT base/large, Qwen3.5-4B, Gemma-4-E4B), with OOD and cross-dataset evaluation on the AI Text Detection Pile.
- **`bench_ssl.ipynb`**: Semi-supervised learning benchmark: CIFAR-10 (2 000 labels) with MeanTeacher / PseudoLabel / PiModel and Forest Covertype (500 labels) with MeanTeacher / PseudoLabel, against supervised baselines, scored on held-out test splits.
- **`bench-tabular-finance.ipynb`**: Challenging tabular use case from algorithmic trading: weekly cross-sectional S&P 500 stock selection (will a stock beat the weekly cross-sectional median over the next 5 trading days?) from technical market indicators plus sector metadata, with skrub `TableVectorizer`/`DatetimeEncoder` preprocessing, walk-forward splits with purge gaps, classical registry models vs **TabICLv2** and Google **TabFM**, and a realized long-short decile spread as the economic metric. Also includes a skrub-vs-naive-encoding ablation and a cross-sectional (per-week) feature normalization experiment.
  - Validated test ROC-AUC (2023-2024 test period): logreg 0.515, **TabICLv2 0.514** (statistically tied for best, zero-shot), TabFM 0.509, mlp/random_forest ~0.508, catboost 0.506, xgboost 0.503, lightgbm 0.497. This near-efficient-market regime is deliberately outside TabICLv2/TabFM's TabArena/TALENT evaluation comfort zone; matching the best fitted classical model without any gradient update on this data is the expected, reasonable outcome (see the notebook's "Reading the results" section for the full discussion). The skrub ablation changes ROC-AUC by at most 0.004; the cross-sectional normalization idea is a documented negative result (helps 2/8 models).
- **`bench-tabular-climate.ipynb`**: Climate change forecasting use case: 12-month-ahead global temperature anomaly regression from three joined public records (NASA GISTEMP global/NH/SH, Mauna Loa CO2, NOAA ONI/ENSO), skrub preprocessing, chronological splits, and the same classical vs foundation-model line-up in regression form. The 2015+ test period sits above the training target range, making it an extrapolation stress test. Also includes a skrub-vs-naive-date-encoding ablation and a persistence-residual reframing experiment.
  - Validated test R2: mlp 0.071 (only model beating persistence), persistence -0.165, ridge -0.168 (skrub) / **+0.072** (naive ordinal date encoding — skrub's periodic date features measurably hurt ridge here), tabfm -5.01, catboost -5.76, random_forest -6.72, lightgbm -6.36, xgboost -7.92, tabicl_v2 -7.77. Negative R2 here means "worse than the low-variance test-period mean," not "no signal" (see the notebook's worked example). Reframing the target as a persistence residual is a large, genuine improvement for the bounded-output models: **TabICLv2 goes from R2 -7.77 to +0.15** and **TabFM from -5.01 to +0.09**, both then beating the original persistence and mlp baselines.

Recipe modelling moved out of notebooks: `bench_recipe_methods.py` benchmarks embedding backends and predictors for the Recipe Lab tasks and `train_recipe_models.py` trains the deployed models (see `nut_agent/README.md`).

All notebooks above have been executed end-to-end via papermill with real (non-smoke) runs; their outputs are saved in-place. To reproduce a run:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/papermill ml_pipeline/<name>.ipynb ml_pipeline/<name>.executed.ipynb --cwd ml_pipeline -k python3
```

### Data Artifacts
- **`recipes_df.csv`**, **`recipes_df_test.csv`**, **`recipes_df_test_bis.csv`**: Recipe datasets with embeddings, text, nutrition, and time fields used by the recipe tasks.
- **`data/`** (gitignored): notebook datasets resolved by `utils/kaggle_utils.ensure_kaggle_dataset` (CIFAKE, ArtiFact subset, shoes, S&P 500 tables) plus `data/climate/` CSVs downloaded from NASA/NOAA. Public Kaggle datasets download anonymously via the kaggle CLI with a kagglehub cache fallback; no credentials required.

### Testing
- **`tests/`**: 71 unit tests covering scoring (`ComputeScore`), sklearn wrappers, PyTorch and sklearn pipelines (single-split and k-fold), checkpoint save/load round-trips, and `BenchmarkRunner` epoch normalization + smoke runs.

```bash
cd ml_pipeline && python -m pytest tests/ -v
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

Use `utils.utils.load_model` with a `checkpoint_id` from `index.jsonl`.
For sklearn joblib artifacts, `load_model` returns the direct estimator when the checkpoint was saved from a raw sklearn model, or restores the inner `.model` when loading into a wrapper class.
