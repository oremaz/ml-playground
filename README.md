# ML Pipeline Research Workspace

This folder is the working area for benchmarking ML pipelines across vision, tabular, and text tasks. The goal is to compare model families and augmentation strategies with a shared registry and consistent evaluation.

---

## Repository Overview

### Core Modules
- **`pipelines_torch/`**: The core training/evaluation framework: shared pipeline classes, model registries for vision/tabular/text (including the TabICLv2 and Google TabFM tabular foundation models), and a benchmark runner to compare models and log metrics. See `pipelines_torch/README.md`. The repository requirements install the pinned TabFM release and its PyTorch backend directly from the official repository.
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
- **`bench-imai-artifact.ipynb`**: Trains five image detectors on a local subset of the multi-generator ArtiFact dataset (200x200 inputs), with a leak-free 20% hold-out carved out before training and cross-dataset checks on CIFAKE and the shoes dataset. The notebook retains the observed OOD failure as its primary result, then adds an unexecuted, gated ConvNeXt improvement experiment using class balancing and shortcut-resistant resolution, JPEG, blur, and colour randomisation. CIFAKE remains an external distribution-shift evaluation and is not added to training.
- **`bench-aitextdetect.ipynb`**: AI-generated text detection trained on MAGE with TF-IDF baselines and six QLoRA models: ModernBERT base/large, Qwen3.5-4B-Base, Gemma-4-E4B, Nanbeige4.2-3B, and LiquidAI LFM2.5-2.6B. Evaluation covers MAGE, the AI Text Detection Pile, and a streamed 2,000-row external OOD peer-review benchmark generated with GPT-4o, Claude Sonnet 3.5, Gemini 1.5 Pro, Qwen 2.5 72B, and Llama 3.1 70B. Gemma 4 uses its multimodal backbone in text-only mode; Nanbeige and LFM use their causal backbones with trainable classification heads. Previous Gemma 3 metrics were removed; rerun the notebook before publishing updated comparisons.
- **`bench-tabular-stroke.executed.ipynb`**: Rare early recurrent-stroke ranking from the fully anonymous, openly licensed International Stroke Trial table (19,435 rows, about 4% positive). The concise progression is dataset and objective, a 1991-1994/1995/1996 temporal split, first-pass preprocessing and SMOTE ablations, then classical models, a CUDA MLP, TabICLv2, TabFM, and the official TabPFN classifier. Selection prioritizes recall, precision, and lift inside a fixed 10% monitoring budget, with PR-AUC and Brier score as supporting criteria. A validation-selected percentile-rank ensemble is the single improvement experiment. The refactored notebook requires a new GPU run before updated results are published.

Recipe modelling moved out of notebooks: `bench_recipe_methods.py` benchmarks BGE, Jina v5, and LiquidAI LFM2.5 embedding backends with classification methods for the Recipe Lab tasks; `train_recipe_models.py` trains the deployed models (see `nut_agent/README.md`).
Pass one or more Hugging Face IDs with `--embedding-models` to override the
default embedding list. The food-image zero-shot benchmark similarly accepts
`--models`; prefix an ID with `clip=` or `jina=` when backend auto-detection is
not sufficient.

The original ArtiFact benchmark retains its validated outputs. Its improvement
block, the text-detection notebook, and the refactored stroke notebook require
new GPU runs. Stroke is intentionally kept as a single portfolio notebook rather
than duplicated into executed and unexecuted files.

### Data Artifacts
- **`recipes_df.csv`**, **`recipes_df_test.csv`**, **`recipes_df_test_bis.csv`**: Recipe datasets with embeddings, text, nutrition, and time fields used by the recipe tasks.
- **`data/`** (gitignored): notebook datasets resolved by `utils.kaggle_utils.ensure_kaggle_dataset` (CIFAKE, ArtiFact subset, and shoes). Public Kaggle datasets download anonymously via the kaggle CLI with a kagglehub cache fallback; no credentials required.
  - `data/tabular_stroke/IST_corrected.csv` is downloaded directly from the University of Edinburgh DataShare record and verified against its SHA-256 checksum by the stroke notebook.

### Testing
- **`tests/`**: 73 unit tests covering scoring (`ComputeScore`), sklearn wrappers, PyTorch and sklearn pipelines (single-split and k-fold), checkpoint save/load round-trips, and `BenchmarkRunner` epoch normalization + smoke runs.

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
