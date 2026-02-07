# ML Pipeline Research Workspace

This folder is the working area for benchmarking ML pipelines across vision, tabular, and text tasks. The goal is to compare model families and augmentation strategies with a shared registry and consistent evaluation.

---

## Repository Overview

### Core Modules
- **`pipelines_torch/`**: The core training/evaluation framework: shared pipeline classes, model registries for vision/tabular/text, and a benchmark runner to compare models and log metrics. See `pipelines_torch/README.md`.
- **`data_augmentation/`**: Augmentation registries and implementations for text, image, and tabular data (classical methods plus SMOTE-style samplers and LLM-based text transforms). See `data_augmentation/README.md`.
- **`utils/`**: Shared helpers used by notebooks and pipelines:
  - **`utils/data.py`**: CSV loading, embedding parsing, splits, label encoding, class balance stats, meal type filtering.
  - **`utils/metrics.py`**: Metric registry for classification/regression, plus ROC-AUC and PR-AUC wrappers.
  - **`utils/visualization.py`**: Confusion matrices, regression plots, metric histories, bar charts, per-class reports.
  - **`utils/kaggle_utils.py`**: Kaggle/local dataset resolution and downloads.
- **`utils/utils.py`**: Results directory handling and save/load for PyTorch, sklearn, and HF/PEFT models.

### Experiments & Notebooks
- **`benchmark_app.py`**: **Streamlit UI** for interactive model benchmarking. Upload CSV datasets, select models/augmentations from registries, configure training parameters (epochs, batch size, learning rate), and visualize results in real-time. Supports classification and regression tasks with automatic metric computation.
- **`bench-imai.ipynb`**: I trained and benchmarked image classifiers that discriminate AI-generated images from real ones using the CIFAKE dataset. I compared multiple vision backbones, saved checkpoints, evaluated on the CIFAKE test split, and then reused the saved weights to check cross-dataset generalization on the "Shoes: AI vs Real" Kaggle dataset (with an optional step to download FatFormer weights).
- **`bench-food.ipynb`**: I benchmarked multiple embedding‑based models to predict recipe difficulty, meal type, nutrition, and prep time, and validated them on external test sets (`recipes_df_test.csv`, `recipes_df_test_bis.csv`).

### Data Artifacts
- **`recipes_df.csv`**, **`recipes_df_test.csv`**, **`recipes_df_test_bis.csv`**: Recipe datasets with embeddings, text, nutrition, and time fields used by the recipe notebook tasks.

### External Research Code
- **`third_party/`**: A folder that contains copies of external research implementations (e.g., official model repos). These are not authored here; the pipeline adapters point to them, and each subfolder keeps its upstream README and license.

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
- `results/<run>/<checkpoint_id>.pkl` for sklearn joblib
- `results/<run>/<checkpoint_id>/` for HuggingFace `save_pretrained`

### Loading a checkpoint

Use `utils.utils.load_model` with a `checkpoint_id` from `index.jsonl`.
