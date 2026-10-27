# Model pipelines and registries

This package connects models to fitting, evaluation, augmentation and checkpoint
storage. Start here for the library API. The [project README](../../README.md)
describes the CSV assistant, whose frozen development folds and reserved final
partition are managed by a separate experiment runtime.

## Choose a training route

| Route | Use it for | Main contract |
|---|---|---|
| [skore comparison](../../docs/skore.md) | New ordinary sklearn model comparisons | Delegates CV, metrics, inspection and cached reports to skore |
| `GeneralPipelineSklearn` | Classical estimators | Fits arrays, optionally runs CV, exposes validation metrics and fitted model |
| `GeneralPipeline` | PyTorch modules | Owns loaders, optimizer, epochs, mixed precision and training restart state |
| `HuggingFaceQLoRAWrapper.fit` | LoRA/QLoRA text classifiers and regressors | Uses Hugging Face Trainer and native Trainer checkpoints |
| `BenchmarkRunner` | Registered model/augmentation comparisons | Fits combinations and writes model/metric artifacts |
| [playground runtime](../playground_runtime.py) | Reviewed tabular experiments | Owns preprocessing and configured holdout/group/temporal splits |
| [forecasting runtime](../forecasting_runtime.py) | Univariate forecasts | Uses shared past-only contexts, origins and horizons |

The pipeline convenience CV uses shuffled stratified folds for classification
and shuffled folds for regression. It is not a group or chronological split.
Classical CV finishes with a fit on all supplied training rows; Torch CV averages
fold weights. The CSV assistant has its own selection protocol and does not use
these behaviors as its final-evaluation policy.

## A complete CPU example

This fits a classical model with three development folds, then evaluates once on
an explicit reserved test partition. No downloads, API calls or file writes are
needed. Run Python from the repository root with `PYTHONPATH=.`:

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ml_pipeline.pipelines_torch.base import GeneralPipelineSklearn

X, y = make_classification(n_samples=200, n_features=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)
pipeline = GeneralPipelineSklearn(
    LogisticRegression(max_iter=1000), metrics=[accuracy_score],
    use_kfold=True, k_folds=3, random_state=42,
)
history = pipeline.fit(X_train, y_train)
print(pipeline.evaluate(X_test, y_test))
```

Preprocessing is outside this minimal example. Fit learned transformations on
training rows within each fold, as the playground runtime does. Augmentation
also belongs only to the training portion; see the [augmentation guide](../data_augmentation/README.md).

## Registry models and saved comparisons

[models.py](models.py) exposes `CLASSIFICATION_MODEL_REGISTRY`,
`REGRESSION_MODEL_REGISTRY` and their combined `MODEL_REGISTRY`. Common names:

| Family | Classification / regression keys |
|---|---|
| Simple controls | `constant_classifier`, `linear_classifier` / `constant_regressor`, `linear_regressor` |
| MLP | `mlp_classifier`, `deep_mlp_classifier` / `mlp_regressor`, `deep_mlp_regressor` |
| Tree models | `random_forest_*`, `xgboost_*`, `lightgbm_*`, `catboost_*` with `classifier` or `regressor` suffix |
| Stacking | `stacking_classifier` / `stacking_regressor` |
| NVIDIA Kumo | `kumo_classifier` / `kumo_regressor`; `size` selects small, medium or large |
| Tabular foundation models | `tabicl_classifier`, `tabfm_classifier` / `tabicl_regressor`, `tabfm_regressor` |
| Text adapters | `hf_qlora_classifier`, `llama_cpp_classifier` / `hf_qlora_regressor`, `llama_cpp_regressor` |

For a new classical comparison, prefer the short [skrub/skore path](../../docs/skore.md).
The legacy runner below remains useful for reproducing existing artifacts and
using its native training/restart features.

Registry values are constructors; instantiate them with model-specific arguments.
Optional backends need their packages and weights. Vision models have their own
[registry and get_model helper](vision_models.py), including simple CNNs,
ResNet, CLIP, DINO and timm backbones. Forecasting models implement
`forecast(contexts, horizon)`, rather than a tabular `predict_proba` interface;
see [the adapter metadata](../forecasting_models.py) for registered names,
checkpoint revisions, quantiles and licenses.

Using the same `X_train` and `y_train` as above, this comparison writes artifacts
under `results/pipeline_example/` (or `ML_PIPELINE_RESULTS_DIR/pipeline_example/`):

```python
from ml_pipeline.pipelines_torch.benchmark import BenchmarkRunner
from ml_pipeline.pipelines_torch.models import MODEL_REGISTRY

runner = BenchmarkRunner(
    model_configs=[{
        "name": "linear_classifier",
        "class": MODEL_REGISTRY["linear_classifier"],
        "params": {},
    }],
    augmentations=[None], metrics=[accuracy_score],
    use_kfold=True, k_folds=3, path_start="pipeline_example", random_state=42,
)
runner.run(X_train, y_train)
```

`run()` writes the checkpoint index and metric files; it does not return a
DataFrame. Choose a run directory deliberately: an existing completed checkpoint
can be reused rather than fitted again. The saved registry is managed by
[utils.py](../utils/utils.py); restart state has a separate role described below.

## Read or extend the implementation

| File | Responsibility |
|---|---|
| [evaluation.py](evaluation.py) | Prediction routes, scoring helpers and shared CV summaries |
| [sklearn_pipeline.py](sklearn_pipeline.py) | Classical fitting, CV and completed-fold recovery |
| [base.py](base.py) | PyTorch training; stable public pipeline imports |
| [tabular_models.py](tabular_models.py) | sklearn adapters and configurable MLPs |
| [hf_models.py](hf_models.py), [hf_training.py](hf_training.py) | Text model construction/inference and Trainer fitting |
| [llama_models.py](llama_models.py) | Frozen llama.cpp feature adapter |
| [benchmark.py](benchmark.py) | Model/augmentation combinations and artifact registration |
| [ssl_algorithms.py](ssl_algorithms.py) | Shared PseudoLabel, PiModel and MeanTeacher losses |

Add an adapter in its corresponding model module, expose it through the
appropriate registry, then check fitting, score shapes, class ordering and saved
model reloads. Public imports from `base.py` and `models.py`, including the shorter
historical `pipelines_torch` paths, remain available for existing code and pickles.

Semi-supervised vision classes are exposed by [ss_vision_models.py](ss_vision_models.py).
Their `ssl_loss` returns supervised loss, unsupervised loss and logs.
[ss_models.py](ss_models.py) supplies `SemiSupervisedTabular.step`, which combines
the two losses and returns `(total_loss, logs)`; call `post_step()` after the
optimizer step when using MeanTeacher. Labeled and unlabeled batches have
explicit roles; these are separate helpers, not an automatic choice by the CSV agent.

## Resuming interrupted training

`BenchmarkRunner` assigns a separate restart directory to each model and
augmentation configuration under `results/<run>/.restart/`. Torch fitting saves
model and best validation weights, optimizer, scheduler, mixed precision scaler,
shuffle state, random streams and training progress atomically. A scheduler
`SIGUSR1` or `SIGTERM` requests a save at the next completed minibatch and raises
`CheckpointPause`. Running the same configuration and data again restores the
saved state. Float32 NumPy inputs share tensor storage when their representation
allows it.

Hugging Face QLoRA fitting uses native Trainer checkpoints, including optimizer,
scheduler, scaler and RNG files. The checkpoint directory becomes visible only
after all files are saved. Best-model paths are remapped when the atomic staging
directory is committed. Completed models preserve the random streams used to
initialize later benchmark models.

Classical k-fold fitting saves each completed fold and the final full-data fit.
An unfinished classical fit restarts that fold because arbitrary estimators do
not expose resumable optimizer state. Restart validation rejects changed data
or configuration. These restart files complement the final model registry.

Text classifiers align both nested and outer special-token IDs with their
tokenizer. Attached Gemma 4 and causal fallback heads normalize PEFT's forwarded
hidden-state option before calling the backbone.

Saved QLoRA classifiers use the same architecture-specific head loader as
training, including Gemma 4, Nanbeige and LFM. Explicit CPU reload requests are
honored even when CUDA is available. Attached heads declare mean-loss gradient
accumulation semantics, consume Trainer-only keyword arguments and disable
unused decoding caches.

## Validation

Run the relevant checks from the repository root:

```bash
PYTHONPATH=. uv run python -m pytest ml_pipeline/tests/test_compute_score.py \
  ml_pipeline/tests/test_training_restart.py \
  ml_pipeline/tests/test_model_import_compatibility.py -q
```

PEFT-specific tests require the text dependencies. See the
[validation record](../../docs/cleanup_validation.md) for the environments and
checks actually run in this checkout. CPU fixtures do not establish real-model
GPU accuracy or compatibility with every optional backend.
