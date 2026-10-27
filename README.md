# ML Playground

ML benchmarking utilities and a Streamlit assistant for exploring CSV datasets.
An LLM proposes a prediction task, prepares the data and revises experiments
within a user-defined budget. Models share evaluation splits, and each workspace
saves the code, predictions and observable conversation.

`ml_pipeline` is also used by [NutriCoach](https://github.com/oremaz/nutricoach)
and [AI Content Robustness](https://github.com/oremaz/ai-content-robustness).

## Read the code and evidence

The core design separates LLM proposals from measured results: the assistant
chooses experiments, while the runtime owns splits, fitting, scoring and saved
artifacts. Follow this path to inspect that boundary:

| Component | Start here | What to inspect |
|---|---|---|
| Experiment lifecycle | [agent_loop.py](ml_pipeline/agent_loop.py) | Bounded trials, model approval, selection and final evaluation |
| Trial constraints | [agent_trial_config.py](ml_pipeline/agent_trial_config.py) | Approved changes, complete-fold selection and baseline comparisons |
| Execution contract | [playground_runtime.py](ml_pipeline/playground_runtime.py) | Dataset preparation, frozen splits and benchmark dispatch |
| Standard comparison UI | [standard_benchmark_app.py](ml_pipeline/standard_benchmark_app.py) | Raw tables, explicit splitters and saved reports |
| Standard comparisons | [skore_benchmark.py](ml_pipeline/skore_benchmark.py) | Shared folds and native skrub/skore evaluation and reporting |
| Foundation-model adapter | [kumo_models.py](ml_pipeline/pipelines_torch/kumo_models.py) | Training context, class ordering, median regression and reload state |
| Model adapters | [tabular_models.py](ml_pipeline/pipelines_torch/tabular_models.py) | sklearn/PyTorch interfaces and baseline controls |
| Restart state | [training_checkpoint.py](ml_pipeline/utils/training_checkpoint.py) | Atomic saves, random streams and interruption handling |
| Evaluation evidence | [logged experiments](experiments/autonomous/README.md) | Reserved scores, replay checks and complete conversations |

The four logged agent evaluations have mixed outcomes: bikes improve on reserved
data, finance keeps its initial model, and energy and activity worsen. The
[explanation review](experiments/autonomous/explanation_review.md) records mistakes
that execution checks alone do not catch. These records demonstrate an inspectable
workflow, not consistent gains from autonomous search.

A minimal library example uses an explicit train/test split and a registry model:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from ml_pipeline.pipelines_torch.models import MODEL_REGISTRY

X, y = make_classification(n_samples=200, n_features=8, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)
model = MODEL_REGISTRY["linear_classifier"]()
model.fit(X_train, y_train)
print((model.predict(X_test) == y_test).mean())
```

This illustrates the adapter API; the assistant's development-fold and final
selection protocol is described below.

## Run

The core library supports Python 3.11+; the research/evaluation extras need
3.11.4+ for skore. Toto 2.0 requires Python 3.12+.

```bash
uv venv
uv pip install -e ".[research,dev]"
PYTHONPATH=. uv run streamlit run ml_pipeline/benchmark_app.py
```

Add `uv pip install -e ".[evaluation]"` for [skrub/skore comparisons](docs/skore.md).
In the app, select **Standard comparison** to run classical models on raw CSV
columns, choose independent/group/chronological folds, and reopen cached reports.
Use development data here and retain a separate final test set. The existing
**Manual benchmark** workflow remains available for augmentation, native training
and historical artifacts.
Kumo Tabular uses `uv pip install -e ".[kumo]"` in a separate Torch 2.7+
environment; its registry keys are `kumo_classifier` and `kumo_regressor`, with
`size="small"`, `"medium"` or `"large"`. Existing Torch 2.6 experiment environments
remain usable without that extra. See the [dated model audit](docs/model_audit.md)
for provider/size-preserving changes, revision pins and validation limits.

Add `uv pip install -e ".[forecasting]"` for forecasting foundation models.
Optional backends need their own packages and weights. For the core library and
CPU tests, use `.[dev]` instead of `.[research,dev]`.

The assistant supports OpenRouter, Groq and local Qwen. Set `OPENROUTER_API_KEY`
or `GROQ_API_KEY`, or configure the provider in the sidebar. Defaults are
`openrouter/free`, `qwen/qwen3.8-27b` and local `Qwen/Qwen3.8-27B`; IDs are
editable. Reasoning defaults to `high` for hosted providers and `xhigh` for local
Qwen. Local BF16 inference needs roughly 58 GB for weights plus working memory;
optional 4-bit loading uses bitsandbytes.

## CSV assistant

1. Upload CSVs, optionally describe the objective, and set a budget of **1 to 5 trials**.
2. The assistant reads pandas profiles and sample rows, proposes a target, joins,
   preprocessing and evaluation protocol, then generates the preparation script.
   Inspect the proposal and approve or edit the model families once.
3. The agent diagnoses errors and compares approved models, feature subsets, preprocessing
   and parameters. Any trial can compare multiple models. Supported CPU methods run in
   up to three isolated worker processes, with bounded threads.
4. After selection is locked, the app evaluates the saved model on reserved rows.
   Follow-ups reuse saved scripts and results. A manual script-review workflow is also available.

Preparation and inference errors have bounded automatic recovery with saved failure logs.
Interrupted runs preserve committed comparisons; incomplete trials consume budget.

Native tools cover data inspection, model approval, feature preparation,
benchmarking, trial diagnostics, additional metrics and visualization. A persisted
state machine handles the loop without a separate orchestration framework.
Measured findings come from code; the LLM proposes hypotheses and follow-up tests.
Conversational explanations still need review, and more trials do not guarantee gains.

Autonomous tabular sessions freeze three development folds: expanding chronological
windows for temporal data, disjoint groups for grouped data, and stratified/plain folds
for independent classification/regression rows. Selection uses the mean primary score;
final evaluation uses the saved model from the last fold, without a test-driven refit.
The training-mean/prior constant is an eligible untuned fallback. Regularized linear
models provide another simple control when approved.

Standard classification metrics include accuracy, macro F1, ROC-AUC and binary
average precision. Regression uses RMSE, MAE and R². Each method's scores are
reported alongside a constant reference derived only from training labels.
The agent can define additional diagnostics such as recall for a class, precision within a
review budget, or error on a specified subgroup. Definitions are explicit,
immutable and applied to all compared trials. These diagnostics supplement the
selection metric, which stays fixed during search. Diagnostic names use `custom_`;
the harness adds a missing prefix automatically and returns the canonical name.

## Models and evaluation

| Family | Implementations |
|---|---|
| Tabular | Kumo Tabular (small/medium/large), Constant/prior, Ridge/logistic regression, Random Forest, XGBoost, LightGBM, CatBoost, MLPs, stacking, TabICLv2, TabFM |
| Forecasting | Timestral-209M, Toto 2.0, TimesFM 3.0, Chronos-2, naive baselines |
| Text and vision | Hugging Face/PEFT, GGUF, torchvision/timm and image augmentation utilities |

The assistant supports tabular classification, regression and a separate
univariate forecasting task. Text/GGUF wrappers are available in the manual
workflow. Forecasting compares identical past-only windows and reports median
forecasts, MAE/RMSE/sMAPE and available quantiles. It does not automatically
convert forecasts into calibrated event probabilities. Upstream model licenses
apply; TimesFM 3.0 pretrained weights are non-commercial.

Holdout, group, temporal and cross-validation splits are available. Iterative
search uses frozen development folds and a reserved final partition. Forward
labels require their observation-end timestamps so temporal boundaries can be
purged. Preprocessors fit on training rows; resampling affects training only.
Incomplete comparisons cannot win selection. Binary probability metrics use the
saved label encoding's class 1 as positive. Final evaluation currently supports
saved sklearn-compatible estimators and the forecasting adapters.

See the [augmentation guide](ml_pipeline/data_augmentation/README.md) for input/output contracts and training-only transformations.

See [pipeline documentation](ml_pipeline/pipelines_torch/README.md) for model
registries, training, checkpoints and resuming interrupted experiments.

## Saved experiments

Workspaces live in `results/playground_memory/`, configurable with
`ML_PIPELINE_PLAYGROUND_MEMORY`. They retain CSVs, proposals, scripts, parameters,
splits, checkpoints, predictions and observable conversations/tool calls.
The app reopens saved workspaces and exports JSON/Markdown histories. Private
model thinking and provider credentials are excluded from conversation exports.

Generated Python executes on the trusted server with that process's file and
network permissions. Uploaded data and logs remain in local shared workspaces;
profiles and conversation context are sent to the selected LLM provider.
Long agent requests compact repeated inspection details; full tool outputs
remain in the saved execution traces.

The latest [four logged evaluations](experiments/autonomous/README.md) include
complete conversations for [energy](experiments/autonomous/energy/conversation.md),
[bike sharing](experiments/autonomous/bikes/conversation.md),
[wearable activity](experiments/autonomous/activity/conversation.md) and
[finance](experiments/autonomous/finance/conversation.md), using local Qwen3.8-27B
with xhigh reasoning. Reserved-data results are mixed: bikes improved, finance
retained its first-trial model, and energy and activity worsened. The
[action and explanation review](experiments/autonomous/explanation_review.md)
also identifies an ineffective feature experiment and unsupported conversational
claims. These retrospective runs demonstrate the workflow, not general superiority.

## Development

```bash
PYTHONPATH=. uv run python -m pytest ml_pipeline/tests -q
```

Tests for optional backends require the corresponding extras. Agent evaluation records
completion, recoveries, tool protocol validity, models per trial, scores and explanation
review candidates. `python -m ml_pipeline.agent_evaluation records... --output comparison.json`
compares LLM runs only when data, objective, targets, folds, harness and budgets match;
keyword review flags are not automatic judgments of scientific correctness.

The agent loop delegates instructions and schemas to `agent_protocol.py`, trial
constraints to `agent_trial_config.py` and validation plots to `agent_trial_views.py`.
Model implementations are separated into tabular, Hugging Face and llama.cpp
adapters. `pipelines_torch/models.py` retains the registries and public class
imports, including historical module identities used by serialized models.
The cleanup preserves training logic, seeds, checkpoint formats and experiment
commands. Model-default and optional-dependency updates are listed separately in
the [model audit](docs/model_audit.md); use recorded model IDs and environments
for historical runs.
Reproducing a logged experiment requires its recorded data, environment, source
revision and saved configuration, not only the current README command.

Source lives in `ml_pipeline/`. Compatibility imports under `pipelines_torch`,
`utils` and `data_augmentation` support existing consumers and checkpoints.

The [2026-10-27 cleanup checks](docs/cleanup_validation.md) record the current
validation commands, results and environment limitations.
