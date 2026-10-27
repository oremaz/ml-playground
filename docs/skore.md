# skrub/skore and the custom experiment harness

`skrub` was already available through `preprocessing="skrub"`, implemented as a
TableVectorizer followed by imputation and scaling. `skore` is now an optional
comparison/reporting backend. For new ordinary sklearn comparisons, prefer
[skore_benchmark.py](../ml_pipeline/skore_benchmark.py), which delegates fitting,
metrics and reports to the libraries rather than recreating their evaluation engine.

## What adds value beyond the agent?

| Responsibility | Existing contribution | Recommended owner |
|---|---|---|
| Mixed-table preprocessing | Standard encoders or a skrub TableVectorizer | skrub for new estimator-aware pipelines |
| Classical cross-validation, metrics and plots | Considerable bespoke code | skore for new standard comparisons |
| Group or temporal CV | A custom split builder and validation rules | Supply explicit splits to skore; this capability is not unique to this repo |
| Forward-label purging and frozen research partitions | Observation-end checks, hashed manifests and row identities | Keep the domain protocol around the evaluation backend |
| Torch/Hugging Face training recovery | Optimizer, scheduler, scaler, sampler and RNG state; scheduler signals | Keep the native training/checkpoint layer |
| Training-only resampling | Fold-local augmentations and existing experiment flags | Retain the current route, or explicitly build a compatible resampling pipeline |
| Foundation-model forecasting | Past-only origins, horizons, quantiles and adapter metadata | Keep the forecasting runtime |
| Replaying existing experiments | Saved models, predictions, preprocessing and source receipts | Preserve the original artifacts, environment and final-model policy |

There is useful engineering outside the agent, especially the experiment protocol
and resumable native training. Routine sklearn fitting, CV summaries and plots do
not provide a compelling reason to maintain a parallel reporting framework.
No local head-to-head study establishes better predictive performance than skrub/skore.

Skore already accepts explicit CV splitters and index iterables, caches prediction
results, and compares fitted reports. It can evaluate any suitably sklearn-compatible
estimator; neural models are not inherently excluded. What it does not replace here
is the native optimizer/checkpoint implementation or the domain-specific protocol.
Sources: [skore evaluation API](https://docs.skore.probabl.ai/stable/reference/api/skore.evaluate.html),
[report structure and caching](https://docs.skore.probabl.ai/stable/user_guide/reporters.html),
[skrub tabular_pipeline](https://skrub-data.org/stable/generated/skrub.tabular_pipeline.html).

## Run a standard comparison

The app's **Standard comparison** workflow now calls this backend directly.
Upload a CSV or use a local path, select a target and models, and choose seeded
independent folds, disjoint groups or chronological row order. Raw features go
straight to fold-local skrub pipelines; the legacy manual UI's whole-table
preparation is not used. Chronological rows must be sorted before loading, and
forward-label purging still belongs to the research runtime.

Per-model JSON uses the names shown in that UI, for example
`{"random_forest": {"n_estimators": 200}, "kumo": {"size": "small", "device": "cpu"}}`.
Use model-native preprocessing for Kumo's mixed-table recipe. Comparisons save
under `results/standard_comparisons/` (respecting `ML_PIPELINE_RESULTS_DIR`), and
the UI can reopen those reports and download their metrics. It fits development
folds and does not designate or retrain a deployable final model.

The previous manual runner and agent have not been migrated automatically.
Their augmentation, restart, metric, artifact and final-model conventions remain
available. Accordingly this change simplifies new standard comparisons; it does
not eliminate all bespoke evaluation code or prove backend-equivalent replay.

Install in the ML Playground environment with `uv pip install -e ".[evaluation]"`.
Skore 0.28 requires Python 3.11.4+. This complete CPU example uses synthetic data:

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from ml_pipeline.skore_benchmark import evaluate_models

X, y = make_regression(n_samples=100, n_features=6, random_state=42)
report = evaluate_models(
    {"Ridge": Ridge(),
     "Random Forest": RandomForestRegressor(random_state=42),
     "Gradient Boosting": HistGradientBoostingRegressor(random_state=42)},
    X, y, splitter=5, n_jobs=1,
)
print(report.metrics.summarize().frame())
```

The helper applies `skrub.tabular_pipeline` to each estimator. Use
`preprocessing="none"` if supplying already constructed pipelines or intentionally
using prepared arrays without additional preprocessing. Pass an explicit
`pos_label` for a binary metric convention. Classification and regression models
cannot be mixed in one report.

Integer `splitter=5` retains skore's unshuffled (stratified) folds. The legacy
BenchmarkRunner uses shuffled, seeded folds. Use an explicit seeded sklearn
splitter or saved indices to preserve an existing comparison's actual protocol.
Changing the backend does not justify changing the folds.

Use `registry_estimator("linear_regressor", alpha=...)` to obtain a cloneable
native estimator from the existing registry. Kumo is also sklearn-compatible;
use `preprocessing="none"` when its native table recipe should handle mixed
columns, and select its size/device explicitly. Native Torch/QLoRA/GGUF adapters
keep their dedicated training workflows instead of being silently converted.

## Save, restore and reuse

Set `output_dir` to a **new directory** to save `protocol.json`, `metrics.csv`
and `report.joblib`. The protocol records the materialized row indices. The
report stores each CV report through skore's `to_dict`/`from_dict` state interface
and reconstructs comparisons with `skore.compare`, without refitting.
`load_report(path)` restores the cached report without fitting again. One model
returns a CrossValidationReport; several return a ComparisonReport.

Only pass development data to a search comparison. For an already fitted model
and a final test set, skore's explicit `splitter="prefit"` interface can create
an evaluation report without another split or training fit. The helper above is
for CV; it deliberately does not select when final evaluation is permitted.

Existing agent runs, checkpoint names, canonical metric CSVs, class ordering and
last-development-fold final-model selection are unchanged. The old path remains
available for their exact replay and for the native features not migrated here.
Automatic replacement would change experimental semantics rather than merely
shorten the implementation.

## Validation

The integration checks compare predictions with independently fitted skrub
pipelines on identical indices, enforce group/row disjointness, and verify report
persistence without additional fitting. Run:

```bash
PYTHONPATH=. uv run python -m pytest ml_pipeline/tests/test_skore_benchmark.py -q
```

On a filesystem-restricted VM, `SKB_DATA_DIRECTORY` selects a writable skrub cache.
Use `n_jobs=1` for GPU foundation models or a small CPU allocation; parallel jobs
are an explicit resource choice, not part of the score definition.
