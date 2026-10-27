# Cleanup validation (2026-10-27)

The cleanup separates the agent protocol, trial constraints and validation views,
and separates model adapters from the stable registry module. Experiment commands,
splits, seeds and checkpoint formats are preserved. Model and dependency
updates are recorded separately in [the model audit](model_audit.md).

## Checks in this checkout

- Shared VM environment: **185 passed, 2 skipped, 6 deselected**.
- Research environment with PEFT: the six deselected checks **all passed**.
- The README's synthetic-data example and Streamlit startup passed.
- Fitted sklearn and PyTorch objects serialized using the original pre-cleanup
  source reloaded using the new modules with identical predictions.
- Syntax-tree comparisons preserved all 28 agent functions, 25 model class/function
  definitions, and both agent instruction strings. The wrapper's fit method is
  inherited from its training mixin; numerical method bodies are preserved.

Run from the ML Playground root. This VM has no project-local environment, so
these checks reused existing environments without synchronizing dependencies:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2
PYTHONPATH=. UV_PROJECT_ENVIRONMENT=../../.venv uv run --no-sync python -m pytest \
  ml_pipeline/tests -q -p no:cacheprovider \
  -k 'not attached_classifier_adapter_reload and not gemma_classifier_accepts_peft and not attached_causal_classifier_gradient'
PYTHONPATH=. UV_PROJECT_ENVIRONMENT=../ai-content-robustness/.venv uv run --no-sync python -m pytest \
  ml_pipeline/tests/test_checkpoint.py ml_pipeline/tests/test_training_restart.py \
  -q -p no:cacheprovider \
  -k 'attached_classifier_adapter_reload or gemma_classifier_accepts_peft or attached_causal_classifier_gradient'
```

The shared environment lacks PEFT; the research environment lacks imbalanced-learn,
so neither alone runs the entire suite. No existing environment dependencies were installed or upgraded.
Historical GPU experiments and live model/provider inference were not rerun.
Use their original recorded source, environment, data and configurations when
reproducing published tables or resuming an existing experiment.

## Component documentation and model/evaluation follow-up

- Extracted evaluation and sklearn pipeline classes from `base.py`; all five
  moved class bodies match the original syntax trees apart from documentation
  whitespace and historical module identities. Compatibility tests cover the
  original imports and fitted-object serialization.
- Updated pipeline, augmentation and experiment guides to describe actual APIs,
  training-only transformations and the distinction between historical protocols.
- Repeated shared-environment suite: **197 passed, 3 skipped, 6 deselected**.
  The extra skip is the optional skore integration, absent from that environment.
- The six PEFT checks passed again in the existing research environment.
- In an isolated temporary Python 3.12 environment with Torch 2.7.1 CPU,
  SDM 0.2.1, skore 0.28.0 and skrub 0.10.0: **17 focused checks passed** after
  the final provider/model changes. These cover Kumo, skore and model-loader
  contracts. No existing experiment environment was synchronized.
- Real pretrained Kumo small classification and regression ran on tiny CPU data,
  including reload and batched prediction. A mixed nullable numeric/categorical
  classification comparison also ran through skore. These are interface checks,
  not evidence of comparative model quality.
- The Ridge/Random Forest/Histogram Gradient Boosting five-fold example ran;
  both single and comparison reports saved and reloaded without additional fits.
  Pipeline and augmentation README Python examples also ran successfully.
- Read-only Slurm inspection still showed the same **17 pending jobs** after the
  follow-up. No jobs, frozen source snapshots or saved experiment data were changed.

To repeat the optional checks, install the evaluation and Kumo extras in a
separate Torch 2.7+ environment, then run from this repository root:

```bash
PYTHONPATH=. uv run python -m pytest \
  ml_pipeline/tests/test_kumo.py \
  ml_pipeline/tests/test_skore_benchmark.py \
  ml_pipeline/tests/test_model_refresh.py -q
```

The Kumo medium/large checkpoints, gated DINOv3 inference, real Qwen vision
inference, local text augmentation and live hosted APIs were not exercised.
New defaults do not recreate results from older weights: historical reproduction
still requires the original recorded environment, source, data and configuration.

## Request audit follow-up

The earlier skore addition was a library API, not an app integration. The app now
has a **Standard comparison** workflow using that API directly, including raw
mixed-table input, seeded independent/group/chronological splitters, saved
reports and metrics download. It does not call the manual runner's whole-table
preparation. Legacy manual and agent workflows remain available with their
existing artifact and final-model policies.

- **8 checks passed** across `test_standard_benchmark_app.py` and
  `test_skore_benchmark.py`, in the isolated evaluation environment described
  above. Real classification and regression UI runs fit the three example model
  families, retain raw missing/categorical features for fold-local preprocessing,
  and reopen saved reports without refitting. Group and temporal splitter checks
  also passed.
- The first UI run caught retained feature selection after switching targets;
  feature widget state now follows the selected target/group context. The final
  eight-check run passed after that correction.
- Standard-comparison startup also passed in the shared environment without
  skore installed. Optional imports remain deferred until execution/reopening.
- Syntax, undefined-global and relative-document-link checks passed. These do
  not establish universal model rankings or full historical GPU reproduction.

Repeat the app/backend checks from this repository root with evaluation and UI
dependencies installed:

```bash
PYTHONPATH=. uv run python -m pytest \
  ml_pipeline/tests/test_standard_benchmark_app.py \
  ml_pipeline/tests/test_skore_benchmark.py -q
```

This is an additional route for standard comparisons, not a complete replacement
of the existing training framework. The substantial bespoke parts that remain,
and the reasons to preserve their experimental semantics, are listed in
[the comparison](skore.md).
