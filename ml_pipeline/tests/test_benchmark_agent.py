import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from ml_pipeline import benchmark_agent as agent
from ml_pipeline import playground_runtime as runtime
from ml_pipeline.utils.playground_trace import read_events


@pytest.fixture
def table():
    return pd.DataFrame({"x": np.arange(40, dtype=float), "category": ["a", "b"] * 20,
                         "target": [0, 1] * 20, "group": np.repeat(np.arange(10), 4),
                         "time": pd.date_range("2020-01-01", periods=40).astype(str)})


@pytest.fixture
def config():
    return {"task": "classification", "table": "table.csv", "target": "target",
            "features": ["x", "category"], "models": ["random_forest_classifier"],
            "augmentations": ["none"], "preprocessing": "standard", "split": "kfold",
            "folds": 2, "seed": 17, "epochs": 1,
            "model_params": {"random_forest_classifier": {"n_estimators": 4, "n_jobs": 1}}}


def test_profiles_bound_samples_and_columns():
    frame = pd.DataFrame({"long": ["x" * 200, None], "number": [1, np.inf], "extra": [2, 3]})
    profile = agent.profile_tables({"data": frame}, sample_rows=1, max_columns=2)["data"]
    assert profile["rows"] == 2
    assert profile["omitted_columns"] == 1
    assert profile["column_names"] == ["long", "number", "extra"]
    assert len(profile["sample"]) == 1
    assert len(profile["sample"][0]["long"]) == 160
    json.dumps(profile, allow_nan=False)


def test_wide_table_proposal_keeps_unprofiled_features_and_excludes_identifiers(tmp_path, monkeypatch):
    frame = pd.DataFrame({**{f"x{i}": [0., 1.] for i in range(80)}, "subject": [1, 2], "target": [0, 1]})
    workspace = agent.create_workspace(tmp_path, [("wide.csv", frame.to_csv(index=False).encode())])
    monkeypatch.setattr(agent, "complete_json", lambda *a, **k: {
        "task": "classification", "table": "wide.csv", "target": "target", "features": ["*"],
        "exclude_features": ["subject"], "models": ["random_forest_classifier"], "split": "group", "split_column": "subject"})
    plan = agent.propose(workspace, {}, "Generalize to unseen people")
    assert plan["features"] == [f"x{i}" for i in range(80)]


def test_workspace_preserves_uploaded_bytes_and_safe_filenames(tmp_path):
    content = b"x,target\n1,0\n2,1\n"
    workspace = agent.create_workspace(tmp_path, [("../../outside.csv", content)])
    manifest = agent.read_json(workspace / "tables.json")
    assert (workspace / manifest["../../outside.csv"]["path"]).read_bytes() == content
    assert not (tmp_path / "outside.csv").exists()


def test_preprocessing_learns_training_rows_only():
    preprocessor = runtime.make_preprocessor({})
    preprocessor.fit(pd.DataFrame({"x": [1.0, 3.0], "c": ["known", "known"]}))
    transformed = preprocessor.transform(pd.DataFrame({"x": [np.nan, 1000], "c": ["new", "new"]}))
    assert transformed[0, 0] == pytest.approx(0)
    assert transformed[1, 0] == pytest.approx(998)
    assert transformed.shape == (2, 2)


def test_reference_predictors_use_training_labels_not_validation_labels():
    reference = runtime.reference_scores(np.array([1., 3.]), np.array([100., 100.]), "regression")
    assert reference["name"] == "training_mean" and reference["prediction"] == 2
    assert reference["scores"]["mae"] == 98
    reference = runtime.reference_scores(np.array([0, 0, 0, 1]), np.array([0, 1]), "classification")
    assert reference["prediction"] == 0 and reference["scores"]["accuracy"] == .5
    assert reference["scores"]["pr_auc"] == .5 and reference["scores"]["roc_auc"] == .5


def test_skrub_preprocessing_handles_new_categories_and_missing_values():
    pytest.importorskip("skrub")
    preprocessor = runtime.make_preprocessor({"preprocessing": "skrub"})
    train = pd.DataFrame({"x": [1.0, 3.0, 2.0, 5.0], "c": ["a", "b", "a", "b"]})
    preprocessor.fit(train)
    validation = preprocessor.transform(pd.DataFrame({"x": [np.nan], "c": ["new"]}))
    assert len(validation) == 1
    assert np.isfinite(validation).all()


@pytest.mark.parametrize("strategy,column", [("holdout", ""), ("kfold", ""), ("group", "group"), ("temporal", "time")])
def test_reproducible_splits_and_boundaries(table, config, strategy, column):
    config.update(split=strategy, split_column=column)
    y = table.target.to_numpy()
    splits = runtime.split_indices(table, y, config)
    repeated = runtime.split_indices(table, y, config)
    for (train, validation), (a, b) in zip(splits, repeated):
        np.testing.assert_array_equal(train, a)
        np.testing.assert_array_equal(validation, b)
        assert not set(train) & set(validation)
        if strategy == "group":
            assert not set(table.group.iloc[train]) & set(table.group.iloc[validation])
        if strategy == "temporal":
            assert table.time.iloc[train].max() < table.time.iloc[validation].min()


def test_temporal_equal_timestamps_stay_together(table, config):
    table["time"] = np.repeat(pd.date_range("2020-01-01", periods=10).astype(str), 4)
    config.update(split="temporal", split_column="time", validation_fraction=0.23)
    train, validation = runtime.split_indices(table, table.target.to_numpy(), config)[0]
    assert table.time.iloc[train].max() < table.time.iloc[validation].min()


def test_temporal_split_purges_overlapping_label_windows(table, config):
    table["label_end"] = pd.to_datetime(table.time) + pd.Timedelta(days=5)
    config.update(split="temporal", split_column="time", label_end_column="label_end")
    train, validation = runtime.split_indices(table, table.target.to_numpy(), config)[0]
    boundary = pd.to_datetime(table.time.iloc[validation]).min()
    assert table.label_end.iloc[train].max() < boundary
    assert len(train) + len(validation) < len(table)
    with pytest.raises(ValueError, match="require a temporal split"):
        runtime.split_indices(table, table.target.to_numpy(), {**config, "split": "holdout"})


def test_dataset_rejects_target_leakage_and_missing_targets(table, config):
    with pytest.raises(ValueError, match="excluding the target"):
        runtime.prepare_dataset(table, {**config, "features": ["target"]})
    table.loc[0, "target"] = np.nan
    with pytest.raises(ValueError, match="missing targets"):
        runtime.prepare_dataset(table, config)


def test_classification_metrics_use_positive_class_probabilities():
    y = np.array([0, 1, 0, 1])
    probabilities = np.array([[0.9, 0.1], [0.3, 0.7], [0.7, 0.3], [0.1, 0.9]])
    scores = runtime.score_predictions(y, probabilities.argmax(1), probabilities, "classification")
    assert scores["pr_auc"] == scores["roc_auc"] == 1


def test_benchmark_persists_fold_models_preprocessors_and_metrics(tmp_path, table, config):
    output = tmp_path / "run"
    summary = runtime.benchmark(table, config, output)
    assert summary.loc[0, "completed_folds"] == 2
    assert summary.loc[0, "expected_folds"] == 2
    assert len(list(output.glob("*.pkl"))) == 2
    splits = agent.read_json(output / "splits.json")
    for fold, indices in enumerate(splits, 1):
        preprocessor = joblib.load(output / f"preprocessor_fold_{fold}.joblib")
        assert preprocessor[-1].mean_[0] == pytest.approx(table.x.iloc[indices["train"]].mean())
    assert len(pd.read_csv(output / "fold_metrics.csv")) == 2
    statistics = agent.read_json(output / "dataset_summary.json")
    assert statistics["prepared_rows"] == len(table)
    assert statistics["splits"][0]["validation_class_counts"] == {"0": 10, "1": 10}
    assert statistics["splits"][0]["validation_positive_prevalence"] == 0.5


def test_failed_method_does_not_hide_successful_baseline(tmp_path, table, config):
    config["models"].append("mlp_classifier")
    config["model_params"]["mlp_classifier"] = {"hidden_dims": ["bad"]}
    output = tmp_path / "run"
    summary = runtime.benchmark(table, config, output, trace_workspace=tmp_path)
    assert summary.model.tolist() == ["random_forest_classifier"]
    assert len(agent.read_json(output / "errors.json")) == 2
    assert len([event for event in read_events(tmp_path)
                if event["event"] == "execution.error" and event["operation"] == "model.benchmark"]) == 2


@pytest.mark.parametrize("task,model", [("classification", "mlp_classifier"), ("regression", "mlp_regressor")])
def test_torch_and_regression_paths(tmp_path, table, config, task, model):
    config.update(task=task, models=[model], split="holdout", batch_size=8, model_workers=2,
                  model_params={model: {"hidden_dims": [4], "dropout": 0}})
    if task == "regression":
        table["target"] = table.x * 0.4
    summary = runtime.benchmark(table, config, tmp_path / "run")
    assert len(summary) == 1
    assert "f1_macro" in summary if task == "classification" else "rmse" in summary
    parallelism = agent.read_json(tmp_path / "run" / "parallelism.json")
    assert parallelism["requested"] == 2 and parallelism["effective"] == 1


def test_generated_revisions_include_previous_code_and_errors(tmp_path, table, config, monkeypatch):
    workspace = agent.create_workspace(tmp_path, [("table.csv", table.to_csv(index=False).encode())])
    agent.write_json(workspace / "plan.json", config)
    source = "def build_dataset(tables, config):\n    return tables[config['table']].copy()\n"
    payloads = []
    def complete(settings, system, payload, **kwargs):
        payloads.append(payload)
        return {"script": source, "explanation": "Prepared data"}
    monkeypatch.setattr(agent, "complete_json", complete)
    first, _ = agent.generate_script(workspace, {}, config)
    agent.write_json(workspace / "last_result.json", {"error": "bad join"})
    second, _ = agent.generate_script(workspace, {}, config, "Fix the join")
    assert first != second
    assert payloads[-1]["previous_script"] == source
    assert payloads[-1]["last_result"]["error"] == "bad join"
    assert agent.read_json(second.with_suffix(".json")) == config


def test_truncated_provider_response_is_rejected(monkeypatch):
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    monkeypatch.setattr(agent, "create_client", lambda *args, **kwargs: client)
    with pytest.raises(ValueError, match="truncated"):
        agent.complete_json({"provider": "openrouter", "model": "openrouter/free"}, "prompt", {})


def test_execution_environment_removes_provider_credentials(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "private")
    monkeypatch.setenv("GROQ_API_KEY", "private")
    environment = agent.execution_environment()
    assert "OPENROUTER_API_KEY" not in environment
    assert "GROQ_API_KEY" not in environment
    assert environment["PYTHONPATH"]


def test_saved_script_executes_validated_join_in_subprocess(tmp_path, table, config):
    lookup = pd.DataFrame({"group": np.arange(10), "group_value": np.arange(10) * 3})
    workspace = agent.create_workspace(tmp_path, [
        ("table.csv", table.to_csv(index=False).encode()),
        ("lookup.csv", lookup.to_csv(index=False).encode()),
    ])
    config.update(features=["x", "category", "group_value"], output="runs/joined", split="holdout")
    agent.write_json(workspace / "execution.json", config)
    script = workspace / "experiment.py"
    script.write_text(
        "from ml_pipeline.playground_runtime import run_experiment\n"
        "def build_dataset(tables, config):\n"
        "    return tables['table.csv'].merge(tables['lookup.csv'], on='group', validate='many_to_one')\n"
        "if __name__ == '__main__':\n"
        "    run_experiment(build_dataset)\n")
    result = subprocess.run([sys.executable, str(script), str(workspace / "execution.json")],
                            cwd=workspace, env=agent.execution_environment(), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert (workspace / "runs/joined/summary.csv").exists()
    events = read_events(workspace)
    operations = {event["operation"] for event in events if event["event"] == "execution.end"}
    assert {"experiment.run", "data.load", "data.prepare", "preprocessing.fit_transform", "model.benchmark"} <= operations


def test_streamlit_preserves_reviewed_configuration_after_generation(tmp_path, table, config, monkeypatch):
    from streamlit.testing.v1 import AppTest

    workspace = agent.create_workspace(tmp_path, [("table.csv", table.to_csv(index=False).encode())])
    agent.write_json(workspace / "plan.json", config)
    monkeypatch.setenv("ML_PIPELINE_PLAYGROUND_MEMORY", str(tmp_path))
    monkeypatch.setattr(agent, "complete_json", lambda *args, **kwargs: {
        "explanation": "Ready", "script":
        "from ml_pipeline.playground_runtime import run_experiment\n"
        "def build_dataset(tables, config):\n    return tables[config['table']]\n"
        "if __name__ == '__main__':\n    run_experiment(build_dataset)\n"})
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "benchmark_app.py"), default_timeout=60)
    app.session_state["playground_workspace"] = str(workspace)
    app.run()
    assert not app.exception
    next(widget for widget in app.number_input if widget.label == "Training epochs").set_value(3).run()
    next(button for button in app.button if button.label == "Generate or revise benchmark script").click().run()
    assert not app.exception
    assert next(widget for widget in app.number_input if widget.label == "Training epochs").value == 3
    assert not next(button for button in app.button if button.label == "Run benchmark").disabled
    next(button for button in app.button if button.label == "Run benchmark").click().run(timeout=180)
    assert not app.exception
    result = agent.read_json(workspace / "last_result.json")
    assert result["returncode"] == 0, result["log"]
    assert result["config"]["epochs"] == 3
    assert result["labels"] == {"0": "0", "1": "1"}
    assert result["summary"]
    reopened = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "benchmark_app.py"), default_timeout=60)
    reopened.session_state["playground_workspace"] = str(workspace)
    reopened.run()
    assert not reopened.exception
    assert next(widget for widget in reopened.number_input if widget.label == "Training epochs").value == 3
    assert not next(button for button in reopened.button if button.label == "Run benchmark").disabled
    assert any(expander.label == "Conversation and execution trace" for expander in reopened.expander)
    assert {event["operation"] for event in read_events(workspace)} >= {"script.execute", "experiment.run"}
