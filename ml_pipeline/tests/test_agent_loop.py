"""Behavioral tests for bounded native tool use and protected evaluation."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml_pipeline import agent_loop as agent
from ml_pipeline.benchmark_agent import read_json


def legacy_session(*args, **kwargs):
    return agent.create_session(*args, structured_notes=False, **kwargs)


def config():
    return {"task": "classification", "target": "target", "features": ["x", "z"],
            "models": ["random_forest_classifier"], "augmentations": ["none"],
            "preprocessing": "standard", "split": "temporal", "split_column": "date",
            "label_end_column": "end", "validation_fraction": .2, "metric": "pr_auc", "seed": 42,
            "model_params": {"random_forest_classifier": {"n_estimators": 5, "max_depth": 2, "n_jobs": 1}}}


def frame():
    rng = np.random.default_rng(9)
    dates = pd.date_range("2020-01-01", periods=150)
    x = rng.normal(size=len(dates))
    return pd.DataFrame({"x": x, "z": rng.normal(size=len(x)), "target": (x > .5).astype(int),
                         "date": dates, "end": dates + pd.Timedelta(days=3)})


def call(name, **args):
    return {"id": "call_" + name, "type": "function", "function": {"name": name,
             "arguments": json.dumps({"reason": "Test a useful hypothesis.", **args})}}


def native(*calls, content=""):
    return {"role": "assistant", "content": content, "tool_calls": list(calls)}


def approve(folder, monkeypatch, budget=1):
    legacy_session(folder, frame(), config(), max_trials=budget, context="Rare event prediction")
    monkeypatch.setattr(agent, "complete_message", lambda *a, **k: native(call("request_model_selection", models=config()["models"])))
    state = agent.advance(folder, {})
    assert state["status"] == "awaiting_models"
    assert not state["trials"]
    return agent.approve_models(folder, config()["models"], source="simulated_user")


def test_one_approval_native_protocol_and_real_trial(tmp_path, monkeypatch):
    folder = tmp_path / "agent"
    approve(folder, monkeypatch)
    responses = iter([
        native(call("run_benchmark", overrides_json="{}"), content="Start with the reviewed small forest."),
        native(call("inspect_trial", trial=1), content="Accuracy alone is insufficient; inspect rare-event ranking."),
        native(call("visualize_results")),
        {"role": "assistant", "content": "The forest ranks rare events. One validation split is uncertain; no trading claim."}])
    monkeypatch.setattr(agent, "complete_message", lambda *a, **k: next(responses))
    state = agent.advance(folder, {})
    assert state["status"] == "completed"
    assert len(state["trials"]) == 1
    assert state["trials"][0]["status"] == "completed", state["trials"][0]
    assert state["best"] is not None
    messages = read_json(folder / "messages.json")
    assistant_ids = [c["id"] for m in messages for c in m.get("tool_calls", [])]
    tool_ids = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    assert assistant_ids == tool_ids
    assert sum("simulated_user" in m.get("content", "") for m in messages) == 1
    assert (folder / "conversation.md").exists()
    assert (folder / "validation.png").exists()
    diagnostic = agent._trial_diagnostics(folder, state, 1)["diagnostics"][0]
    assert "pr_auc" in diagnostic["training_scores"]
    assert diagnostic["training_rows"] == len(read_json(folder / "partitions.json")["train_in_development"])
    assert diagnostic["validation_scores"]["pr_auc"] == pytest.approx(state["best"]["pr_auc"])
    counts = diagnostic["confusion_counts"]
    assert diagnostic["predicted_positive_at_default"] == counts["true_positive"] + counts["false_positive"]
    assert counts["true_positive"] + counts["false_negative"] == state["trials"][0]["dataset_summary"]["splits"][0]["validation_class_counts"]["1"]
    before = (folder / "messages.json").read_bytes()
    evaluation = agent.evaluate_selected(folder)
    assert evaluation["test_rows"] > 0
    assert (folder / "messages.json").read_bytes() == before
    assert agent.evaluate_selected(folder) == evaluation
    with pytest.raises(ValueError, match="No model-selection"):
        agent.approve_models(folder, config()["models"])


def test_protected_partitions_and_training_label_purge(tmp_path):
    cfg = config()
    state = legacy_session(tmp_path, frame(), cfg, max_trials=3, context="Objective")
    parts = read_json(tmp_path / "partitions.json")
    assert not set(parts["test"]) & set(parts["development"])
    dev = joblib.load(tmp_path / "development.joblib")
    test = joblib.load(tmp_path / "evaluation.joblib")
    train = dev.iloc[parts["train_in_development"]]
    validation = dev.iloc[parts["validation_in_development"]]
    assert train.end.max() < validation.date.min()
    assert dev.end.max() < test.date.min()
    request = call("inspect_dataset")
    public = agent.dispatch(tmp_path, state, request)
    assert public["profiles"]["development"]["rows"] == len(dev)
    assert "test" not in json.dumps(public).lower()
    with pytest.raises(ValueError, match="user model selection"):
        agent.dispatch(tmp_path, state, call("run_benchmark", overrides_json="{}"))
    with pytest.raises(ValueError, match="Complete the agent"):
        agent.evaluate_selected(tmp_path)


def test_invalid_requests_recover_budget_and_approved_models(tmp_path, monkeypatch):
    state = approve(tmp_path, monkeypatch, budget=3)
    with pytest.raises(ValueError, match="already been requested"):
        agent.dispatch(tmp_path, state, call("request_model_selection", models=config()["models"]))
    with pytest.raises(ValueError, match="reviewed baseline"):
        agent._trial_config(state, {"models": ["lightgbm_classifier"]})
    state["trials"] = [{"number": 1, "fingerprint": "done", "summary": []}]
    with pytest.raises(ValueError, match="approval"):
        agent._trial_config(state, {"models": ["lightgbm_classifier"]})
    with pytest.raises(ValueError, match="Only approved"):
        agent._trial_config(state, {"target": "z"})
    with pytest.raises(ValueError, match="n_estimators"):
        agent._trial_config(state, {"model_params": {"random_forest_classifier": {"n_estimators": 100000}}})
    with pytest.raises(ValueError, match="reviewed numeric"):
        agent.apply_features(frame(), config()["features"], [{"name": "leak", "op": "abs", "left": "target"}])
    state["trials"] *= 3
    with pytest.raises(ValueError, match="budget exhausted"):
        agent.dispatch(tmp_path, state, call("run_benchmark", overrides_json="{}"))


def test_failure_is_returned_to_llm_and_counts_as_trial(tmp_path, monkeypatch):
    approve(tmp_path, monkeypatch, budget=3)
    def fail(*a, **k):
        raise RuntimeError("Optional dependency missing")
    monkeypatch.setattr(agent, "_subprocess", fail)
    responses = iter([native(call("run_benchmark", overrides_json="{}")),
                      native(call("run_benchmark", overrides_json="{}")),
                      {"role": "assistant", "content": "The model failed because its dependency is missing. No score is available."}])
    monkeypatch.setattr(agent, "complete_message", lambda *a, **k: next(responses))
    state = agent.advance(tmp_path, {})
    assert len(state["trials"]) == 1
    assert state["trials"][0]["status"] == "failed"
    assert "Optional dependency missing" in state["trials"][0]["error"]
    assert "Identical configuration" in state["messages"][-2]["content"]
    assert state["best"] is None


def test_rowwise_features_do_not_depend_on_future_rows():
    original = frame()
    recipes = [{"name": "scaled_x", "op": "ratio", "left": "x", "right": "z"}]
    changed = original.copy()
    changed.loc[100:, ["x", "z"]] *= 9
    pd.testing.assert_frame_equal(agent.apply_features(original, ["x", "z"], recipes).iloc[:100],
                                  agent.apply_features(changed, ["x", "z"], recipes).iloc[:100])


def test_headless_ui_resumes_model_selection_and_exports_native_messages(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from ml_pipeline.benchmark_agent import create_workspace, write_json
    from ml_pipeline.utils.playground_trace import export_history

    workspace = create_workspace(tmp_path, [("table.csv", frame().to_csv(index=False).encode())])
    cfg = config()
    cfg["table"] = "table.csv"
    folder = workspace / "agents" / "session"
    legacy_session(folder, frame(), cfg, max_trials=1, context="Rare event test")
    monkeypatch.setattr(agent, "complete_message", lambda *a, **k: native(call("request_model_selection", models=cfg["models"])))
    agent.advance(folder, {})
    write_json(workspace / "ui_config.json", cfg)
    source = f'''
from pathlib import Path
from ml_pipeline.playground_app import _iterative_assistant
from ml_pipeline.benchmark_agent import read_json
workspace = Path({str(workspace)!r})
_iterative_assistant(workspace, {{}}, "", read_json(workspace / "ui_config.json"), 120)
'''
    app = AppTest.from_string(source, default_timeout=120).run()
    assert not app.exception
    assert next(x for x in app.multiselect if x.label == "Model families for this experiment").value == cfg["models"]
    responses = iter([native(call("run_benchmark", overrides_json="{}"), content="Fit the reviewed baseline."),
                      {"role": "assistant", "content": "One trial completed. Rare-event validation remains uncertain."}])
    monkeypatch.setattr(agent, "complete_message", lambda *a, **k: next(responses))
    next(x for x in app.button if x.label == "Approve models and continue").click().run()
    assert not app.exception
    assert read_json(folder / "state.json")["status"] == "completed"
    assert len(app.dataframe) == 1
    history = export_history(workspace)
    assert "agents/session/messages.json" in history["artifacts"]
    assert any(e["event"] == "agent.tool_request" for e in history["events"])
    reopened = AppTest.from_string(source, default_timeout=120).run()
    assert not reopened.exception
    assert not any(x.label == "Approve models and continue" for x in reopened.button)


def test_repair_unsupported_inherited_parameter_without_changing_approved_family(tmp_path, monkeypatch):
    cfg = config()
    cfg["model_params"]["random_forest_classifier"]["typo_parameter"] = 10
    legacy_session(tmp_path, frame(), cfg, max_trials=3, context="Repair a real configuration error")
    responses = iter([native(call("request_model_selection", models=cfg["models"]))])
    monkeypatch.setattr(agent, "complete_message", lambda *a, **k: next(responses))
    agent.advance(tmp_path, {})
    agent.approve_models(tmp_path, cfg["models"])
    responses = iter([
        native(call("run_benchmark", overrides_json="{}")),
        native(call("run_benchmark", overrides_json=json.dumps({"drop_parameters": {
            "random_forest_classifier": ["typo_parameter"]}})),
               content="The inherited parameter is unsupported. Remove it and retry the same approved family."),
        {"role": "assistant", "content": "The repaired model completed. The failed attempt consumed one trial."}])
    state = agent.advance(tmp_path, {})
    assert [trial["status"] for trial in state["trials"]] == ["failed", "completed"]
    assert state["best"]["trial"] == 2
    assert "typo_parameter" not in state["trials"][1]["config"]["model_params"]["random_forest_classifier"]
    assert len([m for m in state["messages"] if m["role"] == "tool" and "approved_models" in json.loads(m["content"])]) == 1


def test_approved_capabilities_disclose_real_wrapper_limits_and_balancing():
    models = ["random_forest_classifier", "lightgbm_classifier", "xgboost_classifier"]
    capabilities = agent.model_capabilities(config(), models)
    json.dumps(capabilities, allow_nan=False)
    assert "scale_pos_weight" not in capabilities["lightgbm_classifier"]["supported_parameter_defaults"]
    assert "class_weight" in capabilities["lightgbm_classifier"]["supported_parameter_defaults"]
    assert "sample weights" in capabilities["lightgbm_classifier"]["automatic_balancing"]
    assert capabilities["random_forest_classifier"]["pipeline_balancing_requested"] is True
    assert capabilities["random_forest_classifier"]["automatic_balancing_enabled"] is False
    assert capabilities["lightgbm_classifier"]["automatic_balancing_enabled"] is True
    regression = agent.model_capabilities({"task": "regression"}, ["lightgbm_regressor"])
    assert "n_estimators" in regression["lightgbm_regressor"]["supported_parameter_defaults"]
    assert regression["lightgbm_regressor"]["automatic_balancing_enabled"] is False


def test_class_balancing_control_changes_fitted_probabilities(tmp_path):
    from ml_pipeline.playground_runtime import benchmark
    rng = np.random.default_rng(33)
    size = 400
    data = pd.DataFrame({"x": rng.normal(size=size), "z": rng.normal(size=size),
                         "target": (rng.random(size) < .12).astype(int)})
    cfg = {**config(), "split": "holdout", "label_end_column": "", "models": ["lightgbm_classifier"],
           "model_params": {"lightgbm_classifier": {"n_estimators": 5, "num_leaves": 3, "n_jobs": 1}},
           "strict_model_params": True, "save_predictions": True}
    means = {}
    for flag in [True, False]:
        output = tmp_path / str(flag)
        benchmark(data, {**cfg, "use_class_weights": flag}, output)
        predictions = pd.read_csv(next(output.glob("predictions_*.csv")))
        means[flag] = predictions.probability_1.mean()
    assert means[False] < means[True]


def test_reviewed_preparation_runs_join_and_rejects_changed_upload(tmp_path):
    from ml_pipeline.benchmark_agent import create_workspace
    left = frame().drop(columns="z")
    left["key"] = np.arange(len(left))
    right = pd.DataFrame({"key": left.key, "z": np.ones(len(left))})
    workspace = create_workspace(tmp_path, [("left.csv", left.to_csv(index=False).encode()),
                                             ("right.csv", right.to_csv(index=False).encode())])
    source = "def build_dataset(tables, config):\n    return tables['left.csv'].merge(tables['right.csv'], on='key', validate='one_to_one')\n"
    with pytest.raises(ValueError, match="custom preprocessing"):
        agent.prepare_reviewed_dataset(workspace, source + "\nif __name__ == '__main__':\n    run_experiment(build_dataset, custom_factory)\n", config())
    folder = agent.prepare_reviewed_dataset(workspace, source, config(), timeout=120)
    prepared = joblib.load(folder / "prepared.joblib")
    assert len(prepared) == len(left)
    assert prepared.z.eq(1).all()
    assert not (folder / "state.json").exists()
    assert not list(folder.glob("trial_*"))
    table = workspace / "data" / "table_0.csv"
    table.write_bytes(table.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="Uploaded table changed"):
        agent.prepare_reviewed_dataset(workspace, source, config(), timeout=120)


def test_script_followup_receives_selected_native_trials_without_final_scores(tmp_path, monkeypatch):
    from ml_pipeline import benchmark_agent as builder
    workspace = builder.create_workspace(tmp_path, [("data.csv", frame().to_csv(index=False).encode())])
    cfg = {**config(), "table": "data.csv"}
    builder.write_json(workspace / "plan.json", cfg)
    folder = workspace / "agents" / "selected"
    state = legacy_session(folder, frame(), cfg, max_trials=1, context="Original goal")
    state.update(status="completed", best={"trial": 1, "pr_auc": .2}, conclusion="These particular changes did not help.",
                 metric_definitions=[{"name": "custom_review", "kind": "precision_at_fraction", "fraction": .1}],
                 trials=[{"number": 1, "reason": "Baseline", "status": "completed", "config": cfg,
                          "recipes": [], "summary": [{"pr_auc": .2}]}])
    agent.save_state(folder, state)
    builder.write_json(folder / "evaluation.json", {"scores": {"pr_auc": .99}})
    builder.write_json(workspace / "active_agent.json", {"session": "selected"})
    received = []
    def complete(settings, system, payload, **kwargs):
        received.append(payload)
        return {"script": "def build_dataset(tables, config):\n    return tables[config['table']]\n", "explanation": "Ready"}
    monkeypatch.setattr(builder, "complete_json", complete)
    builder.generate_script(workspace, {}, cfg, "Revise after native trials")
    memory = received[0]["recent_agent_experiment"]
    assert memory["session"] == "selected"
    assert memory["trials"][0]["summary"] == [{"pr_auc": .2}]
    assert memory["metric_definitions"] == state["metric_definitions"]
    assert memory["final_evaluation_already_unlocked"] is True
    assert ".99" not in json.dumps(memory)
    assert received[0]["instruction"] == "Revise after native trials"


def test_better_later_trial_replaces_successful_baseline(tmp_path, monkeypatch):
    rng = np.random.default_rng(17)
    values = rng.normal(size=(600, 2))
    data = pd.DataFrame({"x": values[:, 0], "z": values[:, 1],
                         "target": ((values[:, 0] > 0) != (values[:, 1] > 0)).astype(int)})
    cfg = {**config(), "split": "holdout", "label_end_column": "",
           "model_params": {"random_forest_classifier": {"n_estimators": 1, "max_depth": 1, "n_jobs": 1}}}
    legacy_session(tmp_path, data, cfg, max_trials=3, context="Selection regression on synthetic interaction data")
    monkeypatch.setattr(agent, "complete_message", lambda *a, **k: native(
        call("request_model_selection", models=cfg["models"])))
    agent.advance(tmp_path, {})
    agent.approve_models(tmp_path, cfg["models"])
    baseline = call("run_benchmark", overrides_json="{}")
    baseline["id"] = "call_baseline"
    better = call("run_benchmark", overrides_json=json.dumps({"model_params": {
        "random_forest_classifier": {"n_estimators": 80, "max_depth": 6}}}))
    better["id"] = "call_better"
    responses = iter([native(baseline), native(better),
                      {"role": "assistant", "content": "Retain the stronger validation candidate."}])
    monkeypatch.setattr(agent, "complete_message", lambda *a, **k: next(responses))
    state = agent.advance(tmp_path, {})
    first, second = [trial["summary"][0]["pr_auc"] for trial in state["trials"]]
    assert second > first
    assert state["best"]["trial"] == 2
    assert state["best"]["pr_auc"] == second
    execution = state["trials"][1]["execution"]
    actual = execution["fitted_parameters"][0]["parameters"]
    assert actual["n_estimators"] == 80
    assert actual["max_depth"] == 6
    assert execution["balancing"]["random_forest_classifier"]["automatic_balancing_enabled"] is False
    changes = {item["setting"]: item for item in execution["changes_from_incumbent"]}
    assert changes["model_params.random_forest_classifier.n_estimators"] == {
        "setting": "model_params.random_forest_classifier.n_estimators", "before": 1, "after": 80}
    comparison = state["trials"][1]["comparison"]
    assert comparison["improved_incumbent"] is True
    assert comparison["candidates"][0]["delta_from_previous_incumbent"] == pytest.approx(second - first)
    assert (tmp_path / "trial_2" / "execution_feedback.json").exists()
    assert agent.evaluate_selected(tmp_path)["selection"]["trial"] == 2


def test_execution_feedback_exposes_baseline_inheritance_and_unchanged_balancing(tmp_path):
    cfg = config()
    cfg["models"] = ["random_forest_classifier", "lightgbm_classifier"]
    cfg["model_params"]["lightgbm_classifier"] = {"n_estimators": 5}
    baseline = {"number": 1, "config": cfg, "recipes": [], "summary": []}
    state = {"config": cfg, "approved_models": cfg["models"], "trials": [baseline], "recipes": []}
    weighted = agent._trial_config(state, {"use_class_weights": True})
    trial = {"number": 2, "config": weighted, "recipes": []}
    state["trials"].append(trial)
    result = agent._execution_feedback(tmp_path, state, trial, {"trial": 1})
    assert result["changes_from_baseline"] == []
    assert result["balancing"]["lightgbm_classifier"]["automatic_balancing_enabled"] is True
    assert result["balancing"]["random_forest_classifier"]["automatic_balancing_enabled"] is False
    unweighted = agent._trial_config(state, {"use_class_weights": False})
    state["trials"][1]["config"] = unweighted
    next_config = agent._trial_config(state, {"model_params": {"random_forest_classifier": {"max_depth": 4}}})
    next_trial = {"number": 3, "config": next_config, "recipes": []}
    state["trials"].append(next_trial)
    result = agent._execution_feedback(tmp_path, state, next_trial, {"trial": 2})
    assert result["resolved_settings"]["use_class_weights"] is True
    changes = {item["setting"]: item for item in result["changes_from_incumbent"]}
    assert changes["use_class_weights"]["before"] is False
    assert changes["use_class_weights"]["after"] is True


def test_trial_comparison_respects_error_metric_direction():
    row = {"model": "random_forest_regressor", "augmentation": "none", "rmse": 4.,
           "completed_folds": 1, "expected_folds": 1}
    first = {"number": 1, "summary": [row]}
    second = {"number": 2, "summary": [{**row, "rmse": 2.}]}
    incumbent = {"trial": 1, **row}
    state = {"config": {"metric": "rmse"}, "trials": [first, second]}
    state["best"] = agent._best(state)
    result = agent._trial_comparison(state, second, incumbent)
    assert result["direction"] == "minimize"
    assert result["improved_incumbent"] is True
    assert result["candidates"][0]["delta_from_previous_incumbent"] == -2.


def test_multiclass_inspection_omits_arbitrary_label_correlations(tmp_path):
    data = frame()
    data['target'] = np.resize(['walking', 'sitting', 'standing'], len(data))
    cfg = {**config(), 'metric': 'f1_macro'}
    state = legacy_session(tmp_path / 'original', data, cfg, max_trials=1, context='Activities')
    first = agent.dispatch(tmp_path / 'original', state, call('inspect_dataset'))
    data['target'] = data.target.map({'walking': 'A', 'sitting': 'Z', 'standing': 'B'})
    renamed = legacy_session(tmp_path / 'renamed', data, cfg, max_trials=1, context='Activities')
    second = agent.dispatch(tmp_path / 'renamed', renamed, call('inspect_dataset'))
    assert first['train_target_correlations'] == second['train_target_correlations'] == {}
    assert 'arbitrary multiclass' in first['warning']


@pytest.mark.parametrize('task', ['regression', 'classification'])
def test_task_specific_validation_plot_uses_selected_predictions(tmp_path, monkeypatch, task):
    import matplotlib.axes
    root = tmp_path / 'trial_1' / 'output'
    root.mkdir(parents=True)
    row = {'model': 'example', 'augmentation': 'none', 'checkpoint_id': 'candidate',
           'completed_folds': 1, 'expected_folds': 1, 'rmse': .5, 'f1_macro': .8}
    pd.DataFrame([row]).to_csv(root / 'fold_metrics.csv', index=False)
    pd.DataFrame({'y_true': [0, 1, 2], 'y_pred': [0, 2, 2]}).to_csv(root / 'predictions_candidate.csv', index=False)
    (root / 'labels.json').write_text(json.dumps({'0': 'sitting', '1': 'standing', '2': 'walking'}))
    metric = 'rmse' if task == 'regression' else 'f1_macro'
    state = {'config': {'task': task, 'metric': metric}, 'best': {'trial': 1, **row},
             'trials': [{'number': 1, 'status': 'completed', 'summary': [row]}]}
    labels = []
    original = matplotlib.axes.Axes.set_title
    def capture(self, label, *args, **kwargs):
        labels.append(label)
        return original(self, label, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, 'set_title', capture)
    result = agent.visualize(tmp_path, state)
    assert 'Selected candidate: validation only' in labels
    assert ('predictions' if task == 'regression' else 'confusion matrix') in result['description']
    assert (tmp_path / 'validation.png').stat().st_size > 1000
