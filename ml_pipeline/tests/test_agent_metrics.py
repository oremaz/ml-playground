import json

import numpy as np
import pandas as pd
import pytest

from ml_pipeline import agent_loop as agent
from ml_pipeline.agent_metrics import score_metric, validate_metric
from ml_pipeline.benchmark_agent import read_json
from ml_pipeline.tests.test_agent_evidence import native, review


def test_operational_error_metric_uses_original_slice_and_reports_empty_rows():
    frame = pd.DataFrame({"hour": [7, 12, 18, 23], "target": [10., 30., 50., 20.]})
    cfg = {"task": "regression", "target": "target", "features": ["hour"]}
    definition = {"name": "custom_commute_mae", "kind": "mae", "where": [{"column": "hour", "op": "in", "value": [7, 18]}]}
    validate_metric(definition, frame, cfg)
    result = score_metric(definition, frame, frame.target, [12, 100, 44, 90])
    assert result["value"] == 4 and result["rows"] == 2
    empty = {**definition, "where": [{"column": "hour", "op": "eq", "value": 4}]}
    assert score_metric(empty, frame, frame.target, [12, 100, 44, 90])["value"] is None
    with pytest.raises(ValueError, match="never targets"):
        validate_metric({**definition, "where": [{"column": "target", "op": "ge", "value": 40}]}, frame, cfg)


def test_class_recall_uses_saved_semantic_label_order_and_ranking_is_binary_only():
    frame = pd.DataFrame({"x": [1, 2, 3, 4], "activity": ["walking", "sitting", "sitting", "standing"]})
    cfg = {"task": "classification", "target": "activity", "features": ["x"]}
    definition = {"name": "custom_sitting_recall", "kind": "class_recall", "class_label": "sitting"}
    validate_metric(definition, frame, cfg)
    result = score_metric(definition, frame, [2, 0, 0, 1], [2, 0, 1, 1], label_names=["sitting", "standing", "walking"])
    assert result["value"] == .5 and result["class_rows"] == 2
    with pytest.raises(ValueError, match="binary"):
        validate_metric({"name": "custom_budget_recall", "kind": "recall_at_fraction", "fraction": .1}, frame, cfg)
    for kind, expected in [("precision_at_fraction", 1.), ("recall_at_fraction", .5), ("lift_at_fraction", 2.)]:
        result = score_metric({"name": "custom_budget", "kind": kind, "fraction": .25}, frame, [0, 1, 1, 0], [0, 1, 0, 0],
                              np.array([[.9, .1], [.1, .9], [.3, .7], [.8, .2]]))
        assert result["value"] == expected


def test_diagnostic_is_backfilled_fixed_and_evaluated_after_selection(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    x = rng.normal(size=240)
    frame = pd.DataFrame({"x": x, "hour": np.tile(np.arange(24), 10), "target": 2 * x + rng.normal(size=240)})
    cfg = {"task": "regression", "target": "target", "features": ["x", "hour"], "split": "holdout",
           "metric": "rmse", "seed": 42, "validation_fraction": .2, "preprocessing": "standard",
           "models": ["random_forest_regressor"], "augmentations": ["none"],
           "model_params": {"random_forest_regressor": {"n_estimators": 5, "max_depth": 2, "n_jobs": 1}}}
    agent.create_session(tmp_path, frame, cfg, max_trials=2, context="Demand during busy hours")
    monkeypatch.setattr(agent, "complete_message", lambda *a, **k: native("request_model_selection", "approve", models=cfg["models"]))
    agent.advance(tmp_path, {})
    agent.approve_models(tmp_path, cfg["models"])
    definition = {"name": "custom_busy_mae", "kind": "mae", "where": [{"column": "hour", "op": "ge", "value": 16}]}
    responses = iter([
        native("run_benchmark", "baseline", overrides_json="{}"),
        native("evaluate_metric", "metric", definition_json=json.dumps({**definition, "name": "busy_mae"})),
        native("evaluate_metric", "repeat_metric", definition_json=json.dumps(definition)),
        native("evaluate_metric", "bad_redefinition", definition_json=json.dumps({**definition, "name": "busy_mae", "kind": "rmse"})),
        native("review_trial", "next", **review(1, "continue", model=cfg["models"][0], metric="rmse",
               overrides={"model_params": {cfg["models"][0]: {"max_depth": 4}}})),
        native("review_trial", "stop", **{**review(2, "stop", model=cfg["models"][0], metric="rmse"), "hypothesis": ""}),
    ])
    monkeypatch.setattr(agent, "complete_message", lambda *a, **k: next(responses))
    state = agent.advance(tmp_path, {})
    assert state["status"] == "completed" and state["config"]["metric"] == "rmse"
    assert state["metric_definitions"] == [definition]
    for trial in state["trials"]:
        results = trial["metric_results"][0]["metrics"]
        assert results[0]["rows"] > 0 and results[0]["value"] is not None
        assert read_json(tmp_path / f"trial_{trial['number']}" / "diagnostic_metrics.json")["definitions"] == [definition]
    assert state["notes"][1]["hypothesis"] == "" and "finding" in state["notes"][1]
    messages = (tmp_path / "messages.json").read_bytes()
    final = agent.evaluate_selected(tmp_path)
    assert final["diagnostic_metrics"][0]["name"] == definition["name"]
    assert final["diagnostic_metrics"][0]["rows"] > 0
    assert (tmp_path / "messages.json").read_bytes() == messages
    errors = [json.loads(m["content"]) for m in state["messages"] if m["role"] == "tool"]
    assert [e["definition"] for e in errors if "definition" in e] == [definition, definition]
    assert sum("error" in e for e in errors) == 1
    assert any("immutable" in e.get("error", "") for e in errors)


@pytest.mark.parametrize("name", ["sitting_recall", "custom_sitting_recall"])
def test_metric_names_are_canonical_without_changing_input(name):
    frame = pd.DataFrame({"x": [0, 1], "target": ["SITTING", "STANDING"]})
    config = {"task": "classification", "target": "target", "features": ["x"]}
    definition = {"name": name, "kind": "class_recall", "class_label": "SITTING"}
    result = validate_metric(definition, frame, config)
    assert result == {**definition, "name": "custom_sitting_recall"}
    assert definition["name"] == name
    assert validate_metric(result, frame, config) == result


def test_standard_metric_name_is_namespaced_without_changing_primary():
    frame = pd.DataFrame({"x": [0, 1], "target": [0, 1]})
    config = {"task": "classification", "target": "target", "features": ["x"], "metric": "accuracy"}
    result = validate_metric({"name": "accuracy", "kind": "accuracy"}, frame, config)
    assert result == {"name": "custom_accuracy", "kind": "accuracy"}
    assert config["metric"] == "accuracy"


@pytest.mark.parametrize("name", [None, 12, "", "custom_", "123recall", "has spaces", "MixedCase", "a" * 42])
def test_invalid_metric_names_remain_rejected(name):
    frame = pd.DataFrame({"x": [0, 1], "target": [0, 1]})
    config = {"task": "classification", "target": "target", "features": ["x"]}
    with pytest.raises(ValueError, match="metric name"):
        validate_metric({"name": name, "kind": "accuracy"}, frame, config)
