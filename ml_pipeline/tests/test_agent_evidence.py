import json

import numpy as np
import pandas as pd
import pytest

from ml_pipeline import agent_loop as agent
from ml_pipeline.agent_evidence import research_note, trial_evidence


def native(name, identifier, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": identifier, "type": "function",
            "function": {"name": name, "arguments": json.dumps({"reason": "Test an explicit hypothesis", **args})}}]}


def review(number, decision, model="random_forest_classifier", metric="pr_auc", overrides=None):
    return {"trial": number, "evidence_ids": [f"t{number}.{model}.none.{metric}"],
            "hypothesis": "Additional capacity may capture the interaction.",
            "alternative": "The difference may depend on this particular split.",
            "next_test": "Compare the same model with more capacity." if decision == "continue" else "",
            "distinguishes": "Whether this capacity change helps on the fixed split." if decision == "continue" else "",
            "decision": decision, "overrides_json": json.dumps(overrides or {}), "recipes_json": "[]"}


def test_structured_review_binds_evidence_and_executes_exact_next_trial(tmp_path, monkeypatch):
    rng = np.random.default_rng(17)
    values = rng.normal(size=(600, 2))
    data = pd.DataFrame({"x": values[:, 0], "z": values[:, 1],
                         "target": ((values[:, 0] > 0) != (values[:, 1] > 0)).astype(int)})
    cfg = {"task": "classification", "target": "target", "features": ["x", "z"],
           "models": ["random_forest_classifier"], "augmentations": ["none"], "preprocessing": "standard",
           "split": "holdout", "metric": "pr_auc", "seed": 42, "validation_fraction": .2,
           "model_params": {"random_forest_classifier": {"n_estimators": 1, "max_depth": 1, "n_jobs": 1}}}
    agent.create_session(tmp_path, data, cfg, max_trials=2, context="Synthetic interaction selection control")
    monkeypatch.setattr(agent, "complete_message", lambda *a, **k: native("request_model_selection", "approve", models=cfg["models"]))
    agent.advance(tmp_path, {})
    agent.approve_models(tmp_path, cfg["models"])
    responses = iter([
        native("run_benchmark", "baseline", overrides_json="{}"),
        native("inspect_trial", "inspect", trial=1),
        {"role": "assistant", "content": "An unsupported free-form conclusion must not terminate the session."},
        native("review_trial", "review_1", **review(1, "continue", overrides={"model_params": {
            "random_forest_classifier": {"n_estimators": 80, "max_depth": 6}}})),
        native("review_trial", "review_2", **review(2, "stop")),
    ])
    monkeypatch.setattr(agent, "complete_message", lambda *a, **k: next(responses))
    state = agent.advance(tmp_path, {})
    assert state["status"] == "completed"
    assert state["protocol_version"] == 2
    assert len(state["notes"]) == len(state["trials"]) == 2
    assert state["best"]["trial"] == 2
    assert state["notes"][0]["next_config"] == state["trials"][1]["config"]
    assert state["notes"][0]["next_recipes"] == state["trials"][1]["recipes"]
    assert state["notes"][1]["measured"][0]["value"] == state["trials"][1]["summary"][0]["pr_auc"]
    assert "unsupported free-form" not in state["conclusion"]
    assert "do not establish" in state["conclusion"]
    assert (tmp_path / "trial_1" / "research_note.json").exists()
    assert agent.evaluate_selected(tmp_path)["selection"]["trial"] == 2
    requests = [call["id"] for message in state["messages"] for call in message.get("tool_calls", [])]
    replies = [message["tool_call_id"] for message in state["messages"] if message["role"] == "tool"]
    assert requests == replies


def evidence_state():
    row = {"model": "random_forest_classifier", "augmentation": "none", "pr_auc": .21,
           "completed_folds": 1, "expected_folds": 1}
    trial = {"number": 1, "status": "completed", "summary": [row]}
    return {"config": {"metric": "pr_auc"}, "trials": [trial], "notes": []}


@pytest.mark.parametrize("change,match", [
    ({"evidence_ids": ["t1.invented_score"]}, "unknown evidence"),
    ({"trial": 2}, "latest trial"),
    ({"evidence_ids": []}, "nonempty"),
    ({"alternative": ""}, "alternative"),
    ({"next_test": "", "decision": "continue"}, "distinguishes"),
    ({"hypothesis": "x" * 701}, "concise"),
])
def test_research_note_rejects_missing_or_fabricated_evidence(change, match):
    with pytest.raises(ValueError, match=match):
        research_note(evidence_state(), {**review(1, "stop"), **change})


def test_measured_primary_scores_are_supplied_by_code_even_if_not_cited():
    state = evidence_state()
    args = {**review(1, "stop"), "evidence_ids": ["t1.status"]}
    note = research_note(state, args)
    assert note["measured"][0]["value"] == .21
    assert "independently of the decision threshold" in trial_evidence(state["trials"][0], "pr_auc")[-1]["description"]
