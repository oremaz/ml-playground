"""Logged CSV-to-agent experiments on energy, bike demand and wearable activity."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml_pipeline.agent_evidence import trial_evidence
from ml_pipeline.agent_loop import (SYSTEM, advance, approve_models, create_session, evaluate_selected,
                                   prepare_reviewed_dataset, save_state, visualize)
from ml_pipeline.benchmark_agent import create_workspace, generate_script, propose, read_json, write_json


CONTEXTS = {
    "energy": """I have a smart-house sensor CSV and want a virtual meter: estimate appliance electricity
consumption in Wh for each ten-minute interval when the energy meter is unavailable. Sensors from
that same interval are available; this is estimation, not a future-weather forecast. Only room/building
temperature and humidity sensors (T1..T9, RH_1..RH_9) and calendar information are available inputs.
The lights meter and airport weather are unavailable inputs. Author metadata says rv1 and rv2 are
random noise controls. Future intervals should be held out chronologically. I care about errors and
systematic underestimation during evening hours 17..22, in addition to overall RMSE. This is a single
house: do not claim transfer to new buildings. Propose a suitable target, preparation and CPU models.""",
    "bikes": """I operate a bike-sharing service. At the START of an hour, estimate the total rental
count during that hour. Calendar information is known; weather observations are available only from
the PREVIOUS hour. The CSV contains the hour's realized weather, which must not be used directly.
Construct timestamps from dteday and hr, and join previous-hour weather at exactly timestamp minus
one hour with validated cardinality. Keep absent histories as NaN for train-fitted imputation. Current
hour cnt, casual and registered are unavailable inputs; casual+registered equals the target, not a
useful predictive feature. instant is a record ID. Compare on chronologically later hours. Each rental
count becomes observable at timestamp plus one hour: provide a label_end_column for purging. I care
especially about commuter hours 7..9 and 16..19 on working days. Use overall RMSE for selection.
Propose preparation and CPU models; do not use target lags or any full-data learned transform.""",
    "activity": """These are smartphone motion-sensor features from people walking, taking stairs,
sitting, standing or lying down. Predict the activity for a person not seen during training. subject
identifies people, not an input feature; keep subjects disjoint across training, validation and final
evaluation. Use macro F1 as the primary metric, with accuracy and class-specific errors as diagnostics.
I care about confusing sitting with standing. There are 561 provided sensor features, already engineered
by the dataset authors. Do not truncate to the first columns just because detailed profiles are bounded.
Use features=['*'] with explicit exclusions if retaining all sensors; you may choose a justified subset.
The supplied CSV combines the original train and test files; use the app's own reproducible group split
and do not claim comparability with the published split or new preprocessing of raw sensor signals.
Propose preprocessing and CPU model families from the registry.""",
}


def settings():
    return {"provider": "local", "model": "Qwen/Qwen3.8-27B", "device": "cuda", "quantization": "none", "reasoning": "xhigh"}


def snapshot(root, data):
    package = Path(__file__).resolve().parent
    sources = {}
    for name in ["agent_domain_experiment.py", "agent_loop.py", "agent_evidence.py", "agent_metrics.py",
                 "benchmark_agent.py", "playground_runtime.py", "playground_app.py", "utils/local_qwen.py",
                 "utils/llm_provider.py", "utils/data_split.py", "pipelines_torch/models.py", "pipelines_torch/base.py"]:
        content = (package / name).read_bytes()
        path = root / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        sources[name] = hashlib.sha256(content).hexdigest()
    write_json(root / "provenance.json", {"created_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"), "sources": sources, "datasets": read_json(data / "manifest.json"),
        "dependencies": {name: version(name) for name in ["torch", "transformers", "numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "streamlit"]},
        "llm": settings(), "local_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"})


def replay_review(root, previous):
    folder = root / "prompt_check"
    old = read_json(previous / "state.json")
    state = deepcopy(old)
    state.update(status="running", notes=[], pending=[], llm_steps=0, protocol_version=2)
    evidence = [{"trial": trial["number"], "config": trial["config"], "evidence": trial_evidence(trial, old["config"]["metric"])}
                for trial in old["trials"]]
    state["messages"] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps({
        "objective": "Review the completed finance trials using only the supplied measurements.",
        "reviewed_config": old["config"], "max_trials": old["max_trials"], "trials": evidence,
        "best": old["best"], "trials_remaining": 0})}]
    folder.mkdir(parents=True, exist_ok=True)
    for trial in old["trials"]:
        target = folder / f"trial_{trial['number']}"
        target.mkdir(exist_ok=True)
        for name in ["output", "input.joblib", "diagnostics.json"]:
            source = previous / target.name / name
            if source.exists() and not (target / name).exists():
                (target / name).symlink_to(source)
    (folder / "development.joblib").symlink_to(previous / "development.joblib")
    save_state(folder, state)
    state = advance(folder, settings())
    assert state["status"] == "completed" and len(state["trials"]) == len(old["trials"])
    write_json(folder / "replay.json", {"source": str(previous), "new_fitting": False,
        "previous_note": old["notes"][-1], "new_note": state["notes"][-1],
        "purpose": "Prompt replay on the same measured finance facts; not a new predictive experiment or blinded reasoning evaluation."})
    print("PROMPT_REPLAY", json.dumps(state["notes"][-1]), flush=True)


def reviewed_config(domain, plan):
    task = "classification" if domain == "activity" else "regression"
    target = {"energy": "Appliances", "bikes": "cnt", "activity": "activity"}[domain]
    if plan["task"] != task or plan["target"] != target:
        raise ValueError(f"Proposal needs review: expected {task} target {target}, got {plan['task']} / {plan['target']}")
    split = "group" if domain == "activity" else "temporal"
    if plan.get("split") != split or not plan.get("split_column"):
        raise ValueError(f"Proposal needs the requested {split} validation")
    allowed = [f"{family}_{'classifier' if task == 'classification' else 'regressor'}" for family in ["random_forest", "lightgbm", "xgboost"]]
    if set(plan["models"]) - set(allowed):
        raise ValueError("Proposal exceeded the explicitly available CPU model families")
    forbidden = {"energy": {"lights", "rv1", "rv2", "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"},
                 "bikes": {"instant", "casual", "registered", "weathersit", "temp", "atemp", "hum", "windspeed"},
                 "activity": {"subject"}}[domain]
    if set(plan["features"]) & forbidden:
        raise ValueError(f"Unavailable or prohibited model inputs: {set(plan['features']) & forbidden}")
    if domain == "bikes" and not plan.get("label_end_column"):
        raise ValueError("The bike task requires label observation-end times")
    return {**plan, "metric": "f1_macro" if domain == "activity" else "rmse", "preprocessing": "standard",
            "seed": 42, "validation_fraction": .2, "device": "cpu", "augmentations": ["none"],
            "model_params": {name: {"n_estimators": 100, "n_jobs": 4} for name in allowed}}


def check_preparation(domain, frame, raw, config):
    if len(frame) != len(raw):
        raise ValueError("The preparation unexpectedly changed row counts; review before fitting")
    if domain == "activity":
        np.testing.assert_array_equal(frame.activity, raw.activity)
        np.testing.assert_array_equal(frame.subject, raw.subject)
        return
    column = config["split_column"]
    timestamps = pd.to_datetime(frame[column], utc=True)
    expected = pd.to_datetime(raw.date if domain == "energy" else raw.dteday, utc=True)
    if domain == "bikes":
        expected += pd.to_timedelta(raw.hr, unit="h")
        ends = pd.to_datetime(frame[config["label_end_column"]], utc=True)
        np.testing.assert_array_equal(ends, expected + pd.Timedelta(hours=1))
    np.testing.assert_array_equal(timestamps, expected)
    np.testing.assert_array_equal(frame[config["target"]], raw[config["target"]])


def check_feature_independence(domain, script, raw, frame, config):
    import runpy
    build = runpy.run_path(str(script), run_name="reviewed_preparation_audit")["build_dataset"]
    changed = raw.copy()
    changed[config["target"]] = changed[config["target"]].iloc[::-1].to_numpy()
    altered = build({config["table"]: changed}, config)
    pd.testing.assert_frame_equal(frame[config["features"]], altered[config["features"]])
    if domain in ("energy", "bikes"):
        cutoff = len(raw) // 2
        changed = raw.copy()
        columns = ([f"T{i}" for i in range(1, 10)] + [f"RH_{i}" for i in range(1, 10)]) if domain == "energy" else ["weathersit", "temp", "atemp", "hum", "windspeed"]
        first = cutoff + 1 if domain == "energy" else cutoff
        changed.loc[first:, columns] = changed.loc[first:, columns] * 10 + 123
        altered = build({config["table"]: changed}, config)
        pd.testing.assert_frame_equal(frame.iloc[:cutoff + 1][config["features"]], altered.iloc[:cutoff + 1][config["features"]])


def experiment(root, data, domain):
    folder = root / domain
    folder.mkdir(exist_ok=True)
    csv = data / f"{domain}.csv"
    manifest = read_json(data / "manifest.json")[domain]
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == manifest["sha256"]
    context = CONTEXTS[domain] + "\nAvailable fit resources: CPU Random Forest, LightGBM and XGBoost only; optional pretrained packages are outside this experiment. Use standard preprocessing initially. Registry defaults are the baseline, with 100 trees and at most 4 threads per method. The agent has at most 5 trials and can stop early. Additional task-specific diagnostics are available when useful."
    workspace = create_workspace(folder / "memory", [(csv.name, csv.read_bytes())])
    write_json(folder / "workspace.json", {"path": str(workspace), "context": context})
    print("PROPOSE", domain, flush=True)
    plan = propose(workspace, settings(), context)
    config = reviewed_config(domain, plan)
    write_json(folder / "reviewed_config.json", config)
    write_json(folder / "review.json", {"simulated_reviewer": True,
        "changes": "Fixed requested primary metric, standard preprocessing, seed 42, .2 validation fraction and 100-tree/4-thread CPU resource baseline. Model families, selected features, preparation and split column come from the LLM proposal."})
    print("SCRIPT", domain, "features", len(config["features"]), "models", config["models"], flush=True)
    script, _ = generate_script(workspace, settings(), config)
    import ast
    if any(isinstance(node, ast.Call) and getattr(node.func, "attr", "").startswith("fit") for node in ast.walk(ast.parse(script.read_text()))):
        raise ValueError("Review required: learned fitting in the preparation script")
    agent_folder = prepare_reviewed_dataset(workspace, script.read_text(), config, timeout=900)
    prepared = joblib.load(agent_folder / "prepared.joblib")
    raw = pd.read_csv(csv)
    check_preparation(domain, prepared, raw, config)
    check_feature_independence(domain, script, raw, prepared, config)
    write_json(folder / "preparation_checks.json", {"row_counts_and_targets_preserved": True,
        "features_independent_of_target_values": True, "future_sensor_mutation_cannot_change_past_inputs": domain != "activity"})
    write_json(folder / "agent.json", {"path": str(agent_folder)})
    state = create_session(agent_folder, prepared, config, max_trials=5, context=context, timeout=900)
    def progress(state):
        print("AGENT", domain, state["status"], "step", state["llm_steps"], "trials", len(state["trials"]), flush=True)
    state = advance(agent_folder, settings(), progress=progress)
    assert state["status"] == "awaiting_models"
    approved = state["proposed_models"]
    if set(approved) - set(config["model_params"]):
        raise ValueError("Agent requested a family outside the simulated user's available CPU methods")
    approve_models(agent_folder, approved, source="simulated_user")
    state = advance(agent_folder, settings(), progress=progress)
    assert state["status"] == "completed" and len(state["notes"]) == len(state["trials"])
    visualize(agent_folder, state)
    result = evaluate_selected(agent_folder)
    rows = [{"trial": trial["number"], **row} for trial in state["trials"] for row in trial.get("summary", [])]
    pd.DataFrame(rows).to_csv(folder / "trial_results.csv", index=False)
    write_json(folder / "result.json", {"domain": domain, "approved_models": approved, "trials": len(state["trials"]),
        "selected": state["best"], "final_evaluation": result, "metric_definitions": state.get("metric_definitions", [])})
    print("COMPLETE", domain, json.dumps(state["best"]), flush=True)


def main(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Use a GPU allocation for the local 27B LLM")
    torch.set_num_threads(4)
    root, data = args.output.resolve(), args.data.resolve()
    root.mkdir(parents=True, exist_ok=True)
    snapshot(root, data)
    if args.review_source:
        replay_review(root, args.review_source.resolve())
    outcomes = []
    for domain in args.domains:
        try:
            experiment(root, data, domain)
            outcomes.append({"domain": domain, "status": "completed"})
        except Exception as exc:
            import traceback
            error = {"domain": domain, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            outcomes.append(error)
            print(json.dumps(error), flush=True)
        write_json(root / "status.json", outcomes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-source", type=Path)
    parser.add_argument("--domains", nargs="+", choices=list(CONTEXTS), default=list(CONTEXTS))
    main(parser.parse_args())
