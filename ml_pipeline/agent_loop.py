"""A persisted, bounded tool loop over a reviewed supervised experiment."""

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from ml_pipeline.agent_protocol import (
    _tool,
    tool_catalog,
    LEGACY_SYSTEM,
    SYSTEM,
)
from ml_pipeline.agent_trial_config import (
    model_capabilities,
    apply_features,
    _trial_config,
    _best,
    _experiment_settings,
    _setting_changes,
    _execution_feedback,
    _trial_comparison,
)
from ml_pipeline.agent_trial_views import (
    _trial_diagnostics,
    visualize,
)

import joblib
import numpy as np
import pandas as pd

from ml_pipeline.benchmark_agent import (complete_message, execution_environment, profile_tables,
                                        read_json, tabular_models, write_json)
from ml_pipeline.playground_runtime import benchmark, prepare_dataset, score_predictions, split_indices
from ml_pipeline.utils.playground_trace import append_event, operation
from ml_pipeline.agent_evidence import trial_evidence, research_note, measured_conclusion
from ml_pipeline.agent_metrics import validate_metric, evaluate_saved_metrics, score_metric


def _records(frame):
    return json.loads(frame.to_json(orient="records"))


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _compact_result(value):
    if isinstance(value, dict):
        return {key: _compact_result(item) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) > 80 and all(isinstance(item, str) for item in value):
            return {"count": len(value), "sample": value[:8], "sha256": _fingerprint(value),
                    "full_list": "Saved trial config/research_note; original columns are in the reviewed configuration."}
        return [_compact_result(item) for item in value]
    return value


def _subprocess(command, cwd, log_path, timeout):
    with Path(log_path).open("w") as log:
        process = subprocess.Popen(command, cwd=cwd, env=execution_environment(), stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f"Worker exited {code}: {Path(log_path).read_text()[-3000:]}")


def prepare_reviewed_dataset(workspace, source, config, timeout=1800):
    """Execute the approved preparation once, outside the native tool dispatcher."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and str(getattr(node.func, "attr", "")).startswith("fit"):
            raise ValueError("Learned fitting is forbidden in deterministic preparation; use train-fitted preprocessing")
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "run_experiment":
            if len(node.args) > 1 or any(item.arg == "preprocessor_factory" for item in node.keywords):
                raise ValueError("Iterative mode supports standard/skrub preprocessing. Run reviewed custom preprocessing with Run benchmark.")
    workspace = Path(workspace).resolve()
    folder = workspace / "agents" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    (folder / "preparation.py").write_text(source)
    write_json(folder / "preparation_config.json", config)
    with operation(workspace, "agent.prepare_reviewed", {"script_sha256": hashlib.sha256(source.encode()).hexdigest(),
                                                       "session": str(folder.relative_to(workspace))}):
        _subprocess([sys.executable, "-m", "ml_pipeline.agent_loop", "prepare", str(folder), str(workspace)],
                    workspace, folder / "preparation.log", timeout)
    return folder


def create_session(folder, frame, config, *, max_trials, context, test_fraction=0.2, timeout=1800, structured_notes=True):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "state.json").exists():
        raise ValueError("Session already exists; resume it instead")
    if isinstance(max_trials, bool) or max_trials not in range(1, 6):
        raise ValueError("Trial budget must be an integer from 1 to 5")
    if config.get("split") == "kfold" and config.get("validation_protocol") != "robust":
        raise ValueError("Iterative search currently requires a fixed holdout, group or temporal split")
    if not 0.05 <= test_fraction <= 0.4:
        raise ValueError("Final evaluation fraction must be between 0.05 and 0.4")
    config = deepcopy(config)
    frame = frame.reset_index(drop=True)
    _, y, _ = prepare_dataset(frame, config)
    outer_config = {**config, "validation_fraction": test_fraction}
    if config.get("split") == "kfold":
        outer_config["split"] = "holdout"
    development, test = split_indices(frame, y, outer_config)[0]
    dev = frame.iloc[development].reset_index(drop=True)
    _, dev_y, _ = prepare_dataset(dev, config)
    if config.get("validation_protocol") == "robust" and config["task"] != "forecasting":
        from ml_pipeline.agent_validation import development_splits
        folds = development_splits(dev, dev_y, config)
        path = folder / "validation_splits.json"
        write_json(path, [{"train": a.tolist(), "validation": b.tolist()} for a, b in folds])
        config.update(_split_manifest=str(path), _split_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    train, validation = split_indices(dev, dev_y, config)[-1]
    joblib.dump(dev, folder / "development.joblib")
    joblib.dump(frame.iloc[test].reset_index(drop=True), folder / "evaluation.joblib")
    write_json(folder / "partitions.json", {"development": development.tolist(), "test": test.tolist(),
               "train_in_development": train.tolist(), "validation_in_development": validation.tolist(),
               "source_sha256": hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()})
    public = {"objective": context, "reviewed_config": config, "max_trials": max_trials,
              "protocol": "Frozen development folds; mean primary metric selects the winner. Final evaluation is inaccessible to tools.",
              "development_folds": [{"train_rows": len(a), "validation_rows": len(b)} for a, b in split_indices(dev, dev_y, config)],
              "development_rows": len(dev), "train_rows": len(train), "validation_rows": len(validation),
              "available_models": tabular_models(config["task"])}
    state = {"status": "running", "protocol_version": 2 if structured_notes else 1, "notes": [],
             "config": deepcopy(config), "max_trials": max_trials,
             "timeout": timeout, "trials": [], "recipes": [], "approved_models": None,
             "approval_requested": False, "pending": [], "messages": [
                 {"role": "system", "content": SYSTEM if structured_notes else LEGACY_SYSTEM},
                 {"role": "user", "content": json.dumps(public, ensure_ascii=False)}],
             "llm_steps": 0, "best": None}
    save_state(folder, state)
    append_event(folder, "agent.created", data=public)
    return state


def save_state(folder, state):
    write_json(Path(folder) / "state.json", state)
    write_json(Path(folder) / "messages.json", state["messages"])
    parts = ["# Agent conversation", "Observable messages and native tool results. Hidden model reasoning is not recorded."]
    for index, message in enumerate(state["messages"], 1):
        parts.append(f"## {index}. {message['role']}" + (f" ({message['name']})" if message.get("name") else ""))
        if message.get("content"):
            parts.append(message["content"])
        if message.get("tool_calls"):
            parts.append("```json\n" + json.dumps(message["tool_calls"], indent=2) + "\n```")
        if message.get("tool_call_id"):
            parts.append(f"Tool call ID: {message['tool_call_id']}")
    (Path(folder) / "conversation.md").write_text("\n\n".join(parts) + "\n")


def approve_models(folder, models, *, source="user"):
    state = read_json(Path(folder) / "state.json")
    if state["status"] != "awaiting_models" or state["approved_models"] is not None:
        raise ValueError("No model-selection request is pending")
    if not models or len(set(models)) != len(models) or set(models) - set(tabular_models(state["config"]["task"])):
        raise ValueError("Choose unique registered tabular models")
    state["approved_models"] = models
    state["status"] = "running"
    call = state["pending"].pop(0)
    result = {"approved_models": models, "input_source": source,
              "capabilities": model_capabilities(state["config"], models),
              "instruction": "Use these model families; first trial keeps reviewed defaults for their parameters."}
    _tool_reply(folder, state, call, result)
    append_event(folder, "agent.model_selection", data=result)
    save_state(folder, state)
    return state


def _tool_reply(folder, state, call, result):
    state["messages"].append({"role": "tool", "tool_call_id": call["id"],
                              "content": json.dumps(result, allow_nan=False)})
    append_event(folder, "agent.tool_result", operation=call["function"]["name"],
                 operation_id=call["id"], data=result)


def _benchmark_tool(folder, state, args, *, reviewed=False):
    if state.get("protocol_version", 1) >= 2 and state["trials"] and not reviewed:
        raise ValueError("Use review_trial to cite evidence and execute the next discriminating experiment")
    if len(state["trials"]) >= state["max_trials"]:
        raise ValueError("Trial budget exhausted; inspect, visualize or conclude using actual results")
    incumbent = deepcopy(state["best"])
    config = _trial_config(state, json.loads(args["overrides_json"]))
    fingerprint = _fingerprint({"config": config, "recipes": state["recipes"]})
    if any(t["fingerprint"] == fingerprint for t in state["trials"]):
        raise ValueError("Identical configuration already tried; inspect its result instead")
    number = len(state["trials"]) + 1
    trial_folder = folder / f"trial_{number}"
    trial_folder.mkdir()
    dev = apply_features(joblib.load(folder / "development.joblib"), state["config"]["features"], state["recipes"])
    joblib.dump(dev, trial_folder / "input.joblib")
    write_json(trial_folder / "config.json", config)
    write_json(trial_folder / "features.json", state["recipes"])
    script = "from pathlib import Path\nimport joblib\nfrom ml_pipeline.benchmark_agent import read_json\nfrom ml_pipeline.playground_runtime import benchmark\nroot = Path(__file__).resolve().parent\nbenchmark(joblib.load(root / 'input.joblib'), read_json(root / 'config.json'), root / 'output', trace_workspace=root.parent)\n"
    (trial_folder / "experiment.py").write_text(script)
    trial = {"number": number, "reason": args["reason"], "fingerprint": fingerprint,
             "config": config, "recipes": deepcopy(state["recipes"]), "status": "running"}
    state["trials"].append(trial)
    save_state(folder, state)
    started = time.monotonic()
    try:
        _subprocess([sys.executable, "-u", str(trial_folder / "experiment.py")], folder,
                    trial_folder / "execution.log", state["timeout"])
        output = trial_folder / "output"
        trial.update(status="completed", summary=_records(pd.read_csv(output / "summary.csv")),
                     dataset_summary=read_json(output / "dataset_summary.json"),
                     errors=read_json(output / "errors.json") if (output / "errors.json").exists() else [])
        if config["task"] == "forecasting":
            state.setdefault("forecast_revisions", {}).update({name: descriptor["resolved_revision"] for name, descriptor
                in read_json(output / "forecast_models.json").items() if descriptor.get("resolved_revision")})
    except Exception as exc:
        trial.update(status="failed", error=f"{type(exc).__name__}: {exc}", summary=[])
    trial["elapsed_seconds"] = time.monotonic() - started
    state["best"] = _best(state)
    trial["execution"] = _execution_feedback(folder, state, trial, incumbent)
    trial["comparison"] = _trial_comparison(state, trial, incumbent)
    evaluate_saved_metrics(folder, state, trial)
    trial["evidence"] = trial_evidence(trial, state["config"]["metric"])
    write_json(trial_folder / "execution_feedback.json", {"execution": trial["execution"], "comparison": trial["comparison"]})
    save_state(folder, state)
    return _compact_result({**{k: v for k, v in trial.items() if k not in ("fingerprint", "config", "recipes")},
            "incumbent": state["best"], "trials_remaining": state["max_trials"] - number,
            "next": ("Use review_trial with evidence_ids, an untested hypothesis and alternative, and the exact next test or stop. "
                     if state.get("protocol_version", 1) >= 2 else "") + "Explain the actual comparison and executed changes, not intended changes. "
                    "Use inspect_trial to check train/validation gaps before claiming underfit or overfit. "
                    "A negative result only describes this tested configuration on this split; it does not support a feature ceiling or identify a cause. "
                    "For continue ask a prospective question about one comparison; for stop leave hypothesis empty and state unresolved uncertainty."})


def dispatch(folder, state, call):
    name = call["function"]["name"]
    args = json.loads(call["function"]["arguments"])
    schema = next((x["function"]["parameters"] for x in tool_catalog() if x["function"]["name"] == name), None)
    if schema is None or not isinstance(args, dict) or set(args) - set(schema["properties"]) or set(schema["required"]) - set(args):
        raise ValueError("Unknown tool or invalid tool arguments")
    if not isinstance(args["reason"], str) or not args["reason"].strip():
        raise ValueError("Give a short public purpose for this tool")
    append_event(folder, "agent.tool_request", operation=name, operation_id=call["id"], data=args)
    if name == "request_model_selection":
        if state["approval_requested"]:
            raise ValueError("Model selection has already been requested; use the approved families")
        if not args["models"] or set(args["models"]) - set(tabular_models(state["config"]["task"])):
            raise ValueError("Propose registered model families")
        state.update(approval_requested=True, status="awaiting_models", proposed_models=args["models"],
                     approval_reason=args["reason"])
        return None
    if name == "inspect_dataset":
        dev = joblib.load(folder / "development.joblib")
        config = state["config"]
        _, y, _ = prepare_dataset(dev, config)
        train, validation = split_indices(dev, y, config)[-1]
        numeric = dev.iloc[train][config["features"]].select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
        multiclass = config["task"] == "classification" and len(np.unique(y)) > 2
        associations = {} if multiclass else numeric.corrwith(pd.Series(y[train], index=numeric.index)).dropna().sort_values().to_dict()
        columns = [config["target"]] + config["features"]
        if config["task"] == "forecasting":
            columns += [config["split_column"]] + ([config["series_column"]] if config.get("series_column") else [])
        return {"profiles": profile_tables({"development": dev[columns]}, sample_rows=2),
                "train_rows": len(train), "validation_rows": len(validation),
                "train_class_counts": {str(key): int(value) for key, value in dev.iloc[train][config["target"]].value_counts().items()}
                    if config["task"] == "classification" else None,
                "train_target_correlations": associations,
                "validation_folds": [{"train_rows": len(a), "validation_rows": len(b)} for a,b in split_indices(dev,y,config)],
                "warning": ("Correlations with arbitrary multiclass label codes are omitted. " if multiclass else "") +
                    "Associations are training-only; missing values are imputed during training, never globally."}
    if state["approved_models"] is None:
        raise ValueError("Request user model selection before preparing or benchmarking")
    if name == "prepare_features":
        recipes = json.loads(args["recipes_json"])
        dev = joblib.load(folder / "development.joblib")
        apply_features(dev, state["config"]["features"], recipes)
        if not state["trials"] and recipes:
            raise ValueError("Benchmark the reviewed baseline before changing features")
        state["recipes"] = recipes
        return {"prepared_features": recipes, "applies_to": "Subsequent trials only; previous trials unchanged."}
    if name == "run_benchmark":
        return _benchmark_tool(folder, state, args)
    if name == "evaluate_metric":
        definition = validate_metric(json.loads(args["definition_json"]), joblib.load(folder / "development.joblib"), state["config"])
        definitions = state.setdefault("metric_definitions", [])
        prior = next((item for item in definitions if item["name"] == definition["name"]), None)
        if prior is not None and prior != definition:
            raise ValueError("Metric definitions are immutable; use a new name for a different diagnostic")
        if prior is None:
            if len(definitions) >= 6:
                raise ValueError("At most six additional diagnostic metrics per search")
            definitions.append(definition)
        results = []
        for trial in state["trials"]:
            values = evaluate_saved_metrics(folder, state, trial)
            results.append({"trial": trial["number"], "results": values,
                            "evidence": trial_evidence(trial, state["config"]["metric"])})
        write_json(folder / "metric_definitions.json", definitions)
        return {"definition": definition, "validation_only": True, "by_trial": results,
                "caution": "Exploratory diagnostics, not a new selection metric. Counts and definitions are shared across all compared models."}
    if name == "review_trial":
        return _review_trial(folder, state, args)
    if name == "inspect_trial":
        result = _trial_diagnostics(folder, state, args["trial"])
        number = args["trial"] or len(state["trials"])
        state["trials"][number - 1]["diagnostics"] = result
        evidence = trial_evidence(state["trials"][number - 1], state["config"]["metric"])
        return {**result, "evidence": evidence}
    if name == "visualize_results":
        return visualize(folder, state)
    raise ValueError("Unknown tool")


def _review_trial(folder, state, args):
    note = research_note(state, args)
    overrides, recipes = json.loads(args["overrides_json"]), json.loads(args["recipes_json"])
    if args["decision"] == "continue":
        if len(state["trials"]) >= state["max_trials"]:
            raise ValueError("Trial budget exhausted; record the latest evidence with decision='stop'")
        proposed = {**state, "recipes": recipes}
        apply_features(joblib.load(folder / "development.joblib"), state["config"]["features"], recipes)
        config = _trial_config(proposed, overrides)
        fingerprint = _fingerprint({"config": config, "recipes": recipes})
        if any(trial["fingerprint"] == fingerprint for trial in state["trials"]):
            raise ValueError("Identical configuration already tried; propose an informative comparison or stop")
        incumbent = state.get("best")
        reference = state["trials"][incumbent["trial"] - 1] if incumbent else state["trials"][-1]
        note["next_changes"] = _setting_changes(_experiment_settings(reference["config"], reference["recipes"]),
                                                _experiment_settings(config, recipes))
        note["next_config"] = config
        note["next_recipes"] = recipes
    elif overrides != {} or recipes != []:
        raise ValueError("A stop note must have overrides_json='{}', recipes_json='[]'")
    state.setdefault("notes", []).append(note)
    write_json(folder / f"trial_{note['trial']}" / "research_note.json", note)
    append_event(folder, "agent.research_note", data=note)
    save_state(folder, state)
    if args["decision"] == "continue":
        state["recipes"] = recipes
        result = _benchmark_tool(folder, state, {"reason": args["next_test"], "overrides_json": args["overrides_json"]}, reviewed=True)
        return _compact_result({"research_note": note, "benchmark": result})
    state["status"] = "completed" if state.get("best") else "stopped_without_result"
    state["conclusion"] = measured_conclusion(state)
    append_event(folder, "agent.finished", data={"best": state["best"], "trials": len(state["trials"]), "status": state["status"]})
    return {"research_note": note, "measured_summary": state["conclusion"], "status": state["status"]}


def restore_interrupted_trial(folder):
    folder = Path(folder)
    state = read_json(folder / "state.json")
    if not state["trials"] or state["trials"][-1]["status"] != "running":
        return False
    trial = state["trials"][-1]
    output = folder / f"trial_{trial['number']}" / "output"
    incumbent = deepcopy(state.get("best"))
    marker = output / "completed.json"
    if marker.exists() and hashlib.sha256((output / "summary.csv").read_bytes()).hexdigest() == read_json(marker)["summary_sha256"]:
        trial.update(status="completed", summary=_records(pd.read_csv(output / "summary.csv")),
                     dataset_summary=read_json(output / "dataset_summary.json"), errors=read_json(output / "errors.json"))
    else:
        trial.update(status="failed", summary=[], error="Worker interrupted before a complete comparison was committed; this trial consumes budget.")
    state["best"] = _best(state)
    trial["execution"] = _execution_feedback(folder, state, trial, incumbent)
    trial["comparison"] = _trial_comparison(state, trial, incumbent)
    evaluate_saved_metrics(folder, state, trial)
    trial["evidence"] = trial_evidence(trial, state["config"]["metric"])
    if state["pending"]:
        call = state["pending"].pop(0)
        _tool_reply(folder, state, call, _compact_result({key: value for key, value in trial.items()
                    if key not in ("config", "recipes", "fingerprint")}))
    append_event(folder, "agent.interrupted_trial_recovered", data={"trial": trial["number"], "status": trial["status"]})
    save_state(folder, state)
    return True


def advance(folder, settings, *, progress=None):
    """Advance until the single approval pause or a final assistant answer."""
    folder = Path(folder).resolve()
    state = read_json(folder / "state.json")
    if state["status"] != "running":
        return state
    limit = 8 + state["max_trials"] * 5
    limit = state.get("message_limit", limit)
    try:
        while state["status"] == "running":
            if state["pending"]:
                call = state["pending"][0]
                try:
                    with operation(folder, "agent.tool", {"call": call}) as trace:
                        result = dispatch(folder, state, call)
                        trace["result"] = result
                except Exception as exc:
                    result = {"error": f"{type(exc).__name__}: {exc}", "instruction": "Use the saved result or correct this request; do not claim success."}
                if state["status"] == "awaiting_models":
                    save_state(folder, state)
                    break
                state["pending"].pop(0)
                _tool_reply(folder, state, call, result)
                if state["status"] != "running":
                    for pending in state["pending"]:
                        _tool_reply(folder, state, pending, {"error": "Session ended; this additional call was not executed"})
                    state["pending"] = []
                save_state(folder, state)
                if progress:
                    progress(state)
                continue
            if state["llm_steps"] >= limit:
                state["messages"].append({"role": "user", "content": "The bounded conversation budget is exhausted. Explain the actual results and limitations now. No further tools are available."})
                message = complete_message(settings, state["messages"], [], workspace=folder)
            else:
                message = complete_message(settings, state["messages"], tool_catalog(), workspace=folder)
            state["llm_steps"] += 1
            state["messages"].append(message)
            state["pending"] = message.get("tool_calls", []).copy()
            if not state["pending"]:
                if state.get("protocol_version", 1) >= 2 and state["llm_steps"] <= limit:
                    state["messages"].append({"role": "user", "content": "Use the native tools. After a trial, call review_trial to cite measured evidence and either execute the next test or stop. Free-form conclusions are not accepted as measured results."})
                    save_state(folder, state)
                    continue
                state["status"] = "completed" if state["best"] else "stopped_without_result"
                state["conclusion"] = measured_conclusion(state) if state.get("protocol_version", 1) >= 2 else message["content"]
                append_event(folder, "agent.finished", data={"best": state["best"], "trials": len(state["trials"]),
                                                            "status": state["status"]})
            if state["llm_steps"] > limit and state["pending"]:
                for call in state["pending"]:
                    _tool_reply(folder, state, call, {"error": "Conversation budget exhausted"})
                state["pending"] = []
                state["status"] = "stopped_at_message_limit"
            save_state(folder, state)
            if progress:
                progress(state)
    except Exception:
        save_state(folder, state)
        raise
    return state


def evaluate_selected(folder):
    """Unlock final scores only after the conversation and selection have ended."""
    folder = Path(folder).resolve()
    if (folder / "evaluation.json").exists():
        return read_json(folder / "evaluation.json")
    state = read_json(folder / "state.json")
    if state["status"] != "completed" or not state["best"]:
        raise ValueError("Complete the agent and lock validation selection before final evaluation")
    best = state["best"]
    trial = state["trials"][best["trial"] - 1]
    root = folder / f"trial_{best['trial']}" / "output"
    if state["config"]["task"] == "forecasting":
        from ml_pipeline.forecasting_runtime import benchmark_forecasting
        config = deepcopy(trial["config"])
        config["models"] = [best["model"]]
        descriptor = read_json(root / "forecast_models.json")[best["model"]]
        if descriptor.get("resolved_revision"):
            config.setdefault("model_params", {}).setdefault(best["model"], {})["revision"] = descriptor["resolved_revision"]
        dev, test = joblib.load(folder / "development.joblib"), joblib.load(folder / "evaluation.joblib")
        combined = pd.concat([dev, test], ignore_index=True)
        output = folder / "final_forecast"
        if output.exists():
            raise ValueError("An interrupted final forecast exists; inspect it before explicitly retrying")
        summary = benchmark_forecasting(combined, config, output, evaluation_rows=np.arange(len(dev), len(combined)), trace_workspace=folder)
        stats = read_json(output / "dataset_summary.json")
        scores = {key: float(summary.iloc[0][key]) for key in ("mae", "rmse", "smape")}
        result = {"selection": best, "test_rows": len(test), "scores": scores,
                  "evaluated_observations": stats["evaluated_observations"], "forecast_windows": stats["forecast_windows"],
                  "training_protocol": "Locked zero-shot checkpoint; rolling observed history, no final-driven fitting or selection."}
        write_json(folder / "evaluation.json", result)
        append_event(folder, "agent.final_evaluation", data=result)
        return result
    rows = pd.read_csv(root / "fold_metrics.csv")
    row = rows[(rows.model == best["model"]) & (rows.augmentation == best["augmentation"])].sort_values("fold").iloc[-1]
    checkpoint = root / f"{row.checkpoint_id}.pkl"
    if not checkpoint.exists():
        raise ValueError("Final evaluation currently supports saved sklearn-compatible models; validation artifacts remain available")
    test = apply_features(joblib.load(folder / "evaluation.joblib"), state["config"]["features"], trial["recipes"])
    config = trial["config"]
    X, y, labels = prepare_dataset(test, config)
    if labels is not None:
        encoding = read_json(root / "labels.json")
        inverse = {v: int(k) for k, v in encoding.items()}
        y = np.array([inverse[str(value)] for value in test[config["target"]]])
    transformer = joblib.load(root / f"preprocessor_fold_{int(row.fold)}.joblib")
    model = joblib.load(checkpoint)
    transformed = np.asarray(transformer.transform(X), dtype=np.float32)
    probabilities = model.predict_proba(transformed) if config["task"] == "classification" else None
    scores = score_predictions(y, model.predict(transformed), probabilities, config["task"])
    result = {"selection": best, "test_rows": len(test), "scores": scores,
              "training_protocol": "Selected saved model from the last frozen development fold; no final-driven fitting or selection.",
              "checkpoint_fold": int(row.fold),
              "test_positive_prevalence": float(np.mean(y == 1)) if config["task"] == "classification" and len(np.unique(y)) == 2 else None}
    if state.get("metric_definitions"):
        label_names = list(read_json(root / "labels.json").values()) if config["task"] == "classification" else None
        result["diagnostic_metrics"] = [score_metric(definition, test, y, model.predict(transformed), probabilities, label_names)
                                        for definition in state["metric_definitions"]]
    write_json(folder / "evaluation.json", result)
    append_event(folder, "agent.final_evaluation", data=result)
    return result


def _prepare_worker(folder, workspace):
    import runpy
    tables = {}
    for name, item in read_json(workspace / "tables.json").items():
        path = (workspace / item["path"]).resolve()
        if not path.is_relative_to(workspace / "data"):
            raise ValueError("Invalid uploaded data path")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Uploaded table changed: {name}")
        tables[name] = pd.read_csv(path)
    namespace = runpy.run_path(str(folder / "preparation.py"), run_name="reviewed_preparation")
    frame = namespace["build_dataset"](tables, read_json(folder / "preparation_config.json"))
    joblib.dump(frame, folder / "prepared.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare"])
    parser.add_argument("folder", type=Path)
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args()
    _prepare_worker(args.folder, args.workspace)
