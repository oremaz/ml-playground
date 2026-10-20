"""Streamlit assistant workflow for the benchmark playground."""

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pandas as pd
import streamlit as st

from ml_pipeline.benchmark_agent import (create_workspace, execution_environment, interpret_results,
                                        generate_script, method_catalog, propose, read_json,
                                        recent_agent_context, tabular_models, write_json)
from ml_pipeline.utils.llm_provider import PROVIDERS, REASONING_EFFORTS, default_model, default_reasoning
from ml_pipeline.utils.playground_trace import export_history, history_markdown, operation
from ml_pipeline.utils.utils import RESULTS_DIR_OUT


def _provider_settings():
    with st.sidebar:
        st.header("Assistant")
        provider = st.selectbox("LLM provider", PROVIDERS)
        model = st.text_input("LLM model", value=default_model(provider), key=f"model_{provider}")
        key = ""
        device, quantization = None, "none"
        if provider != "local":
            configured = os.getenv(f"{provider.upper()}_API_KEY", "")
            if configured:
                st.caption(f"Using {provider.upper()}_API_KEY from the environment")
            else:
                key = st.text_input("API key", type="password", key=f"api_key_{provider}")
        else:
            st.caption("Qwen BF16 needs about 58 GB for weights plus working memory.")
            device = st.text_input("Local LLM device", value=os.getenv("LOCAL_LLM_DEVICE", "")) or None
            quantization = st.selectbox("Local precision", ["none", "4bit"])
        levels = REASONING_EFFORTS[provider]
        reasoning = st.selectbox("Reasoning effort", levels,
                                 index=levels.index(default_reasoning(provider)),
                                 key=f"reasoning_{provider}")
    return {"provider": provider, "model": model, "api_key": key, "reasoning": reasoning,
            "device": device, "quantization": quantization}


def _configuration(plan, workspace):
    plan_id = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    prefix = f"{workspace.name}_{plan_id}"
    defaults = plan.copy()
    saved = sorted(workspace.glob("revision_*.json"))
    if saved and saved[-1].stat().st_mtime_ns >= (workspace / "plan.json").stat().st_mtime_ns:
        defaults.update(read_json(saved[-1]))
    with st.expander("Review experiment configuration", expanded=True):
        st.write(plan.get("preparation", ""))
        st.caption(plan.get("rationale", ""))
        tasks = ["classification", "regression", "forecasting"]
        task = st.selectbox("Task", tasks, index=tasks.index(defaults["task"]), key=f"task_{prefix}")
        tables = list(read_json(workspace / "tables.json"))
        table = st.selectbox("Primary table", tables, index=tables.index(defaults["table"]), key=f"table_{prefix}")
        target = st.text_input("Target after preparation", value=defaults.get("target", ""), key=f"target_{prefix}")
        if task == "forecasting":
            features_text = "[]"
            st.caption("Forecasts use past values of the numeric target per series. Horizon and stride count observations, not calendar durations.")
        else:
            features_text = st.text_area("Features after preparation (JSON list)",
                                         value=json.dumps(defaults["features"]), key=f"features_{prefix}")
        models = st.multiselect("Methods to compare", tabular_models(task),
                                default=[m for m in defaults["models"] if m in tabular_models(task)], key=f"models_{task}_{prefix}")
        available_augs = list(method_catalog()["augmentations"]) if task == "classification" else ["none"]
        augmentations = st.multiselect("Training augmentations", available_augs,
                                       default=[a for a in defaults.get("augmentations", ["none"]) if a in available_augs],
                                       key=f"augmentations_{task}_{prefix}")
        preprocessing = "none" if task == "forecasting" else st.selectbox("Preprocessing", ["standard", "skrub"],
                                     index=int(defaults.get("preprocessing") == "skrub"), key=f"preprocessing_{prefix}")
        metrics = ["f1_macro", "accuracy", "roc_auc", "pr_auc"] if task == "classification" else ["mae", "rmse", "smape"] if task == "forecasting" else ["rmse", "mae", "r2"]
        metric = st.selectbox("Primary validation metric", metrics,
                              index=metrics.index(defaults["metric"]) if defaults.get("metric") in metrics else 0,
                              key=f"metric_{task}_{prefix}")
        strategies = ["temporal"] if task == "forecasting" else ["holdout", "kfold", "group", "temporal"]
        split = st.selectbox("Validation split", strategies,
                             index=strategies.index(defaults.get("split")) if defaults.get("split") in strategies else 0,
                             key=f"split_{prefix}")
        split_column = st.text_input("Group or time column", value=defaults.get("split_column") or "", key=f"split_column_{prefix}")
        label_end_column = "" if task == "forecasting" else st.text_input("Label end-time column (forward-looking targets)",
                                         value=defaults.get("label_end_column") or "", key=f"label_end_{prefix}")
        forecast_config = {}
        if task == "forecasting":
            forecast_config["series_column"] = st.text_input("Series ID column (empty for one series)", value=defaults.get("series_column", ""), key=f"series_{prefix}")
            for key, label, default, low, high in [
                ("forecast_horizon", "Forecast horizon", 12, 1, 1024),
                ("context_length", "Maximum history observations", 512, 8, 8192),
                ("min_context", "Minimum history observations", 32, 2, 8192),
                ("backtest_windows", "Latest backtest windows per series", 3, 1, 100),
                ("forecast_stride", "Steps between forecast origins", defaults.get("forecast_horizon", 12), 1, 1024)]:
                forecast_config[key] = int(st.number_input(label, min_value=low, max_value=high,
                    value=int(defaults.get(key, default)), key=f"{key}_{prefix}"))
        left, right = st.columns(2)
        with left:
            seed = st.number_input("Seed", min_value=0, max_value=2**32 - 1, value=defaults.get("seed", 42), key=f"seed_{prefix}")
            fraction = st.number_input("Validation fraction", min_value=0.05, max_value=0.8, value=defaults.get("validation_fraction", 0.2), key=f"fraction_{prefix}")
            folds = 5 if task == "forecasting" else st.number_input("CV folds", min_value=2, max_value=10, value=defaults.get("folds", 5), key=f"folds_{prefix}")
            epochs = 10 if task == "forecasting" else st.number_input("Training epochs", min_value=1, max_value=500, value=defaults.get("epochs", 10), key=f"epochs_{prefix}")
        with right:
            device = st.selectbox("Benchmark device", ["cpu", "cuda"], index=int(defaults.get("device") == "cuda"), key=f"device_{prefix}")
            batch = 32 if task == "forecasting" else st.number_input("Training batch size", min_value=2, max_value=2048, value=defaults.get("batch_size", 32), key=f"batch_{prefix}")
            lr = .001 if task == "forecasting" else st.number_input("Training learning rate", min_value=0.000001, max_value=1.0, value=defaults.get("learning_rate", 0.001),
                                 format="%.6f", key=f"lr_{prefix}")
            factor = 2. if task == "forecasting" else st.number_input("Augmentation max factor", min_value=1.0, max_value=10.0, value=defaults.get("max_factor", 2.0), key=f"factor_{prefix}")
        params_text = st.text_area("Per-method parameters (JSON object)", value=json.dumps(defaults.get("model_params", {})), key=f"params_{prefix}")
        try:
            features, params = json.loads(features_text), json.loads(params_text)
            if not isinstance(features, list) or not all(isinstance(c, str) for c in features):
                raise ValueError("Features must be a JSON list of column names")
            if not isinstance(params, dict) or not all(isinstance(v, dict) for v in params.values()):
                raise ValueError("Method parameters must map model names to JSON objects")
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
            return None
    return {"task": task, "table": table, "target": target, "features": features,
            "models": models, "augmentations": augmentations or ["none"], "preprocessing": preprocessing,
            "split": split, "split_column": split_column, "seed": int(seed), "folds": int(folds),
            "validation_fraction": fraction, "epochs": int(epochs), "device": device,
            "batch_size": int(batch), "learning_rate": lr, "max_factor": factor, "model_params": params,
            "metric": metric, "label_end_column": label_end_column, **forecast_config}


def _run(workspace, script, config, timeout):
    with operation(workspace, "script.execute", {
        "config": config, "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "timeout_seconds": timeout}) as trace:
        result = _run_script(workspace, script, config, timeout, trace["operation_id"])
        trace.update(status="failed" if result["returncode"] or result["execution_error"] or not result["summary"] else "completed",
                     returncode=result["returncode"], error=result["execution_error"],
                     output=result["output"], script=result["script"], summary=result["summary"], errors=result["errors"])
        return result


def _run_script(workspace, script, config, timeout, trace_parent):
    ast.parse(script)
    run_id = "run_" + uuid.uuid4().hex
    config_path = workspace / f"{run_id}.json"
    output = workspace / "runs" / run_id
    config = {**config, "output": str(output.relative_to(workspace))}
    write_json(config_path, config)
    script_path = workspace / f"{run_id}.py"
    script_path.write_text(script)
    log_path = workspace / f"{run_id}.log"
    status = st.empty()
    started = time.monotonic()
    execution_error = None
    environment = execution_environment()
    environment.update(ML_PIPELINE_TRACE_PARENT=trace_parent, ML_PIPELINE_TRACE_RUN_ID=run_id)
    with log_path.open("w") as log:
        process = subprocess.Popen([sys.executable, "-u", str(script_path), str(config_path)],
                                   cwd=workspace, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"Benchmark exceeded {timeout} seconds")
                with log_path.open("rb") as reader:
                    reader.seek(max(0, log_path.stat().st_size - 12000))
                    status.code(reader.read().decode(errors="replace"), language="text")
                time.sleep(0.5)
        except Exception as exc:
            execution_error = str(exc)
        finally:
            if process.poll() is None:
                import signal
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    with log_path.open("rb") as reader:
        reader.seek(max(0, log_path.stat().st_size - 16000))
        log_tail = reader.read().decode(errors="replace")
    summary = pd.read_csv(output / "summary.csv") if (output / "summary.csv").exists() else pd.DataFrame()
    result = {"output": str(output.relative_to(workspace)), "script": script_path.name,
              "returncode": process.returncode, "log": log_tail,
              "execution_error": execution_error,
              "config": config,
              "labels": read_json(output / "labels.json") if (output / "labels.json").exists() else None,
              "dataset_summary": read_json(output / "dataset_summary.json") if (output / "dataset_summary.json").exists() else None,
              "summary": json.loads(summary.to_json(orient="records")),
              "errors": read_json(output / "errors.json") if (output / "errors.json").exists() else []}
    write_json(workspace / "last_result.json", result)
    if execution_error or process.returncode or summary.empty:
        st.error("Benchmark failed. The saved log is available for a follow-up revision.")
    else:
        st.success("Benchmark complete")
    return result


def _results(workspace, settings):
    result_path = workspace / "last_result.json"
    if not result_path.exists():
        return
    result = read_json(result_path)
    st.subheader("Results")
    summary = pd.DataFrame(result["summary"])
    if not summary.empty:
        metric_options = [c for c in summary.select_dtypes(include="number") if c not in ("completed_folds", "expected_folds")]
        if metric_options:
            config = result.get("config", {})
            preferred = config.get("metric")
            metric = st.selectbox("Rank by validation metric", metric_options,
                                  index=metric_options.index(preferred) if preferred in metric_options else 0)
            complete = summary[summary.completed_folds == summary.expected_folds]
            ranked = complete.sort_values(metric, ascending=metric in ("mae", "rmse", "smape", "mase", "pinball_loss"))
            st.dataframe(ranked, use_container_width=True)
            if not ranked.empty:
                st.bar_chart(ranked.assign(method=ranked.model + " | " + ranked.augmentation).set_index("method")[[metric]])
        st.download_button("Download all metrics", summary.to_csv(index=False), "benchmark_summary.csv", "text/csv")
    st.caption("Validation estimates used for comparison. Follow-up model selection needs an independent test set.")
    if result.get("dataset_summary"):
        with st.expander("Split counts and class baselines"):
            st.json(result["dataset_summary"])
    if result["errors"]:
        st.warning("Some methods failed. Incomplete cross-validation runs are excluded from the ranking.")
        st.dataframe(pd.DataFrame(result["errors"]), use_container_width=True)
    if result.get("execution_error"):
        st.error(result["execution_error"])
    labels_path = workspace / result["output"] / "labels.json"
    if labels_path.exists():
        st.caption(f"Target encoding (class 1 is positive for binary ROC-AUC/PR-AUC): {read_json(labels_path)}")
    with st.expander("Execution log and artifacts"):
        st.code(result["log"], language="text")
        st.code(str(workspace / result["output"]), language="text")
    if st.button("Explain these results"):
        try:
            with st.spinner("Interpreting validation results..."):
                interpret_results(workspace, settings)
        except Exception as exc:
            st.error(f"Result interpretation failed: {exc}")
    interpretation_path = workspace / "interpretation.json"
    if interpretation_path.exists():
        explanation = read_json(interpretation_path)
        if explanation.get("run") == result["output"]:
            st.write(explanation.get("analysis", ""))


def _iterative_assistant(workspace, settings, source, config, timeout, *, disabled=False, instruction=""):
    import joblib
    from ml_pipeline.agent_loop import (advance, approve_models, create_session, evaluate_selected,
                                        prepare_reviewed_dataset)

    def continue_session(folder):
        status = st.empty()

        def progress(state):
            latest = next((message["content"] for message in reversed(state["messages"])
                           if message["role"] == "assistant" and message.get("content")), "Inspecting results...")
            status.info(f"Trial attempts: {len(state['trials'])}/{state['max_trials']}\n\n{latest}")

        if (workspace / "autonomous.json").exists():
            from ml_pipeline.agent_workflow import resume
            return resume(workspace, settings, progress=progress)
        return advance(folder, settings, progress=progress)

    if not disabled:
        st.write("Let the assistant inspect results and test a few focused improvements. You choose the model families once.")
        budget = st.slider("Maximum benchmark trials", min_value=1, max_value=5, value=3,
                           key=f"agent_budget_{workspace.name}")
        st.caption("One trial can compare several approved methods. The assistant may stop early. Extra trials do not guarantee better performance.")
        test_fraction = st.slider("Reserve for final evaluation", min_value=0.05, max_value=0.4, value=0.2, step=0.05,
                                  key=f"agent_test_fraction_{workspace.name}")
        fixed_holdout = config and config.get("split") != "kfold"
        if not fixed_holdout:
            st.info("Choose a fixed holdout, group or temporal split for iterative comparisons.")
        if st.button("Start iterative experiment", disabled=disabled or not fixed_holdout):
            try:
                with st.spinner("Preparing the reviewed dataset and inspecting the experiment..."):
                    folder = prepare_reviewed_dataset(workspace, source, config, timeout)
                    context_path = workspace / "context.json"
                    context = {"original_context": read_json(context_path) if context_path.exists() else "",
                               "followup": instruction, "previous_agent_experiment": recent_agent_context(workspace)}
                    create_session(folder, joblib.load(folder / "prepared.joblib"), config, max_trials=budget,
                                   context=context, test_fraction=test_fraction, timeout=timeout)
                    st.session_state[f"agent_session_{workspace.name}"] = folder.name
                    continue_session(folder)
                st.rerun()
            except Exception as exc:
                st.error(f"Iterative experiment failed: {exc}. Any completed messages and trials are saved.")
    sessions = sorted((workspace / "agents").glob("*/state.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not sessions:
        return
    selected = st.selectbox("Saved iterative experiment", [p.parent.name for p in sessions],
                            key=f"agent_session_{workspace.name}")
    folder = workspace / "agents" / selected
    write_json(workspace / "active_agent.json", {"session": selected})
    state = read_json(folder / "state.json")
    if state.get("protocol_version", 1) >= 2:
        for note in state.get("notes", []):
            with st.chat_message("assistant"):
                st.write(f"Trial {note['trial']}")
                if note.get("finding"):
                    st.write(note["finding"])
                st.write("Measured")
                for fact in note["measured"]:
                    st.write(fact["description"])
                if note["hypothesis"]:
                    st.write("Hypotheses, not established causes")
                    st.write(note["hypothesis"])
                st.caption("Alternative / unresolved: " + note["alternative"])
                if note["next_test"]:
                    st.write("Next experiment: " + note["next_test"])
                    st.caption("Intended comparison: " + note["distinguishes"])
                if note.get("next_changes"):
                    with st.expander("Exact changes for this experiment"):
                        st.json(note["next_changes"])
        if state.get("conclusion"):
            st.info(state["conclusion"])
    for message in state["messages"] if state.get("protocol_version", 1) < 2 else []:
        if message["role"] == "assistant" and message.get("content"):
            with st.chat_message("assistant"):
                content = message["content"]
                if len(content.split()) <= 180:
                    st.write(content)
                else:
                    paragraphs, words = [], 0
                    for paragraph in content.split("\n\n"):
                        if words + len(paragraph.split()) > 180:
                            break
                        paragraphs.append(paragraph)
                        words += len(paragraph.split())
                    st.write("\n\n".join(paragraphs) if paragraphs else " ".join(content.split()[:180]) + "...")
                    with st.expander("Full explanation"):
                        st.write(content)
        for call in message.get("tool_calls", []):
            try:
                reason = json.loads(call["function"]["arguments"]).get("reason", "")
            except (ValueError, TypeError):
                reason = ""
            if reason:
                st.caption(f"{call['function']['name']}: {reason}")
    if state["status"] == "awaiting_models":
        st.write(state["approval_reason"])
        models = st.multiselect("Model families for this experiment", tabular_models(state["config"]["task"]),
                                default=state["proposed_models"], key=f"agent_models_{selected}")
        if st.button("Approve models and continue", disabled=not models):
            try:
                approve_models(folder, models)
                with st.spinner("Running focused trials and inspecting their results..."):
                    continue_session(folder)
                st.rerun()
            except Exception as exc:
                st.error(f"Agent paused after an error: {exc}. Resume below; saved trials are retained.")
    elif state["status"] == "running" and st.button("Resume iterative experiment"):
        try:
            with st.spinner("Continuing from the saved conversation..."):
                continue_session(folder)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    rows = [{"trial": trial["number"], **row} for trial in state["trials"] for row in trial.get("summary", [])]
    for row in rows:
        for item in state["trials"][row["trial"] - 1].get("metric_results", []):
            if item["model"] == row["model"] and item["augmentation"] == row["augmentation"]:
                row.update({metric["name"]: metric["value"] for metric in item["metrics"]})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.write({"best_validation_candidate": state["best"]})
    if state.get("metric_definitions"):
        with st.expander("Task-specific diagnostic definitions"):
            st.json(state["metric_definitions"])
    for trial in state["trials"]:
        if trial.get("error"):
            st.warning(f"Trial {trial['number']}: {trial['error']}")
        if trial.get("execution"):
            with st.expander(f"Trial {trial['number']}: executed settings and comparison"):
                st.json({"execution": trial["execution"], "comparison": trial.get("comparison")})
    if (folder / "validation.png").exists():
        st.image(str(folder / "validation.png"))
    if state["status"] == "completed" and not (folder / "evaluation.json").exists():
        if st.button("Evaluate the locked selection on reserved data"):
            try:
                evaluate_selected(folder)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    if (folder / "evaluation.json").exists():
        st.write("Final evaluation after validation selection was locked:")
        st.json(read_json(folder / "evaluation.json"))
        st.caption("Starting another search after inspecting these scores reuses the evaluation period.")
    st.download_button("Download full native-tool conversation", (folder / "conversation.md").read_text(),
                       f"agent_{selected}.md", "text/markdown")
    st.caption(f"Saved scripts, trial configurations, logs and native messages: {folder}")


def render_playground():
    st.write("Upload related CSVs, describe your goal, review the assistant's proposed experiment, then benchmark it.")
    settings = _provider_settings()
    root = Path(os.getenv("ML_PIPELINE_PLAYGROUND_MEMORY", str(Path(RESULTS_DIR_OUT) / "playground_memory"))).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with st.sidebar:
        st.header("Saved playgrounds")
        saved = sorted([p.name for p in root.iterdir() if p.is_dir() and (p / "tables.json").exists()])
        chosen = st.selectbox("Workspace", [""] + saved)
        if st.button("Open saved workspace", disabled=not chosen):
            st.session_state["playground_workspace"] = str(root / chosen)
            st.rerun()
    autonomous_mode = st.toggle("Autonomous workflow", value=True,
        help="Prepare data, request model approval once, run bounded trials and evaluate the locked result.")
    auto_budget = st.slider("Autonomous trial budget", 1, 5, 5) if autonomous_mode else 5
    uploads = st.file_uploader("CSV tables", type=["csv"], accept_multiple_files=True)
    context = st.text_area("What would you like to learn?", placeholder="Describe the tables, their relationships, the outcome and any timing constraints.")
    st.caption("Hosted providers receive column statistics and up to five sample rows per table. Uploaded CSV files and scripts are saved locally.")
    if st.button("Analyze CSVs and propose experiments", disabled=not uploads):
        try:
            workspace = create_workspace(root, [(u.name, u.getvalue()) for u in uploads])
            st.session_state["playground_workspace"] = str(workspace)
            with st.spinner("Analyzing table profiles..."):
                if autonomous_mode:
                    from ml_pipeline.agent_workflow import start
                    start(workspace, settings, context, max_trials=auto_budget)
                else:
                    propose(workspace, settings, context)
            st.rerun()
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
    current = st.session_state.get("playground_workspace")
    if not current:
        return
    workspace = Path(current)
    st.caption(f"Saved workspace: {workspace.name}")
    with st.expander("Table profiles"):
        st.json(read_json(workspace / "profiles.json"))
    if (workspace / "autonomous.json").exists():
        flow = read_json(workspace / "autonomous.json")
        st.caption(f"Autonomous phase: {flow['phase']}. Maximum {flow['max_trials']} benchmark trials.")
        if (workspace / "plan.json").exists():
            with st.expander("Proposed task, preparation and evaluation"):
                st.json(read_json(workspace / "plan.json"))
        if flow.get("folder"):
            _iterative_assistant(workspace, settings, "", flow["config"], flow["timeout"], disabled=True)
        elif st.button("Resume autonomous preparation"):
            try:
                from ml_pipeline.agent_workflow import resume
                resume(workspace, settings)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        if flow.get("recoveries"):
            with st.expander("Automatic recovery history"):
                st.json(flow["recoveries"])
        _history(workspace)
        return
    followup = st.text_area("Follow-up or answers for the assistant", key=f"followup_{workspace.name}")
    if st.button("Update proposal from follow-up", disabled=not followup):
        try:
            with st.spinner("Revising the proposal..."):
                propose(workspace, settings, followup)
            st.rerun()
        except Exception as exc:
            st.error(f"Proposal failed: {exc}")
    plan_path = workspace / "plan.json"
    if not plan_path.exists():
        st.info("Provide context or answers in the follow-up field and retry the proposal.")
        _history(workspace)
        return
    plan = read_json(plan_path)
    for insight in plan.get("insights", []):
        st.write(f"• {insight}")
    for question in plan.get("questions", []):
        st.info(question)
    config = _configuration(plan, workspace)
    if config and st.button("Generate or revise benchmark script", disabled=not config["target"] or not config["features"] or not config["models"]):
        try:
            with st.spinner("Writing the experiment script..."):
                path, explanation = generate_script(workspace, settings, config, followup)
            st.session_state[f"explanation_{workspace.name}"] = explanation
            st.rerun()
        except Exception as exc:
            st.error(f"Script generation failed: {exc}")
    revisions = sorted(workspace.glob("revision_*.py"))
    if revisions:
        script_path = revisions[-1]
        st.write(st.session_state.get(f"explanation_{workspace.name}", ""))
        source = st.text_area("Review or edit the generated Python", value=script_path.read_text(),
                              height=360, key=str(script_path))
        st.download_button("Download script", source, script_path.name, "text/x-python")
        st.caption("Run executes this Python on the trusted research VM with the server process's permissions.")
        timeout = st.number_input("Execution timeout (minutes)", min_value=1, max_value=1440, value=30)
        stale = config != read_json(script_path.with_suffix(".json"))
        if stale:
            st.info("Configuration changed. Generate a revised script before running.")
        if st.button("Run benchmark", disabled=not config or stale):
            try:
                _run(workspace, source, config, int(timeout) * 60)
            except Exception as exc:
                st.error(f"Execution failed: {exc}")
        if st.checkbox("Enable iterative assistant", key=f"enable_agent_{workspace.name}"):
            _iterative_assistant(workspace, settings, source, config, int(timeout) * 60,
                                 disabled=not config or stale, instruction=followup)
    _results(workspace, settings)
    _history(workspace)


def _history(workspace):
    with st.expander("Conversation and execution trace"):
        history = export_history(workspace)
        st.caption(history["note"])
        st.download_button("Download conversation and trace (JSON)",
                           json.dumps(history, indent=2, ensure_ascii=False),
                           "playground_history.json", "application/json")
        st.download_button("Download conversation and trace (Markdown)", history_markdown(history),
                           "playground_history.md", "text/markdown")
        conversation, trace_tab, artifacts = st.tabs(["Conversation", "Execution trace", "Saved artifacts"])
        with conversation:
            messages = history["artifacts"].get("conversation.json", {}).get("content", [])
            for message in messages:
                with st.chat_message(message["role"]):
                    if isinstance(message["content"], str):
                        st.write(message["content"])
                    else:
                        st.json(message["content"])
            native = [name for name in history["artifacts"] if name.endswith("/messages.json")]
            for name in native:
                st.write(f"Native-tool conversation: {name}")
                for message in history["artifacts"][name]["content"]:
                    if message["role"] == "system":
                        continue
                    with st.chat_message(message["role"]):
                        st.write(message.get("content", ""))
                        if message.get("tool_calls"):
                            st.json(message["tool_calls"])
        with trace_tab:
            events = history["events"]
            if not events:
                st.info("This experiment predates detailed tracing. Its saved conversation and artifacts are available.")
            else:
                labels = [f"{event['timestamp']} | {event['event']} | {event.get('operation') or ''}"
                          for event in events]
                selected = st.selectbox("Trace event", list(range(len(events))),
                                        format_func=labels.__getitem__, index=len(events) - 1,
                                        key=f"trace_event_{workspace.name}")
                st.json(events[selected])
        with artifacts:
            choices = list(history["artifacts"])
            if choices:
                selected = st.selectbox("Saved conversation artifact", choices)
                content = history["artifacts"][selected]["content"]
                if isinstance(content, (dict, list)):
                    st.json(content)
                else:
                    st.code(content, language="python" if selected.endswith(".py") else "text")
