"""Reproducible local-Qwen study of 1/3/5 native-tool trial budgets."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import runpy
import time

import pandas as pd

from ml_pipeline.agent_loop import (advance, approve_models, create_session, evaluate_selected,
                                    visualize)
from ml_pipeline.benchmark_agent import read_json, write_json
from ml_pipeline.utils.playground_trace import read_events


OBJECTIVE = """At Friday's close, rank S&P 500 stocks by risk of a drop of at least 5% over the next
five trading observations. Primary metric: validation average precision (pr_auc); secondary: ROC-AUC
and useful rare-event recall/precision. Preparation and joins were reviewed previously and are frozen.
Use current/past trailing returns, volatility, volume change, Sector and index return; no future fields.
Compare the two approved CPU tree families. Initial comparison uses the reviewed small baseline.
Use remaining trials only for reasoned improvements; at every trial explain the limitation found and
why the next change might help. Diagnose stability and rare-event ranking where useful. You may
choose hyperparameters, original feature subsets, rowwise interactions and training-only resampling.
The number of trials is a maximum, not a requirement to invent experiments or claim strict gains.
Current constituent coverage and static metadata have survivorship limitations. No trading-edge claims.
"""


def main(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Run inside a GPU allocation; never load the 27B model on a login node")
    torch.set_num_threads(4)
    source = args.previous_workspace.resolve()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    previous = args.compare_to.resolve() if args.compare_to else None
    if previous == root:
        raise ValueError("Comparison source and new output must be different directories")
    dependencies = {"python": platform.python_version()}
    for name in ["torch", "transformers", "openai", "numpy", "pandas", "scikit-learn", "lightgbm", "joblib", "streamlit"]:
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = "not installed"
    write_json(root / "dependencies.json", dependencies)
    write_json(root / "execution_environment.json", {"slurm_job_id": os.getenv("SLURM_JOB_ID"),
               "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda})
    config = read_json(source / "revision_0001.json")
    config["model_params"]["lightgbm_classifier"].pop("verbosity", None)
    script = source / "revision_0001.py"
    manifest = read_json(source / "tables.json")
    tables = {}
    for name, item in manifest.items():
        path = source / item["path"]
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                raise ValueError(f"Uploaded source changed: {name}")
        tables[name] = pd.read_csv(path)
    frame = runpy.run_path(str(script), run_name="frozen_reviewed_preparation")["build_dataset"](tables, config)
    (root / "preparation.py").write_text(script.read_text())
    source_hashes = {}
    package = Path(__file__).resolve().parent
    for filename in ["agent_loop.py", "agent_evidence.py", "agent_finance_experiment.py", "benchmark_agent.py", "playground_runtime.py", "playground_app.py",
                     "forecasting_models.py", "forecasting_runtime.py",
                     "utils/local_qwen.py", "utils/llm_provider.py", "utils/playground_trace.py",
                     "utils/data_split.py",
                     "pipelines_torch/models.py", "pipelines_torch/base.py", "data_augmentation/augmentations.py"]:
        content = (package / filename).read_bytes()
        snapshot = root / "source" / filename
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(content)
        source_hashes[filename] = hashlib.sha256(content).hexdigest()
    write_json(root / "reviewed_config.json", config)
    write_json(root / "provenance.json", {"source_workspace": str(source), "tables": read_json(source / "tables.json"),
               "created_at": datetime.now(timezone.utc).isoformat(),
               "baseline_compatibility": "Removed verbosity from old LightGBM kwargs: the wrapper previously ignored it. This does not change the estimator used in the historical baseline.",
               "source_hashes": source_hashes,
               "comparison_source": str(previous) if previous else None,
               "caveat": "Final period is protected from this new agent search, but was inspected in the historical experiment. This is not a pristine prospective test."})
    print(f"STUDY={root} ROWS={len(frame)}", flush=True)
    settings = {"provider": "local", "model": "Qwen/Qwen3.8-27B", "device": "cuda",
                "quantization": "none", "reasoning": args.reasoning}
    write_json(root / "llm_settings.json", settings)
    started = time.monotonic()
    for budget in args.budgets:
        folder = root / f"budget_{budget}"
        if not (folder / "state.json").exists():
            create_session(folder, frame, config, max_trials=budget, context=OBJECTIVE, timeout=900)
        def progress(state):
            message = state["messages"][-1]
            print(f"BUDGET={budget} STEP={state['llm_steps']} TRIALS={len(state['trials'])} ROLE={message['role']} "
                  + json.dumps(message, ensure_ascii=False)[:2400], flush=True)
        state = advance(folder, settings, progress=progress)
        if state["status"] == "awaiting_models":
            approve_models(folder, ["random_forest_classifier", "lightgbm_classifier"], source="simulated_user")
            state = advance(folder, settings, progress=progress)
        if state["status"] != "completed":
            raise RuntimeError(f"Budget {budget} ended without a completed conversation: {state['status']}")
        visualize(folder, state)
        print(f"COMPLETE BUDGET={budget} BEST={state['best']}", flush=True)
    rows, checks, trial_rows = [], [], []
    reference_partitions, reference_baseline = None, None
    for budget in args.budgets:
        folder = root / f"budget_{budget}"
        state = read_json(folder / "state.json")
        partitions = read_json(folder / "partitions.json")
        if reference_partitions is None:
            reference_partitions = partitions
            reference_baseline = state["trials"][0]["summary"]
            if previous:
                assert partitions == read_json(previous / f"budget_{budget}" / "partitions.json"), "Previous study partitions changed"
                assert reference_baseline == read_json(previous / f"budget_{budget}" / "state.json")["trials"][0]["summary"], "Previous study baseline changed"
        assert partitions == reference_partitions, "Budgets must share exact partitions"
        assert state["trials"][0]["summary"] == reference_baseline, "Budgets must share a reproducible first trial"
        assistant_calls = [call for message in state["messages"] for call in message.get("tool_calls", [])]
        tool_replies = [message for message in state["messages"] if message["role"] == "tool"]
        assert [call["id"] for call in assistant_calls] == [reply["tool_call_id"] for reply in tool_replies]
        for trial in state["trials"]:
            splits = folder / f"trial_{trial['number']}" / "output" / "splits.json"
            if splits.exists():
                assert read_json(splits) == [{"train": partitions["train_in_development"],
                                              "validation": partitions["validation_in_development"]}]
            assert "execution" in trial and "comparison" in trial
            for row in trial.get("summary", []):
                trial_rows.append({"budget": budget, "trial": trial["number"], **row,
                    "selected": trial["number"] == state["best"]["trial"] and row["model"] == state["best"]["model"] and row["augmentation"] == state["best"]["augmentation"],
                    "changes_from_incumbent": json.dumps(trial["execution"]["changes_from_incumbent"]),
                    "pipeline_balancing_requested": trial["execution"]["resolved_settings"]["use_class_weights"]})
        approvals = [e for e in read_events(folder) if e["event"] == "agent.model_selection"]
        assert len(approvals) == 1 and approvals[0]["data"]["input_source"] == "simulated_user"
        assert len(state["trials"]) <= budget
        checks.append({"budget": budget, "all_native_calls_answered": True, "single_simulated_model_approval": True,
                       "identical_partitions_and_baseline": True, "fixed_splits_each_trial": True,
                       "execution_feedback_each_trial": True})
        result = evaluate_selected(folder)
        events = read_events(folder)
        calls = [call for message in state["messages"] for call in message.get("tool_calls", [])]
        errors = [m for m in state["messages"] if m["role"] == "tool" and "error" in json.loads(m["content"])]
        rows.append({"budget": budget, "trials": len(state["trials"]), "best_trial": state["best"]["trial"],
                     "best_model": state["best"]["model"], "validation_pr_auc": state["best"]["pr_auc"],
                     "test_pr_auc": result["scores"]["pr_auc"], "test_roc_auc": result["scores"]["roc_auc"],
                     "native_tool_calls": len(calls), "tool_errors": len(errors),
                     "failed_method_fits": sum(len(trial.get("errors", [])) for trial in state["trials"]),
                     "simulated_user_inputs": sum(e["event"] == "agent.model_selection" for e in events),
                     "tools_used": sorted(set(call["function"]["name"] for call in calls))})
    table = pd.DataFrame(rows)
    table.to_csv(root / "comparison.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(root / "trial_comparison.csv", index=False)
    if previous:
        old = pd.read_csv(previous / "comparison.csv")
        comparison = table.merge(old, on="budget", suffixes=("_current", "_previous"), validate="one_to_one")
        for metric in ["validation_pr_auc", "test_pr_auc", "test_roc_auc"]:
            comparison[f"{metric}_delta"] = comparison[f"{metric}_current"] - comparison[f"{metric}_previous"]
        comparison.to_csv(root / "previous_vs_current.csv", index=False)
    full = {"elapsed_seconds": time.monotonic() - started, "results": rows, "checks": checks,
            "strict_validation_order": bool(table.validation_pr_auc.is_monotonic_increasing and table.validation_pr_auc.is_unique) if len(table) > 1 else None,
            "strict_final_evaluation_order": bool(table.test_pr_auc.is_monotonic_increasing and table.test_pr_auc.is_unique) if len(table) > 1 else None,
            "comparison": "Requested budgets share a reviewed baseline and the same nested split. The earlier one-shot experiment without a reserved partition used a different split.",
            "limitations": ["One dataset, one seed and one local LLM; no statistical significance claim.",
                            "Historical test-period reuse, static metadata and constituent survivorship.",
                            "Validation best-so-far retention guarantees no decrease within one run, not strict improvements or better final performance."]}
    write_json(root / "report.json", full)
    (root / "README.md").write_text("# Native-tool finance experiment\n\n" +
        "Each budget folder contains messages.json, conversation.md, events.jsonl, state.json, trial scripts/configs/logs/checkpoints, validation.png and evaluation.json.\n\n" +
        "Exactly one model approval per conversation is simulated, explicitly labelled in the tool response. Final evaluation is performed after all requested conversations finish.\n\n" +
        table.to_string(index=False) + "\n\n" + json.dumps(full, indent=2) + "\n")
    print(table.to_string(index=False), flush=True)
    print("REPORT=" + str(root / "report.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--compare-to", type=Path, help="Previous study for checks/reporting only, never supplied to the LLM")
    parser.add_argument("--reasoning", choices=["none", "low", "medium", "xhigh"], default="xhigh")
    main(parser.parse_args())
