"""Real-checkpoint validation and high-reasoning conversations for forecasting."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ml_pipeline.agent_loop import advance, approve_models, create_session, evaluate_selected, visualize
from ml_pipeline.benchmark_agent import read_json, write_json
from ml_pipeline.forecasting_models import FORECASTING_MODEL_REGISTRY


MODELS = {"timestral_209m": "mistralai/Timestral-209M", "toto_2_0_313m": "Datadog/Toto-2.0-313m",
          "timesfm_3_0": "google/timesfm-3.0-pytorch", "chronos_2": "amazon/chronos-2"}


def main(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Run this real-model validation inside a GPU allocation")
    torch.set_num_threads(4)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    revisions = {item["model"]: item["revision"] for item in read_json(args.prefetch) if "revision" in item}
    package = Path(__file__).resolve().parent
    sources = {}
    for filename in ["forecasting_experiment.py", "forecasting_models.py", "forecasting_runtime.py", "agent_loop.py",
                     "agent_evidence.py", "playground_runtime.py", "playground_app.py", "benchmark_agent.py",
                     "agent_finance_experiment.py", "utils/local_qwen.py", "utils/llm_provider.py"]:
        content = (package / filename).read_bytes()
        destination = root / "source" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        sources[filename] = hashlib.sha256(content).hexdigest()
    write_json(root / "provenance.json", {"created_at": datetime.now(timezone.utc).isoformat(),
               "source_hashes": sources, "model_revisions": revisions, "gpu": torch.cuda.get_device_name(),
               "dependencies": {name: version(name) for name in ["torch", "transformers", "numpy", "pandas",
                   "timestral", "timesfm", "toto-2", "chronos-forecasting", "gluonts"]}})
    rng = np.random.default_rng(42)
    contexts = [100 + np.arange(128, dtype=np.float32) * .05 + np.sin(np.arange(128) / 5),
                200 + rng.normal(size=96).cumsum()]
    contexts[1][20:23] = np.nan
    checks = []
    for name, model_id in MODELS.items():
        model = None
        started = time.monotonic()
        print("SMOKE", name, flush=True)
        try:
            model = FORECASTING_MODEL_REGISTRY[name](revision=revisions[model_id], device="cuda", local_files_only=True, batch_size=2)
            prediction = model.forecast(contexts, 8)
            single = model.forecast(contexts[:1], 8)
            np.testing.assert_allclose(prediction.point[0], single.point[0], rtol=2e-4, atol=2e-3)
            assert prediction.quantiles is not None and prediction.quantiles.shape == (2, 8, 3)
            assert model.descriptor()["resolved_revision"] == revisions[model_id]
            item = {"model": name, "status": "passed", "seconds": time.monotonic() - started,
                    "descriptor": model.descriptor(), "point": prediction.point.tolist(),
                    "quantiles": prediction.quantiles.tolist(), "independent_batch_items": True}
        except Exception as exc:
            import traceback
            item = {"model": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        finally:
            if model is not None:
                model.close()
        checks.append(item)
        write_json(root / "checkpoint_smoke.json", checks)
        print(json.dumps({key: value for key, value in item.items() if key not in ("point", "quantiles", "traceback")}), flush=True)
    source = args.previous_workspace.resolve()
    manifest = read_json(source / "tables.json")
    item = manifest["sp500_index.csv"]
    data_path = source / item["path"]
    with data_path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
    frame = pd.read_csv(data_path)
    config = {"task": "forecasting", "target": "S&P500", "features": [], "split": "temporal",
              "split_column": "Date", "series_column": "", "metric": "mae", "seed": 42,
              "forecast_horizon": 5, "context_length": 128, "min_context": 64, "backtest_windows": 3,
              "models": ["naive_forecaster", "seasonal_naive_forecaster", *MODELS],
              "model_params": {name: {"revision": revisions[model_id], "local_files_only": True, "batch_size": 4}
                               for name, model_id in MODELS.items()},
              "augmentations": ["none"], "preprocessing": "none", "validation_fraction": .2, "device": "cuda"}
    config["model_params"]["seasonal_naive_forecaster"] = {"season_length": 5}
    write_json(root / "forecast_config.json", config)
    write_json(root / "data_source.json", {"table": "sp500_index.csv", "sha256": item["sha256"], "rows": len(frame),
               "purpose": "Integration backtest, not a generalization or trading-performance study. Pretraining overlap is unknown."})
    if args.skip_agent:
        from ml_pipeline.playground_runtime import benchmark
        benchmark(frame, config, root / "direct_backtest")
        return
    settings = {"provider": "local", "model": "Qwen/Qwen3.8-27B", "device": "cuda", "quantization": "none", "reasoning": "xhigh"}
    folder = root / "forecast_agent"
    create_session(folder, frame, config, max_trials=1, context="Forecast the S&P 500 index level for the next five observed trading dates. Compare zero-shot foundation models against naive forecasts, using validation MAE. One trial is sufficient for this integration check; inspect error by horizon if useful. Do not infer trading value or calibrated drop probabilities. Forecast pretraining overlap is unknown.")
    def progress(state):
        print("AGENT", state["status"], "steps", state["llm_steps"], "trials", len(state["trials"]), flush=True)
    state = advance(folder, settings, progress=progress)
    if state["status"] != "awaiting_models":
        raise RuntimeError(f"Expected a single model approval: {state['status']}")
    approve_models(folder, config["models"], source="simulated_user")
    state = advance(folder, settings, progress=progress)
    assert state["status"] == "completed" and len(state["notes"]) == len(state["trials"]) == 1
    visualize(folder, state)
    evaluation = evaluate_selected(folder)
    write_json(root / "forecast_agent_check.json", {"status": state["status"], "structured_notes": len(state["notes"]),
               "best": state["best"], "final_evaluation": evaluation,
               "all_requested_models_succeeded": len(state["trials"][0]["summary"]) == len(config["models"])})
    if args.finance_budgets:
        from ml_pipeline.agent_finance_experiment import main as finance_main
        finance_main(SimpleNamespace(previous_workspace=source, output=root / "finance_agent",
                                     compare_to=args.compare_to, budgets=args.finance_budgets, reasoning="xhigh"))
    write_json(root / "validation_status.json", {"checkpoint_smokes": [{"model": item["model"], "status": item["status"]} for item in checks],
               "forecast_conversation_saved": True, "finance_budgets": args.finance_budgets})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-workspace", type=Path, required=True)
    parser.add_argument("--prefetch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--finance-budgets", type=int, nargs="*", default=[3])
    parser.add_argument("--skip-agent", action="store_true")
    main(parser.parse_args())
