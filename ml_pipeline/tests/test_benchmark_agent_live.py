"""Opt-in local Qwen validation with the three S&P 500 CSV tables."""

import json
import os
from pathlib import Path
import runpy
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml_pipeline import benchmark_agent as agent
from ml_pipeline import playground_runtime as runtime


pytestmark = pytest.mark.skipif(os.getenv("RUN_BENCHMARK_AGENT_LIVE") != "1",
                               reason="Opt-in cached Qwen and real finance dataset validation")


CONTEXT = """This is an end-to-end smoke experiment, not an investment recommendation.
Use sp500_stocks.csv, sp500_companies.csv and sp500_index.csv.
Objective: at Friday's close, predict a drop of at least 5% in adjusted close over
the next 5 trading observations of the same symbol. Target name: large_drop_next_week.
Restrict prices to 2014-01-01 through 2024-12-31 BEFORE constructing the target.
Use all available symbols, not a universe selected by future returns or liquidity.
Discard source rows with missing/nonpositive adjusted close before shifting. Fail on
duplicate Symbol/Date keys. Sort by Symbol/Date. Calculate future_return_5d using
grouped shift(-5) of Adj Close divided by current Adj Close minus 1. Include
label_end_date from grouped Date.shift(-5). Explicitly drop unavailable future labels
at each symbol's end of history. Target = (future_return_5d <= -0.05).astype(int).
Keep only calendar Friday observations AFTER computing features/labels on daily rows.
Features: trailing 5, 21, 63 observation returns, trailing 21 observation daily-return
volatility, trailing 21 observation volume change, Sector, and same-date S&P500 index
trailing 21 observation return. Rolling windows must include only present/past rows.
Return Date and Symbol as identifiers for independent verification, not features.
Join companies on Symbol with validate='many_to_one', and index features on Date with
validate='many_to_one'. Check join coverage explicitly; retain missing historical index
features for train-only imputation rather than fabricating future values.
Exclude all raw price levels, future returns, label end times, Date, and Symbol from features.
Current company metadata and constituent coverage have historical survivorship limitations;
state these and do not claim a deployable backtest. Use standard preprocessing, only
random_forest_classifier and lightgbm_classifier, no resampling, temporal validation on
the latest 20% of prepared rows with split_column='Date' and label_end_column='label_end_date'.
Use PR-AUC as primary metric and ROC-AUC as secondary. No pretrained benchmark model
downloads, no GPU training. This specification resolves the task; do not ask optional questions.
"""


def test_local_qwen_finance_playground(monkeypatch):
    import torch
    from ml_pipeline import playground_app
    from ml_pipeline.utils.llm_provider import default_model

    assert torch.cuda.is_available(), "A GPU allocation is required; never run this on the login node"
    data_dir = Path(os.environ["PLAYGROUND_FINANCE_DATA"]).resolve()
    memory = Path(os.environ["PLAYGROUND_LIVE_MEMORY"]).resolve()
    started = time.monotonic()
    paths = [data_dir / name for name in ("sp500_stocks.csv", "sp500_companies.csv", "sp500_index.csv")]
    resumed = os.getenv("PLAYGROUND_LIVE_WORKSPACE")
    workspace = Path(resumed).resolve() if resumed else agent.create_workspace(memory, [(p.name, p.read_bytes()) for p in paths])
    print(f"LIVE_WORKSPACE={workspace}", flush=True)
    settings = {"provider": "local", "model": default_model("local"), "device": "cuda",
                "quantization": "none", "reasoning": "none"}
    plan = agent.read_json(workspace / "plan.json") if resumed else agent.propose(workspace, settings, CONTEXT)
    print("LIVE_PROPOSAL=" + json.dumps(plan), flush=True)
    assert plan["task"] == "classification"
    assert plan["target"] == "large_drop_next_week"
    assert plan["split"] == "temporal"
    config = {**{key: plan[key] for key in ("task", "table", "target", "features", "models",
                                           "augmentations", "preprocessing", "split", "split_column", "metric")},
              "label_end_column": "label_end_date", "seed": 42, "folds": 5,
              "validation_fraction": 0.2, "epochs": 10, "device": "cpu", "batch_size": 32,
              "learning_rate": 0.001, "max_factor": 2.0,
              "model_params": {"random_forest_classifier": {"n_estimators": 32, "max_depth": 8, "n_jobs": 4},
                               "lightgbm_classifier": {"n_estimators": 60, "num_leaves": 15, "n_jobs": 4, "verbosity": -1}}}
    assert set(config["models"]) == {"random_forest_classifier", "lightgbm_classifier"}
    if resumed:
        script = workspace / "revision_0001.py"
    else:
        script, _ = agent.generate_script(workspace, settings, config, CONTEXT)
    print(f"LIVE_SCRIPT={script}", flush=True)
    tables = {path.name: pd.read_csv(path) for path in paths}
    namespace = runpy.run_path(str(script), run_name="playground_validation")
    prepared = namespace["build_dataset"](tables, config)
    stocks = tables["sp500_stocks.csv"].copy()
    stocks["Date"] = pd.to_datetime(stocks.Date)
    stocks = stocks.loc[stocks.Date.between("2014-01-01", "2024-12-31") &
                        stocks["Adj Close"].notna() & (stocks["Adj Close"] > 0)].sort_values(["Symbol", "Date"])
    grouped = stocks.groupby("Symbol", sort=False)
    stocks["expected_return"] = grouped["Adj Close"].shift(-5) / stocks["Adj Close"] - 1
    stocks["expected_end"] = grouped.Date.shift(-5)
    checked = prepared.copy()
    checked["Date"] = pd.to_datetime(checked.Date)
    checked = checked.merge(stocks[["Date", "Symbol", "expected_return", "expected_end"]],
                            on=["Date", "Symbol"], how="left", validate="one_to_one")
    assert checked.expected_return.notna().all()
    np.testing.assert_array_equal(checked[config["target"]], (checked.expected_return <= -0.05).astype(int))
    np.testing.assert_array_equal(pd.to_datetime(checked.label_end_date).to_numpy(), checked.expected_end.to_numpy())
    assert (checked.Date.dt.dayofweek == 4).all()
    assert not set(config["features"]) & {"Date", "Symbol", "future_return_5d", "label_end_date", config["target"]}
    perturbed = {name: table.copy() for name, table in tables.items()}
    cutoff = pd.Timestamp("2022-01-01")
    future = pd.to_datetime(perturbed["sp500_stocks.csv"].Date) >= cutoff
    perturbed["sp500_stocks.csv"].loc[future, "Adj Close"] *= 2
    perturbed["sp500_stocks.csv"].loc[future, "Volume"] *= 3
    future_index = pd.to_datetime(perturbed["sp500_index.csv"].Date) >= cutoff
    perturbed["sp500_index.csv"].loc[future_index, "S&P500"] *= 2
    changed = namespace["build_dataset"](perturbed, config)
    def past_features(frame):
        frame = frame.loc[pd.to_datetime(frame.Date) < cutoff]
        return frame.set_index(["Date", "Symbol"])[config["features"]].sort_index()
    pd.testing.assert_frame_equal(past_features(prepared), past_features(changed))
    X, y, _ = runtime.prepare_dataset(prepared, config)
    train, validation = runtime.split_indices(prepared, y, config)[0]
    assert pd.to_datetime(prepared.label_end_date.iloc[train]).max() < pd.to_datetime(prepared.Date.iloc[validation]).min()
    print(f"LIVE_PANEL rows={len(prepared)} train={len(train)} validation={len(validation)}", flush=True)
    quiet_st = SimpleNamespace(empty=lambda: SimpleNamespace(code=lambda *args, **kwargs: None),
                               error=print, success=print)
    with monkeypatch.context() as patch:
        patch.setattr(playground_app, "st", quiet_st)
        result = playground_app._run(workspace, script.read_text(), config, timeout=600)
    assert result["returncode"] == 0, result["log"]
    assert not result["errors"], result["errors"]
    assert len(result["summary"]) == 2
    statistics = result["dataset_summary"]["splits"][0]
    assert statistics["validation_rows"] == len(validation)
    assert statistics["validation_class_counts"]["1"] == int(y[validation].sum())
    report = {"model": settings["model"], "device": torch.cuda.get_device_name(0),
              "rows": len(prepared), "train_rows": len(train), "validation_rows": len(validation),
              "purged_rows": len(prepared) - len(train) - len(validation),
              "positive_fraction_validation": float(y[validation].mean()),
              "future_perturbation_check": "passed",
              "summary": result["summary"], "elapsed_seconds": time.monotonic() - started}
    explanation = agent.interpret_results(workspace, settings)
    revised, _ = agent.generate_script(workspace, settings, config,
        "Keep the experiment configuration and preparation identical. Add a concise module docstring "
        "recording the observed validation metrics and explaining the limits. Use the saved result, no invented scores.")
    assert revised != script
    report["followup_revision"] = revised.name
    report["analysis"] = explanation["analysis"]
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_gpu_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
    report["dataset_summary"] = result["dataset_summary"]
    agent.write_json(workspace / "validation_report.json", report)
    print("LIVE_REPORT=" + json.dumps(report), flush=True)
