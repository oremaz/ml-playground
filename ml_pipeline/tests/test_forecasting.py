import json

import numpy as np
import pandas as pd
import pytest

from ml_pipeline import agent_loop as agent
from ml_pipeline.benchmark_agent import method_catalog, read_json
from ml_pipeline.forecasting_models import FORECASTING_MODEL_REGISTRY, Forecast, NaiveForecaster
from ml_pipeline.forecasting_runtime import forecast_windows, forecast_scores
from ml_pipeline.playground_runtime import benchmark, prepare_dataset, split_indices
from ml_pipeline.tests.test_agent_evidence import native, review


def panel():
    dates = pd.date_range("2020-01-01", periods=200)
    return pd.concat([pd.DataFrame({"time": dates, "series": name,
                                    "value": offset + np.sin(np.arange(200) * 2 * np.pi / 7)})
                      for name, offset in [("a", 0), ("b", 100)]], ignore_index=True)


def config():
    return {"task": "forecasting", "target": "value", "features": [], "split": "temporal",
            "split_column": "time", "series_column": "series", "metric": "mae", "seed": 42,
            "forecast_horizon": 4, "context_length": 32, "min_context": 16, "backtest_windows": 3,
            "models": ["naive_forecaster", "seasonal_naive_forecaster"],
            "augmentations": ["none"], "preprocessing": "none", "validation_fraction": .2, "device": "cpu"}


def test_catalog_has_separate_forecasting_family_without_polluting_classifiers():
    catalog = method_catalog()
    for name in ["timestral_209m", "toto_2_0_313m", "timesfm_3_0", "chronos_2"]:
        assert name in catalog["forecasting"] and name not in catalog["classification"]
    assert len([name for name in FORECASTING_MODEL_REGISTRY if name.startswith("toto_")]) == 5


def test_past_only_windows_isolate_series_and_horizon_values():
    frame, cfg = panel(), config()
    _, y, _ = prepare_dataset(frame, cfg)
    _, validation = split_indices(frame, y, cfg)[0]
    windows, skipped = forecast_windows(frame, cfg, validation)
    assert not skipped and len(windows) == 6
    first = windows[0]
    changed = frame.copy()
    changed.loc[first["rows"], "value"] += 1_000_000
    changed.loc[changed.series == "b", "value"] += 9_000_000
    altered, _ = forecast_windows(changed, cfg, validation)
    np.testing.assert_array_equal(first["context"], altered[0]["context"])
    assert not np.array_equal(first["actual"], altered[0]["actual"])
    assert set(frame.iloc[first["history_rows"]].series) == {first["series_id"]}
    assert pd.to_datetime(frame.iloc[first["history_rows"]].time).max() < pd.to_datetime(frame.iloc[first["rows"]].time).min()


def test_real_naive_backtest_replays_saved_predictions_and_retains_failures(tmp_path, monkeypatch):
    class Broken:
        def __init__(self, device="cpu"):
            raise ImportError("Optional weights unavailable")
    monkeypatch.setitem(FORECASTING_MODEL_REGISTRY, "broken_forecaster", Broken)
    cfg = {**config(), "models": config()["models"] + ["broken_forecaster"]}
    output = tmp_path / "output"
    summary = benchmark(panel(), cfg, output)
    assert len(summary) == 2
    assert summary.set_index("model").loc["seasonal_naive_forecaster", "mae"] < 1e-5
    assert len(read_json(output / "errors.json")) == 1
    for row in pd.read_csv(output / "fold_metrics.csv").itertuples():
        predictions = pd.read_csv(output / f"predictions_{row.checkpoint_id}.csv")
        scores = forecast_scores(predictions.y_true.to_numpy(), predictions.y_pred.to_numpy())
        assert scores["mae"] == pytest.approx(row.mae)
    assert not (output / "preprocessor_fold_1.joblib").exists()


@pytest.mark.parametrize("change,message", [
    ({"split": "holdout"}, "temporal"), ({"features": ["other"]}, "target history"),
    ({"forecast_horizon": 0}, "forecast_horizon"), ({"min_context": 64}, "context_length"),
    ({"augmentations": ["smote"]}, "resampling"), ({"metric": "pr_auc"}, "selection metric"),
])
def test_invalid_forecasting_protocol_is_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        prepare_dataset(panel(), {**config(), **change})


def test_duplicate_times_cannot_silently_mix_entities():
    data = pd.concat([panel(), panel().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        prepare_dataset(data, config())


def test_invalid_uncertainty_and_causal_naive_fill():
    forecast = Forecast(np.ones((1, 2)), np.array([[[2., 1., 3.], [0., 1., 2.]]]))
    with pytest.raises(ValueError, match="cross"):
        forecast.validate(1, 2)
    out = NaiveForecaster().forecast([np.array([1., 2., np.nan])], 3)
    np.testing.assert_array_equal(out.point, [[2., 2., 2.]])


def test_structured_forecast_loop_and_locked_final_evaluation(tmp_path, monkeypatch):
    cfg = {**config(), "models": ["seasonal_naive_forecaster"],
           "model_params": {"seasonal_naive_forecaster": {"season_length": 1}}}
    agent.create_session(tmp_path, panel(), cfg, max_trials=2, context="Compare seasonal patterns")
    monkeypatch.setattr(agent, "complete_message", lambda *a, **k: native("request_model_selection", "approve", models=cfg["models"]))
    agent.advance(tmp_path, {})
    agent.approve_models(tmp_path, cfg["models"])
    responses = iter([
        native("run_benchmark", "baseline", overrides_json="{}"),
        native("review_trial", "note_1", **review(1, "continue", model=cfg["models"][0], metric="mae", overrides={
            "model_params": {cfg["models"][0]: {"season_length": 7}}})),
        native("review_trial", "note_2", **review(2, "stop", model=cfg["models"][0], metric="mae")),
    ])
    monkeypatch.setattr(agent, "complete_message", lambda *a, **k: next(responses))
    state = agent.advance(tmp_path, {})
    assert state["status"] == "completed" and state["best"]["trial"] == 2
    before = (tmp_path / "messages.json").read_bytes()
    result = agent.evaluate_selected(tmp_path)
    assert result["scores"]["mae"] < 1e-5
    assert result["evaluated_observations"] == 2 * 3 * 4
    assert (tmp_path / "messages.json").read_bytes() == before
    assert agent.evaluate_selected(tmp_path) == result
