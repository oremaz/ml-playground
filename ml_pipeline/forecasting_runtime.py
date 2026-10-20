"""Temporal panel backtests with shared origins and strictly past-only contexts."""

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml_pipeline.forecasting_models import FORECASTING_MODEL_REGISTRY


def validate_forecasting_frame(frame, config):
    time_column, target = config.get("split_column"), config.get("target")
    series_column = config.get("series_column", "")
    if config.get("split") != "temporal" or not time_column or time_column == target:
        raise ValueError("Forecasting requires a temporal split and a non-target time column")
    if config.get("features", []) or config.get("label_end_column"):
        raise ValueError("Univariate forecasting consumes target history, not tabular features or forward-derived labels")
    if config.get("augmentations", ["none"]) != ["none"] or config.get("preprocessing", "none") not in ("none", "standard"):
        raise ValueError("Zero-shot forecasting does not use tabular preprocessing or row resampling")
    if not frame.columns.is_unique or target not in frame or time_column not in frame:
        raise ValueError("Forecasting needs unique columns and existing target/time columns")
    if series_column and (series_column not in frame or series_column in (target, time_column) or frame[series_column].isna().any()):
        raise ValueError("Series IDs must exist, be nonmissing and differ from target/time columns")
    times = pd.to_datetime(frame[time_column], utc=True, errors="raise")
    if times.isna().any():
        raise ValueError("Time values must be present")
    keys = pd.DataFrame({"time": times, "series": frame[series_column] if series_column else "single_series"})
    if keys.duplicated().any():
        raise ValueError("Duplicate timestamps within a series; aggregate or disambiguate explicitly")
    values = pd.to_numeric(frame[target], errors="raise").to_numpy(dtype=np.float32)
    if np.isinf(values).any() or not np.isfinite(values).any():
        raise ValueError("Targets may contain missing observations, but must contain finite values and no infinities")
    for key, default, low, high in [("forecast_horizon", 12, 1, 1024), ("context_length", 512, 8, 8192),
                                     ("min_context", 32, 2, 8192), ("backtest_windows", 3, 1, 100),
                                     ("forecast_stride", config.get("forecast_horizon", 12), 1, 1024)]:
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{key} must be an integer in [{low}, {high}]")
    if config.get("min_context", 32) > config.get("context_length", 512):
        raise ValueError("min_context cannot exceed context_length")
    if config.get("metric", "mae") not in ("mae", "rmse", "smape"):
        raise ValueError("Forecast selection metric must be mae, rmse or smape")
    return values, times


def forecast_windows(frame, config, evaluation_rows):
    values, times = validate_forecasting_frame(frame, config)
    horizon = config.get("forecast_horizon", 12)
    context_length, minimum = config.get("context_length", 512), config.get("min_context", 32)
    stride = config.get("forecast_stride", horizon)
    allowed = np.zeros(len(frame), dtype=bool)
    allowed[np.asarray(evaluation_rows, dtype=int)] = True
    groups = frame.groupby(config["series_column"], sort=False, observed=True).indices if config.get("series_column") else {"single_series": np.arange(len(frame))}
    windows = []
    skipped = []
    for series, positions in groups.items():
        positions = np.asarray(positions)
        order = positions[np.argsort(times.iloc[positions].to_numpy(), kind="stable")]
        eligible = np.flatnonzero(allowed[order])
        if not len(eligible):
            continue
        origins = []
        for origin in range(int(eligible[0]), len(order) - horizon + 1, stride):
            future = order[origin:origin + horizon]
            history = order[max(0, origin - context_length):origin]
            if len(history) < minimum or not allowed[future].all():
                continue
            if not np.isfinite(values[history]).any() or not np.isfinite(values[future]).any():
                continue
            origins.append((history, future))
        if not origins:
            skipped.append(str(series))
        for history, future in origins[-config.get("backtest_windows", 3):]:
            assert times.iloc[history].max() < times.iloc[future].min()
            windows.append({"series_id": str(series), "history_rows": history.tolist(), "rows": future.tolist(),
                            "origin_time": str(times.iloc[history[-1]]),
                            "forecast_times": times.iloc[future].astype(str).tolist(),
                            "context": values[history], "actual": values[future]})
    if not windows:
        raise ValueError("No complete forecast windows: reduce horizon/min_context or provide longer series/evaluation periods")
    return windows, skipped


def forecast_scores(actual, point):
    valid = np.isfinite(actual)
    y, p = actual[valid].astype(np.float64), point[valid].astype(np.float64)
    if not len(y) or not np.isfinite(p).all():
        raise ValueError("No finite forecast targets or invalid predictions")
    denominator = np.abs(y) + np.abs(p)
    return {"mae": float(np.mean(np.abs(y - p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            "smape": float(np.mean(np.divide(200 * np.abs(y - p), denominator,
                                             out=np.zeros_like(y), where=denominator != 0)))}


def benchmark_forecasting(frame, config, output_dir, *, evaluation_rows=None, trace_workspace=None,
                          trace_parent=None, trace_run_id=None):
    from ml_pipeline.benchmark_agent import write_json
    from ml_pipeline.playground_runtime import split_indices
    from ml_pipeline.utils.playground_trace import operation
    from ml_pipeline.pipelines_torch.benchmark import set_all_seeds
    values, _ = validate_forecasting_frame(frame, config)
    if not config["models"] or set(config["models"]) - set(FORECASTING_MODEL_REGISTRY):
        raise ValueError("Choose registered forecasting models")
    if len(set(config["models"])) != len(config["models"]):
        raise ValueError("Choose unique forecasting models")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if evaluation_rows is None:
        train, evaluation_rows = split_indices(frame, values, config)[0]
    else:
        train = np.flatnonzero(~np.isin(np.arange(len(frame)), evaluation_rows))
    windows, skipped = forecast_windows(frame, config, evaluation_rows)
    public_windows = [{key: value for key, value in window.items() if key not in ("context", "actual")} for window in windows]
    write_json(output / "forecast_windows.json", public_windows)
    write_json(output / "splits.json", [{"train": np.asarray(train).tolist(), "validation": np.asarray(evaluation_rows).tolist()}])
    write_json(output / "dataset_summary.json", {"prepared_rows": len(frame), "forecast_windows": len(windows),
        "forecast_horizon": config.get("forecast_horizon", 12), "skipped_series": skipped,
        "evaluated_observations": int(sum(np.isfinite(window["actual"]).sum() for window in windows)),
        "protocol": "Rolling observed history: earlier evaluation observations may enter later contexts; no horizon outcome enters its own context. Steps are ordered observations, not calendar durations.",
        "point_statistic": "median for all foundation models", "splits": [{"train_rows": len(train), "validation_rows": len(evaluation_rows)}]})
    write_json(output / "config.json", config)
    actual = np.stack([window["actual"] for window in windows])
    contexts = [window["context"] for window in windows]
    errors, rows, descriptors = [], [], {}
    for name in config["models"]:
        model = None
        try:
            with operation(trace_workspace, "model.forecast", {"model": name, "windows": len(windows),
                           "config": config}, parent_id=trace_parent, run_id=trace_run_id) as trace:
                set_all_seeds(int(config.get("seed", 42)))
                params = config.get("model_params", {}).get(name, {}).copy()
                factory = FORECASTING_MODEL_REGISTRY[name]
                unknown = set(params) - set(inspect.signature(factory).parameters)
                if unknown:
                    raise ValueError(f"Unsupported forecasting parameters for {name}: {sorted(unknown)}")
                if "device" in params:
                    raise ValueError("Set the shared benchmark device, not a per-model device")
                model = factory(device=config.get("device", "cpu"), **params)
                prediction = model.forecast(contexts, config.get("forecast_horizon", 12))
                scores = forecast_scores(actual, prediction.point)
                if prediction.quantiles is not None:
                    mask = np.isfinite(actual)
                    residual = actual[:, :, None] - prediction.quantiles
                    levels = np.asarray(prediction.levels)
                    scores["pinball_loss"] = float(np.maximum(levels * residual, (levels - 1) * residual)[mask].mean())
                    scores["coverage_80"] = float(((actual >= prediction.quantiles[:, :, 0]) & (actual <= prediction.quantiles[:, :, -1]))[mask].mean())
                descriptor = model.descriptor()
                checkpoint = hashlib.sha256(json.dumps({"name": name, "config": config, "descriptor": descriptor}, sort_keys=True).encode()).hexdigest()[:16]
                descriptors[name] = descriptor
                write_json(output / f"{checkpoint}.forecast.json", {"model": name, "parameters": params, "descriptor": descriptor})
                records = []
                for index, window in enumerate(windows):
                    for step, row in enumerate(window["rows"]):
                        record = {"row": row, "window": index, "series_id": window["series_id"], "origin_time": window["origin_time"],
                                  "timestamp": window["forecast_times"][step], "horizon_step": step + 1,
                                  "y_true": actual[index, step], "y_pred": prediction.point[index, step]}
                        if prediction.quantiles is not None:
                            record.update({f"q_{q:g}": prediction.quantiles[index, step, j] for j, q in enumerate(prediction.levels)})
                        records.append(record)
                pd.DataFrame(records).to_csv(output / f"predictions_{checkpoint}.csv", index=False)
                rows.append({"model": name, "augmentation": "none", "fold": 1, "checkpoint_id": checkpoint, **scores})
                pd.DataFrame(rows).to_csv(output / "fold_metrics.csv", index=False)
                trace.update(metrics=scores, descriptor=descriptor, checkpoint_id=checkpoint)
        except Exception as exc:
            errors.append({"model": name, "augmentation": "none", "fold": 1, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if model is not None:
                model.close()
    write_json(output / "errors.json", errors)
    write_json(output / "forecast_models.json", descriptors)
    if not rows:
        raise RuntimeError("All forecasting methods failed; see errors.json")
    summary = pd.DataFrame(rows).drop(columns=["fold", "checkpoint_id"])
    summary["completed_folds"], summary["expected_folds"] = 1, 1
    summary.to_csv(output / "summary.csv", index=False)
    return summary
