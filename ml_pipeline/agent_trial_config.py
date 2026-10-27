"""Reviewed trial constraints, selection and execution comparisons."""

from copy import deepcopy

import joblib
import numpy as np
import pandas as pd

from ml_pipeline.benchmark_agent import read_json


def model_capabilities(config, models):
    if config["task"] == "forecasting":
        import inspect
        from ml_pipeline.forecasting_models import FORECASTING_MODEL_REGISTRY, forecasting_catalog
        catalog = forecasting_catalog()
        return {name: {**catalog[name], "pipeline_balancing_requested": False, "automatic_balancing_enabled": False,
                       "supported_parameter_defaults": {key: parameter.default for key, parameter in inspect.signature(
                           FORECASTING_MODEL_REGISTRY[name]).parameters.items() if parameter.default is not inspect.Parameter.empty},
                       "parameter_scope": "Only registered inference parameters; pretrained revision, horizon, history and origins are fixed."}
                for name in models}
    from sklearn.multioutput import MultiOutputRegressor
    from ml_pipeline.pipelines_torch.models import CLASSIFICATION_MODEL_REGISTRY, REGRESSION_MODEL_REGISTRY
    from ml_pipeline.utils.playground_trace import _serializable
    registry = CLASSIFICATION_MODEL_REGISTRY if config["task"] == "classification" else REGRESSION_MODEL_REGISTRY
    capabilities = {}
    for name in models:
        requested = config["task"] == "classification" and config.get("use_class_weights", True)
        item = {"pipeline_balancing_requested": requested, "automatic_balancing_enabled": None}
        if name.startswith(("random_forest_", "lightgbm_", "xgboost_", "linear_", "constant_", "kumo_")):
            item["automatic_balancing_enabled"] = requested and name.startswith(("lightgbm_", "xgboost_"))
            wrapper = registry[name]()
            estimator = getattr(wrapper, "model", wrapper)
            if isinstance(estimator, MultiOutputRegressor):
                estimator = estimator.estimator
            item["supported_parameter_defaults"] = _serializable(estimator.get_params())
            item["parameter_scope"] = "This registry wrapper exposes these parameter names only; booster aliases absent from this list are rejected."
            item["automatic_balancing"] = ("Balanced training sample weights for classification. Do not also set class_weight or scale_pos_weight unless deliberately compounding weights. Set use_class_weights=false to test without automatic balancing."
                if name.startswith(("lightgbm_", "xgboost_")) else
                "This wrapper does not expose automatic class_weight setting; use the supported class_weight parameter explicitly to compare balancing.")
            if name.startswith("kumo_"):
                item["automatic_balancing"] = "Frozen in-context model; class_weight and sample_weight are unsupported."
        else:
            item["parameter_scope"] = "Use this model's registered constructor parameters; optional dependencies/weights must already be available."
        capabilities[name] = item
    return capabilities


def apply_features(frame, original, recipes):
    frame = frame.copy()
    if not isinstance(recipes, list) or len(recipes) > 8:
        raise ValueError("Provide at most eight rowwise feature recipes")
    for recipe in recipes:
        name, op, left = recipe["name"], recipe["op"], recipe["left"]
        if not isinstance(name, str) or not name or len(name) > 80 or name in frame:
            raise ValueError("Feature names must be new, nonempty strings up to 80 characters")
        if left not in original or not pd.api.types.is_numeric_dtype(frame[left]):
            raise ValueError("Feature inputs must be reviewed numeric features")
        a = frame[left]
        if op in ("ratio", "product", "difference"):
            right = recipe["right"]
            if right not in original or not pd.api.types.is_numeric_dtype(frame[right]):
                raise ValueError("Right input must be a reviewed numeric feature")
            b = frame[right]
            values = a / b.replace(0, np.nan) if op == "ratio" else a * b if op == "product" else a - b
        elif op in ("abs", "square"):
            values = a.abs() if op == "abs" else a ** 2
        else:
            raise ValueError("Unknown rowwise operation")
        frame[name] = values.replace([np.inf, -np.inf], np.nan)
    return frame


def _trial_config(state, overrides):
    if not isinstance(overrides, dict) or set(overrides) - {"models", "model_params", "drop_parameters", "preprocessing", "augmentations", "features", "use_class_weights"}:
        raise ValueError("Only approved models, parameters, preprocessing, augmentations and features may change")
    if not state["trials"] and (overrides or state["recipes"]):
        raise ValueError("First run must use the reviewed baseline: overrides_json='{}', no feature recipes")
    config = deepcopy(state["config"])
    eligible_models = set(state["approved_models"])
    if config.get("validation_protocol") == "robust" and config["task"] != "forecasting":
        from ml_pipeline.agent_validation import control_model
        eligible_models.add(control_model(config["task"]))
    config["models"] = state["approved_models"]
    config["features"] += [r["name"] for r in state["recipes"]]
    drops = overrides.get("drop_parameters", {})
    if not isinstance(drops, dict) or set(drops) - set(state["approved_models"]):
        raise ValueError("Only inherited parameters of approved models may be removed")
    for name, keys in drops.items():
        if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
            raise ValueError("drop_parameters maps each model to a list of parameter names")
        for key in keys:
            if config["task"] == "forecasting" and key in ("model_id", "revision"):
                raise ValueError("The approved forecasting checkpoint and revision cannot be removed")
            config.setdefault("model_params", {}).setdefault(name, {}).pop(key, None)
    parameters = overrides.get("model_params", {})
    if not isinstance(parameters, dict) or set(parameters) - eligible_models:
        raise ValueError("Parameters must belong to approved model families")
    for name, params in parameters.items():
        if not isinstance(params, dict):
            raise ValueError("Model parameters must be objects")
        if name.startswith("constant_") and params:
            raise ValueError("The training-mean/prior reference is untuned")
        if any(key in params for key in ("random_state", "random_seed", "device", "device_type", "task_type", "nthread")):
            raise ValueError("Seed, device and thread settings are controlled by the experiment")
        if config["task"] == "forecasting" and set(params) & {"model_id", "revision"}:
            raise ValueError("Use the approved forecasting checkpoint and its pinned revision")
        if "n_jobs" in params and params["n_jobs"] not in range(1, 5):
            raise ValueError("n_jobs must be 1 to 4")
        if "n_estimators" in params and params["n_estimators"] not in range(1, 401):
            raise ValueError("n_estimators must be 1 to 400")
        config.setdefault("model_params", {}).setdefault(name, {}).update(params)
    config.update({k: v for k, v in overrides.items() if k not in ("model_params", "drop_parameters")})
    if not isinstance(config.get("use_class_weights", True), bool):
        raise ValueError("use_class_weights must be boolean")
    if not config["models"] or set(config["models"]) - eligible_models:
        raise ValueError("Benchmark models require the user's approval")
    allowed = set(state["config"]["features"]) | {r["name"] for r in state["recipes"]}
    if (not config["features"] and config["task"] != "forecasting") or set(config["features"]) - allowed:
        raise ValueError("Only reviewed or explicitly prepared features may be selected")
    if config["task"] == "forecasting":
        if config["features"] or state["recipes"]:
            raise ValueError("Univariate forecasting uses target history, not rowwise tabular features")
        for name, revision in state.get("forecast_revisions", {}).items():
            config.setdefault("model_params", {}).setdefault(name, {})["revision"] = revision
    if config.get("validation_protocol") == "robust" and config["task"] != "forecasting":
        from ml_pipeline.agent_validation import control_model
        control = control_model(config["task"])
        if control not in config["models"]:
            config["models"] = [*config["models"], control]
        config.setdefault("model_params", {})[control] = {}
    config["save_predictions"] = True
    config["strict_model_params"] = True
    config.setdefault("use_class_weights", True)
    return config


def _best(state):
    metric = state["config"]["metric"]
    candidates = [{"trial": trial["number"], **row} for trial in state["trials"] for row in trial.get("summary", [])
                  if row.get(metric) is not None and np.isfinite(row[metric]) and row["completed_folds"] == row["expected_folds"]]
    if not candidates:
        return None
    return (min if metric in ("mae", "rmse", "smape", "mase") else max)(candidates, key=lambda row: row[metric])


def _experiment_settings(config, recipes):
    return {"models": config["models"], "model_params": {
                name: config.get("model_params", {}).get(name, {}) for name in config["models"]},
            "features": config["features"], "feature_recipes": recipes,
            "preprocessing": config.get("preprocessing", "standard"),
            "augmentations": config.get("augmentations", ["none"]),
            "use_class_weights": config.get("use_class_weights", True)}


def _setting_changes(before, after, prefix=""):
    changes = []
    for key in sorted(set(before) | set(after)):
        path = f"{prefix}.{key}" if prefix else key
        if key in before and key in after and isinstance(before[key], dict) and isinstance(after[key], dict):
            changes.extend(_setting_changes(before[key], after[key], path))
        elif before.get(key) != after.get(key) or (key in before) != (key in after):
            changes.append({"setting": path, "before": before.get(key, "(estimator default)"),
                            "after": after.get(key, "(estimator default)")})
    return changes


def _execution_feedback(folder, state, trial, incumbent):
    from ml_pipeline.utils.playground_trace import _serializable
    baseline = state["trials"][0]
    reference = state["trials"][incumbent["trial"] - 1] if incumbent else baseline
    current = _experiment_settings(trial["config"], trial["recipes"])
    capabilities = model_capabilities(trial["config"], trial["config"]["models"])
    balancing = {name: {key: value for key, value in item.items() if key in
                 ("pipeline_balancing_requested", "automatic_balancing_enabled", "automatic_balancing")}
                 for name, item in capabilities.items()}
    fitted = []
    output = folder / f"trial_{trial['number']}" / "output"
    if trial["config"]["task"] == "forecasting" and (output / "forecast_models.json").exists():
        fitted = [{"model": name, "augmentation": "none", "fold": 1, "parameters": descriptor}
                  for name, descriptor in read_json(output / "forecast_models.json").items()]
    if (output / "fold_metrics.csv").exists():
        for row in pd.read_csv(output / "fold_metrics.csv").to_dict("records"):
            checkpoint = output / f"{row['checkpoint_id']}.pkl"
            if checkpoint.exists():
                model = joblib.load(checkpoint)
                from sklearn.multioutput import MultiOutputRegressor
                if isinstance(model, MultiOutputRegressor):
                    model = model.estimators_[0]
                if hasattr(model, "get_params"):
                    params = model.get_params()
                    requested = trial["config"].get("model_params", {}).get(row["model"], {})
                    names = set(requested) | {"n_estimators", "max_depth", "num_leaves", "learning_rate",
                        "min_samples_leaf", "min_child_samples", "max_features", "reg_alpha", "reg_lambda", "class_weight"}
                    fitted.append({"model": row["model"], "augmentation": row["augmentation"],
                                   "fold": row["fold"], "parameters": _serializable({k: params[k] for k in sorted(names) if k in params})})
    warnings = []
    for name, item in balancing.items():
        if item["pipeline_balancing_requested"] and item["automatic_balancing_enabled"] is False:
            warnings.append(f"{name}: pipeline balancing has no automatic effect; only explicit model class_weight applies.")
    return {"inheritance": "Every override resolves against the reviewed baseline, not the previous trial.",
            "resolved_settings": current, "changes_from_baseline": _setting_changes(
                _experiment_settings(baseline["config"], baseline["recipes"]), current),
            "reference_incumbent_trial": reference["number"],
            "changes_from_incumbent": _setting_changes(_experiment_settings(reference["config"], reference["recipes"]), current),
            "balancing": balancing, "fitted_parameters": fitted, "warnings": warnings}


def _trial_comparison(state, trial, incumbent):
    metric = state["config"]["metric"]
    baseline = {(row["model"], row["augmentation"]): row for row in state["trials"][0].get("summary", [])}
    candidates = []
    for row in trial.get("summary", []):
        value = row.get(metric)
        if value is None or row["completed_folds"] != row["expected_folds"]:
            continue
        first = baseline.get((row["model"], row["augmentation"]), {}).get(metric)
        candidates.append({"model": row["model"], "augmentation": row["augmentation"], "score": value,
            "delta_from_same_method_baseline": value - first if first is not None else None,
            "delta_from_previous_incumbent": value - incumbent[metric] if incumbent else None})
    return {"metric": metric, "direction": "minimize" if metric in ("mae", "rmse", "smape", "mase") else "maximize",
            "incumbent_before": incumbent, "incumbent_after": state["best"], "candidates": candidates,
            "improved_incumbent": bool(incumbent and state["best"] and state["best"]["trial"] == trial["number"]),
            "caution": "Validation differences are descriptive; this single split does not establish statistical significance or cause."}

