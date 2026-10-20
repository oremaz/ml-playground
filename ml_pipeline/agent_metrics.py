"""Inspectable task-specific diagnostics on saved validation predictions."""

import re

import numpy as np


KINDS = ("mae", "rmse", "mean_error", "accuracy", "f1_macro", "class_recall",
         "precision_at_fraction", "recall_at_fraction", "lift_at_fraction")


def validate_metric(definition, frame, config):
    if not isinstance(definition, dict) or set(definition) - {"name", "kind", "where", "fraction", "class_label"}:
        raise ValueError("A diagnostic metric uses name, kind, optional where, fraction and class_label")
    name, kind = definition.get("name"), definition.get("kind")
    if isinstance(name, str) and not name.startswith("custom_"):
        name = "custom_" + name
        definition = {**definition, "name": name}
    if not isinstance(name, str) or not re.fullmatch(r"custom_[a-z][a-z0-9_]{0,40}", name):
        raise ValueError("Use a descriptive custom_ metric name with lowercase letters, digits and underscores")
    if kind not in KINDS:
        raise ValueError(f"Supported diagnostic metric kinds: {KINDS}")
    ranking = kind.endswith("_at_fraction")
    regression = kind in ("mae", "rmse", "mean_error")
    if config["task"] != ("regression" if regression else "classification"):
        raise ValueError("Classification diagnostics require classification; error metrics require tabular regression")
    if kind == "class_recall":
        if "class_label" not in definition or str(definition["class_label"]) not in frame[config["target"]].astype(str).unique():
            raise ValueError("class_recall requires an existing original class_label")
    elif "class_label" in definition:
        raise ValueError("class_label applies only to class_recall")
    if ranking:
        if frame[config["target"]].nunique() != 2:
            raise ValueError("Ranking metrics require binary classification")
        fraction = definition.get("fraction")
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
            raise ValueError("Ranking metrics require fraction in (0, 1]")
    elif "fraction" in definition:
        raise ValueError("fraction applies only to ranking metrics")
    where = definition.get("where", [])
    if not isinstance(where, list) or len(where) > 4:
        raise ValueError("where is a list of at most four conditions, combined with AND")
    allowed = set(config["features"]) | {config.get("split_column", "")}
    for condition in where:
        if not isinstance(condition, dict) or set(condition) != {"column", "op", "value"}:
            raise ValueError("Each condition uses column, op and value")
        if condition["column"] not in allowed or condition["column"] not in frame:
            raise ValueError("Metric filters use original reviewed features or the split column, never targets or predictions")
        if condition["op"] not in ("eq", "in", "ge", "le"):
            raise ValueError("Condition op must be eq, in, ge or le")
        if condition["op"] == "in" and (not isinstance(condition["value"], list) or len(condition["value"]) > 100):
            raise ValueError("An in condition requires a list of at most 100 values")
    metric_mask(frame, where)
    return definition


def metric_mask(frame, conditions):
    mask = np.ones(len(frame), dtype=bool)
    for condition in conditions:
        column, value, op = frame[condition["column"]], condition["value"], condition["op"]
        selected = column.eq(value) if op == "eq" else column.isin(value) if op == "in" else column.ge(value) if op == "ge" else column.le(value)
        mask &= selected.fillna(False).to_numpy(dtype=bool)
    return mask


def score_metric(definition, frame, y, predictions, probabilities=None, label_names=None):
    mask = metric_mask(frame, definition.get("where", []))
    y, predicted = np.asarray(y)[mask], np.asarray(predictions).reshape(-1)[mask]
    result = {"name": definition["name"], "kind": definition["kind"], "rows": int(mask.sum()), "value": None}
    if not len(y):
        return result
    if definition["kind"].endswith("_at_fraction"):
        if probabilities is None or np.asarray(probabilities).shape != (len(frame), 2) or not set(np.unique(y)) <= {0, 1}:
            raise ValueError("Ranking diagnostics require encoded binary labels and two-column probabilities")
        scores = np.asarray(probabilities)[mask, 1]
        if not np.isfinite(scores).all():
            raise ValueError("Ranking probabilities must be finite")
        count = max(1, int(np.ceil(len(y) * definition["fraction"])))
        selected = y[np.argsort(-scores, kind="stable")[:count]]
        precision, positives = float(selected.mean()), int(y.sum())
        result.update(selected_rows=count, positive_rows=positives)
        result["value"] = (precision if definition["kind"] == "precision_at_fraction" else
                           float(selected.sum() / positives) if definition["kind"] == "recall_at_fraction" and positives else
                           precision / float(y.mean()) if definition["kind"] == "lift_at_fraction" and positives else None)
    elif definition["kind"] in ("accuracy", "f1_macro", "class_recall"):
        from sklearn.metrics import f1_score
        if definition["kind"] == "class_recall":
            if label_names is None or str(definition["class_label"]) not in label_names:
                raise ValueError("class_label must match the saved label encoding")
            positive = y == label_names.index(str(definition["class_label"]))
            result["class_rows"] = int(positive.sum())
            result["value"] = float((predicted[positive] == y[positive]).mean()) if positive.any() else None
        else:
            result["value"] = float((predicted == y).mean()) if definition["kind"] == "accuracy" else float(f1_score(
                y, predicted, labels=np.arange(len(label_names)) if label_names else None, average="macro", zero_division=0))
    else:
        residual = predicted.astype(np.float64) - y.astype(np.float64)
        if not np.isfinite(residual).all():
            raise ValueError("Diagnostic errors require finite targets and predictions")
        result["value"] = float(np.abs(residual).mean() if definition["kind"] == "mae" else
                                np.sqrt(np.square(residual).mean()) if definition["kind"] == "rmse" else residual.mean())
    return result


def evaluate_saved_metrics(folder, state, trial):
    import joblib
    import pandas as pd
    from pathlib import Path
    from ml_pipeline.benchmark_agent import read_json, write_json

    definitions = state.get("metric_definitions", [])
    if not definitions or trial["status"] != "completed":
        return []
    folder = Path(folder)
    development = joblib.load(folder / "development.joblib")
    root = folder / f"trial_{trial['number']}"
    output = root / "output"
    label_names = list(read_json(output / "labels.json").values()) if (output / "labels.json").exists() else None
    results = []
    rows = pd.read_csv(output / "fold_metrics.csv")
    for (model, augmentation), group in rows.groupby(["model", "augmentation"], sort=False):
        prediction = pd.concat([pd.read_csv(output / f"predictions_{row.checkpoint_id}.csv")
                                for row in group.itertuples()], ignore_index=True)
        probabilities = [column for column in prediction if column.startswith("probability_")]
        results.append({"model": model, "augmentation": augmentation, "folds": len(group), "metrics": [score_metric(
            definition, development.iloc[prediction.row], prediction.y_true, prediction.y_pred,
            prediction[probabilities].to_numpy() if probabilities else None, label_names) for definition in definitions]})
    trial["metric_results"] = results
    write_json(root / "diagnostic_metrics.json", {"definitions": definitions, "results": results,
               "scope": "Pooled frozen validation predictions; primary selection remains the mean fold score."})
    return results
