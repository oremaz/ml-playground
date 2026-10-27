"""Validation-only trial diagnostics and plots."""

import joblib
import numpy as np
import pandas as pd

from ml_pipeline.benchmark_agent import read_json, write_json
from ml_pipeline.playground_runtime import prepare_dataset, score_predictions
from ml_pipeline.agent_trial_config import _best


def _trial_diagnostics(folder, state, number):
    number = number or len(state["trials"])
    if number < 1 or number > len(state["trials"]):
        raise ValueError("No such trial")
    trial = state["trials"][number - 1]
    if trial["status"] != "completed":
        return {"trial": number, "error": trial.get("error", "Trial was interrupted; start a new session")}
    root = folder / f"trial_{number}" / "output"
    if state["config"]["task"] == "forecasting":
        diagnostics = []
        for row in pd.read_csv(root / "fold_metrics.csv").to_dict("records"):
            prediction = pd.read_csv(root / f"predictions_{row['checkpoint_id']}.csv")
            from ml_pipeline.forecasting_runtime import forecast_scores
            by_step = [{"horizon_step": int(step), **forecast_scores(group.y_true.to_numpy(), group.y_pred.to_numpy())}
                       for step, group in prediction.groupby("horizon_step") if group.y_true.notna().any()]
            diagnostics.append({"model": row["model"], "augmentation": "none", "scores_by_horizon": by_step,
                                "windows": int(prediction.window.nunique())})
        return {"trial": number, "diagnostics": diagnostics,
                "caution": "Zero-shot forecasts have no task-fitted training score. Origins and horizons are shared; observed earlier evaluation history may enter later contexts."}
    if state["config"].get("validation_protocol") == "robust":
        from ml_pipeline.agent_diagnostics import diagnose
        return diagnose(folder, state, trial)
    cached = root.parent / "diagnostics.json"
    if cached.exists():
        return read_json(cached)
    dev = joblib.load(folder / "development.joblib")
    prepared = joblib.load(root.parent / "input.joblib")
    X, y_train_source, _ = prepare_dataset(prepared, trial["config"])
    train = read_json(folder / "partitions.json")["train_in_development"]
    rows = pd.read_csv(root / "fold_metrics.csv")
    diagnostics = []
    for row in rows.to_dict("records"):
        item = {"model": row["model"], "augmentation": row["augmentation"]}
        predictions = pd.read_csv(root / f"predictions_{row['checkpoint_id']}.csv")
        if state["config"]["task"] == "classification":
            from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
            labels = list(read_json(root / "labels.json").values())
            values = np.arange(len(labels))
            precision, recall, f1, support = precision_recall_fscore_support(
                predictions.y_true, predictions.y_pred, labels=values, zero_division=0)
            item["per_class"] = [{"label": label, "precision": float(p), "recall": float(r), "f1": float(f), "rows": int(n)}
                                 for label, p, r, f, n in zip(labels, precision, recall, f1, support)]
            item["confusion_matrix"] = {"label_order": labels, "rows_are_true": True,
                                        "counts": confusion_matrix(predictions.y_true, predictions.y_pred, labels=values).tolist()}
        if "probability_1" in predictions and len(read_json(root / "labels.json")) == 2:
            y, p = predictions.y_true.to_numpy(), predictions.probability_1.to_numpy()
            order = np.argsort(-p, kind="stable")[:max(1, int(np.ceil(len(y) * .1)))]
            item.update(positive_prevalence=float(np.mean(y)),
                        precision_top_10_percent=float(np.mean(y[order])),
                        recall_top_10_percent=float(y[order].sum() / max(1, y.sum())),
                        predicted_positive_at_default=int(predictions.y_pred.sum()))
            predicted = predictions.y_pred.to_numpy()
            item["confusion_counts"] = {"true_positive": int(((y == 1) & (predicted == 1)).sum()),
                                        "false_positive": int(((y == 0) & (predicted == 1)).sum()),
                                        "false_negative": int(((y == 1) & (predicted == 0)).sum()),
                                        "true_negative": int(((y == 0) & (predicted == 0)).sum())}
            column = state["config"].get("split_column")
            if state["config"].get("split") == "temporal":
                from sklearn.metrics import average_precision_score, roc_auc_score
                years = pd.to_datetime(dev.iloc[predictions.row][column]).dt.year.to_numpy()
                item["validation_by_year"] = [{"year": int(year), "rows": int((years == year).sum()),
                    "prevalence": float(y[years == year].mean()),
                    "pr_auc": float(average_precision_score(y[years == year], p[years == year])),
                    "pr_auc_over_prevalence": float(average_precision_score(y[years == year], p[years == year]) / y[years == year].mean()),
                    "roc_auc": float(roc_auc_score(y[years == year], p[years == year]))}
                    for year in np.unique(years) if len(np.unique(y[years == year])) == 2]
        checkpoint = root / f"{row['checkpoint_id']}.pkl"
        if checkpoint.exists():
            model = joblib.load(checkpoint)
            transformer = joblib.load(root / f"preprocessor_fold_{row['fold']}.joblib")
            training_X = np.asarray(transformer.transform(X.iloc[train]), dtype=np.float32)
            probabilities = model.predict_proba(training_X) if state["config"]["task"] == "classification" else None
            item["training_scores"] = score_predictions(y_train_source[train], model.predict(training_X),
                                                         probabilities, state["config"]["task"])
            item["validation_scores"] = {key: row[key] for key in item["training_scores"] if key in row}
            item["training_rows"] = len(train)
            from sklearn.multioutput import MultiOutputRegressor
            if isinstance(model, MultiOutputRegressor):
                model = model.estimators_[0]
            if hasattr(model, "feature_importances_"):
                names = transformer.get_feature_names_out()
                item["importance_type"] = getattr(model, "importance_type", "mean_decrease_impurity")
                item["top_feature_importances"] = sorted(
                    [{"feature": str(n), "importance": float(v)} for n, v in zip(names, model.feature_importances_)],
                    key=lambda x: x["importance"], reverse=True)[:8]
        diagnostics.append(item)
    result = {"trial": number, "diagnostics": diagnostics,
            "caution": "Predicted positives include false positives; use confusion_counts for correct detections. "
                       "Training scores are in-sample on original inner-training rows, not augmented rows or a new validation set. "
                       "A training/validation gap is descriptive and may include temporal distribution shift. "
                       "Default-threshold recall does not establish ranking quality or probability calibration. "
                       "Top-decile precision is validation-only; importances are associations, not causal explanations."}
    write_json(cached, result)
    return result


def visualize(folder, state):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    best_scores = []
    for trial in state["trials"]:
        partial = {**state, "trials": state["trials"][:trial["number"]]}
        incumbent = _best(partial)
        if incumbent:
            best_scores.append((trial["number"], incumbent[state["config"]["metric"]]))
        if trial["status"] != "completed":
            continue
        root = folder / f"trial_{trial['number']}" / "output"
        for row in pd.read_csv(root / "fold_metrics.csv").to_dict("records"):
            p = pd.read_csv(root / f"predictions_{row['checkpoint_id']}.csv")
            if "probability_1" in p and len(read_json(root / "labels.json")) == 2:
                precision, recall, _ = precision_recall_curve(p.y_true, p.probability_1)
                axes[0].plot(recall, precision, label=f"{trial['number']}: {row['model']} / {row['augmentation']}", alpha=.75)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Development validation only", ylim=(0, 1))
    description = "Validation PR curves and incumbent trajectory."
    if state["best"] and state["config"]["task"] != "forecasting":
        best = state["best"]
        root = folder / f"trial_{best['trial']}" / "output"
        rows = pd.read_csv(root / "fold_metrics.csv")
        selected_rows = rows[(rows.model == best["model"]) & (rows.augmentation == best["augmentation"])]
        selected = (selected_rows.sort_values("fold") if "fold" in selected_rows else selected_rows).iloc[-1]
        prediction = pd.read_csv(root / f"predictions_{selected.checkpoint_id}.csv")
        if state["config"]["task"] == "regression":
            axes[0].clear()
            axes[0].scatter(prediction.y_true, prediction.y_pred, s=5, alpha=.3)
            low = min(prediction.y_true.min(), prediction.y_pred.min())
            high = max(prediction.y_true.max(), prediction.y_pred.max())
            axes[0].plot([low, high], [low, high], color="black", linestyle="--")
            axes[0].set(xlabel="Observed", ylabel="Predicted", title="Selected candidate: validation only")
            description = "Validation predictions and incumbent trajectory."
        elif len(read_json(root / "labels.json")) > 2:
            axes[0].clear()
            labels = read_json(root / "labels.json")
            order = sorted(labels, key=int)
            ConfusionMatrixDisplay.from_predictions(prediction.y_true, prediction.y_pred,
                labels=[int(key) for key in order], display_labels=[labels[key] for key in order],
                ax=axes[0], colorbar=False, xticks_rotation=45)
            axes[0].set_title("Selected candidate: validation only")
            description = "Validation confusion matrix and incumbent trajectory."
    if state["config"]["task"] == "forecasting":
        description = "Forecast error by horizon and incumbent trajectory."
        axes[0].set(xlabel="Horizon step", ylabel=state["config"]["metric"], title="Forecast error by horizon", ylim=(0, None))
        for trial in state["trials"]:
            if trial["status"] != "completed":
                continue
            for item in _trial_diagnostics(folder, state, trial["number"])["diagnostics"]:
                points = item["scores_by_horizon"]
                axes[0].plot([p["horizon_step"] for p in points], [p[state["config"]["metric"]] for p in points],
                             label=f"{trial['number']}: {item['model']}")
    if axes[0].get_legend_handles_labels()[1]:
        axes[0].legend(fontsize=6)
    if best_scores:
        axes[1].plot(*zip(*best_scores), marker="o")
    axes[1].set(xlabel="Trial", ylabel=state["config"]["metric"], title="Best validation candidate retained")
    fig.tight_layout()
    fig.savefig(folder / "validation.png", dpi=140)
    plt.close(fig)
    return {"artifact": "validation.png", "description": description + " No final test results."}

