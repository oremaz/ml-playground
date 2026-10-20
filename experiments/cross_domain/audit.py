"""Replay every fit and compare baseline/winner on final rows after search ends."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

from ml_pipeline.agent_evidence import observed_finding, trial_evidence
from ml_pipeline.agent_loop import apply_features
from ml_pipeline.agent_metrics import score_metric
from ml_pipeline.benchmark_agent import read_json, write_json
from ml_pipeline.playground_runtime import prepare_dataset, score_predictions, reference_scores


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "live_v1"


def replay_candidate(folder, state, trial, row, frame, *, wrapper_output=False):
    output = folder / f"trial_{trial['number']}" / "output"
    prepared = apply_features(frame, state["config"]["features"], trial["recipes"])
    X, y, labels = prepare_dataset(prepared, trial["config"])
    if labels is not None:
        encoding = {value: int(key) for key, value in read_json(output / "labels.json").items()}
        y = np.array([encoding[str(value)] for value in frame[trial["config"]["target"]]])
    transformer = joblib.load(output / "preprocessor_fold_1.joblib")
    model = joblib.load(output / f"{row['checkpoint_id']}.pkl")
    transformed = np.asarray(transformer.transform(X), dtype=np.float32)
    predictor = model
    if wrapper_output:
        from ml_pipeline.pipelines_torch.models import CLASSIFICATION_MODEL_REGISTRY, REGRESSION_MODEL_REGISTRY
        registry = CLASSIFICATION_MODEL_REGISTRY if state["config"]["task"] == "classification" else REGRESSION_MODEL_REGISTRY
        predictor = registry[row["model"]](**trial["config"]["model_params"].get(row["model"], {}))
        predictor.model = model
    predicted = np.asarray(predictor.predict(transformed)).reshape(-1)
    probabilities = predictor.predict_proba(transformed) if state["config"]["task"] == "classification" else None
    return score_predictions(y, predicted, probabilities, state["config"]["task"]), y, predicted, probabilities


def audit(domain):
    record = RUN / domain
    folder = Path(read_json(record / "agent.json")["path"])
    state = read_json(folder / "state.json")
    assert state["status"] == "completed" and len(state["trials"]) == len(state["notes"]) <= 5
    calls = [call for message in state["messages"] for call in message.get("tool_calls", [])]
    replies = [message for message in state["messages"] if message["role"] == "tool"]
    assert [call["id"] for call in calls] == [reply["tool_call_id"] for reply in replies]
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    approvals = [event for event in events if event["event"] == "agent.model_selection"]
    assert len(approvals) == 1 and approvals[0]["data"]["input_source"] == "simulated_user"
    final_event = next(event for event in events if event["event"] == "agent.final_evaluation")
    requests = [event for event in events if event["event"] == "llm.request"]
    assert all(event["timestamp"] < final_event["timestamp"] for event in requests)
    assert all(event["data"]["reasoning_effort"] == "xhigh" for event in requests)
    assert not any("evaluation.json" in json.dumps(event["data"]["messages"]) for event in requests)
    parts = read_json(folder / "partitions.json")
    assert not set(parts["development"]) & set(parts["test"])
    development, test = joblib.load(folder / "development.joblib"), joblib.load(folder / "evaluation.joblib")
    train = development.iloc[parts["train_in_development"]]
    validation = development.iloc[parts["validation_in_development"]]
    column = state["config"]["split_column"]
    if domain == "activity":
        assert not set(train[column]) & set(validation[column])
        assert not set(development[column]) & set(test[column])
    else:
        assert pd.to_datetime(train[column]).max() < pd.to_datetime(validation[column]).min()
        assert pd.to_datetime(development[column]).max() < pd.to_datetime(test[column]).min()
        if state["config"].get("label_end_column"):
            end = state["config"]["label_end_column"]
            assert pd.to_datetime(train[end]).max() < pd.to_datetime(validation[column]).min()
            assert pd.to_datetime(development[end]).max() < pd.to_datetime(test[column]).min()
    if domain == "bikes":
        workspace = Path(read_json(record / "workspace.json")["path"])
        tables = read_json(workspace / "tables.json")
        raw = pd.read_csv(workspace / tables["bikes.csv"]["path"])
        timestamps = pd.to_datetime(raw.dteday) + pd.to_timedelta(raw.hr, unit="h")
        indexed = raw.set_index(timestamps)
        prepared = joblib.load(folder / "prepared.joblib")
        for name in ["weathersit", "temp", "atemp", "hum", "windspeed"]:
            expected = indexed[name].reindex(timestamps - pd.Timedelta(hours=1)).to_numpy()
            np.testing.assert_allclose(prepared[f"{name}_prev"], expected, equal_nan=True)
        assert not set(["temp", "hum", "windspeed", "casual", "registered", "instant"]) & set(state["config"]["features"])
    metric = state["config"]["metric"]
    _, development_y, _ = prepare_dataset(development, state["config"])
    expected_reference = reference_scores(development_y[parts["train_in_development"]],
                                          development_y[parts["validation_in_development"]], state["config"]["task"])
    facts = {fact["id"]: fact for trial in state["trials"] for fact in trial_evidence(trial, metric)}
    for note in state["notes"]:
        assert note["finding"] == observed_finding(state["trials"][note["trial"] - 1], metric)
        for fact in note["measured"]:
            assert fact == facts[fact["id"]]
        if note["decision"] == "continue":
            following = state["trials"][note["trial"]]
            assert following["config"] == note["next_config"] and following["recipes"] == note["next_recipes"]
    replayed = []
    for trial in state["trials"]:
        if trial["status"] != "completed":
            continue
        output = folder / f"trial_{trial['number']}" / "output"
        if domain != "energy":
            runtime = read_json(output / "runtime_source.json")["playground_runtime_sha256"]
            assert runtime == read_json(ROOT / "runtime_update.json")["playground_runtime.py"]["current_sha256"]
            reference = read_json(output / "dataset_summary.json")["splits"][0]["reference"]
            assert reference["name"] == ("training_prior" if domain == "activity" else "training_mean")
        assert read_json(output / "splits.json") == [{"train": parts["train_in_development"], "validation": parts["validation_in_development"]}]
        prepared = apply_features(development, state["config"]["features"], trial["recipes"])
        X, _, _ = prepare_dataset(prepared, trial["config"])
        numeric = X.iloc[parts["train_in_development"]].select_dtypes(include="number")
        transformer = joblib.load(output / "preprocessor_fold_1.joblib")
        if trial["config"].get("preprocessing") == "standard":
            np.testing.assert_allclose(transformer[0].named_transformers_["numeric"].statistics_,
                                       numeric.median().fillna(0).to_numpy())
        for row in pd.read_csv(output / "fold_metrics.csv").to_dict("records"):
            if domain != "energy":
                for name, value in expected_reference["scores"].items():
                    assert abs(row[f"reference_{name}"] - value) < 1e-9
            model = joblib.load(output / f"{row['checkpoint_id']}.pkl")
            fitted = model.estimators_[0] if isinstance(model, MultiOutputRegressor) else model
            for key, value in trial["config"]["model_params"].get(row["model"], {}).items():
                assert fitted.get_params()[key] == value
            scores, y, predicted, probability = replay_candidate(folder, state, trial, row, validation, wrapper_output=True)
            saved = pd.read_csv(output / f"predictions_{row['checkpoint_id']}.csv")
            assert saved.row.tolist() == parts["validation_in_development"]
            np.testing.assert_array_equal(y, saved.y_true)
            np.testing.assert_allclose(predicted, saved.y_pred.to_numpy(dtype=predicted.dtype), rtol=1e-11, atol=1e-10)
            if probability is not None:
                np.testing.assert_allclose(probability, saved[[f"probability_{i}" for i in range(probability.shape[1])]].to_numpy(dtype=probability.dtype), rtol=1e-11, atol=1e-12)
            for name, value in scores.items():
                assert abs(value - row[name]) < 1e-9, (domain, trial["number"], name, value, row[name])
            label_names = list(read_json(output / "labels.json").values()) if probability is not None else None
            for definition in state.get("metric_definitions", []):
                stored_probability = saved[[f"probability_{i}" for i in range(probability.shape[1])]].to_numpy() if probability is not None else None
                result = score_metric(definition, validation, saved.y_true, saved.y_pred, stored_probability, label_names)
                recorded = next(item for item in trial["metric_results"] if item["model"] == row["model"] and item["augmentation"] == row["augmentation"])
                assert next(item for item in recorded["metrics"] if item["name"] == definition["name"])["rows"] == result["rows"]
                actual = next(item for item in recorded["metrics"] if item["name"] == definition["name"])["value"]
                assert actual is None and result["value"] is None or abs(actual - result["value"]) < 1e-9
            replayed.append({"trial": trial["number"], "model": row["model"], "scores": scores})
    first = state["trials"][0]
    select = min if metric in ("rmse", "mae") else max
    baseline = select(first["summary"], key=lambda row: row[metric])
    selected = state["best"]
    final = {}
    for label, number, model_name, augmentation in [("baseline", 1, baseline["model"], baseline["augmentation"]),
                                                 ("selected", selected["trial"], selected["model"], selected["augmentation"])]:
        trial = state["trials"][number - 1]
        output = folder / f"trial_{number}" / "output"
        row = next(row for row in pd.read_csv(output / "fold_metrics.csv").to_dict("records") if row["model"] == model_name and row["augmentation"] == augmentation)
        scores, y, predicted, probability = replay_candidate(folder, state, trial, row, test)
        label_names = list(read_json(output / "labels.json").values()) if probability is not None else None
        diagnostics = [score_metric(definition, test, y, predicted, probability, label_names)
                       for definition in state.get("metric_definitions", [])]
        final[label] = {"trial": number, "model": model_name, "scores": scores, "diagnostic_metrics": diagnostics}
        if label == "selected":
            assert diagnostics == read_json(folder / "evaluation.json").get("diagnostic_metrics", [])
    for name, value in read_json(folder / "evaluation.json")["scores"].items():
        assert abs(final["selected"]["scores"][name] - value) < 1e-9
    result = {"domain": domain, "native_calls_answered": len(calls), "single_simulated_approval": True,
        "trials": len(state["trials"]), "notes": len(state["notes"]), "no_final_feedback_to_llm": True,
        "features": len(state["config"]["features"]), "train_rows": len(train), "validation_rows": len(validation),
        "final_rows": len(test), "selection_metric": metric, "metric_definitions": state.get("metric_definitions", []),
        "validation_baseline": baseline[metric], "validation_selected": selected[metric],
        "final_baseline_vs_selected": final, "fitted_replays": replayed,
        "interpretation": "Descriptive, adaptively selected validation comparison and a single reserved partition; no significance claim."}
    if state["config"]["task"] == "regression":
        target = state["config"]["target"]
        constant = float(train[target].mean())
        result["training_mean_reference"] = {"prediction": constant,
            "validation": score_predictions(validation[target].to_numpy(dtype=np.float32), np.full(len(validation), constant), None, "regression"),
            "final": score_predictions(test[target].to_numpy(dtype=np.float32), np.full(len(test), constant), None, "regression"),
            "purpose": ("Training-only constant. Energy reference was computed after search. "
                        "Bike/activity validation reference was exposed during search; final scores were not.")}
    if state["config"]["task"] == "classification":
        _, test_y, _ = prepare_dataset(test, state["config"])
        result["training_prior_reference"] = {
            "validation": expected_reference,
            "final": reference_scores(development_y[parts["train_in_development"]], test_y, "classification")}
        result["subjects"] = {"training": sorted(map(int, train[column].unique())),
                              "validation": sorted(map(int, validation[column].unique())),
                              "final": sorted(map(int, test[column].unique()))}
    write_json(record / "independent_checks.json", result)
    return result


def main():
    updates = read_json(ROOT / "runtime_update.json") if (ROOT / "runtime_update.json").exists() else {}
    updates.update(read_json(ROOT / "activity_source_update.json")["sources"])
    updates.update(read_json(ROOT / "context_source_update.json")["sources"])
    for name, digest in read_json(RUN / "provenance.json")["sources"].items():
        assert hashlib.sha256((Path("ml_pipeline") / name).read_bytes()).hexdigest() == updates.get(name, {}).get("current_sha256", digest)
        assert hashlib.sha256((RUN / "source" / name).read_bytes()).hexdigest() == digest
    assert all(record["status"] == "completed" for record in read_json(RUN / "status.json"))
    results = [audit(domain) for domain in ["energy", "bikes", "activity"]]
    write_json(ROOT / "independent_checks.json", results)
    print(json.dumps([{key: value for key, value in result.items() if key != "fitted_replays"} for result in results], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    RUN = args.run.resolve()
    main()
