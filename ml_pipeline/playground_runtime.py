"""Training runtime used by saved, generated benchmark scripts."""

import inspect
import hashlib
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score, recall_score,
                             mean_absolute_error, mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import (GroupShuffleSplit, KFold, StratifiedKFold,
                                     train_test_split)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from ml_pipeline.benchmark_agent import read_json, tabular_models, write_json
from ml_pipeline.utils.playground_trace import operation


def make_preprocessor(config):
    if config.get("preprocessing", "standard") == "skrub":
        from skrub import TableVectorizer
        encoder = TableVectorizer()
    elif config.get("preprocessing", "standard") == "standard":
        encoder = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True),
             make_column_selector(dtype_include=np.number)),
            ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                                         OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=32,
                                                       sparse_output=False)),
             make_column_selector(dtype_exclude=np.number)),
        ])
    else:
        raise ValueError("Preprocessing must be standard or skrub")
    return make_pipeline(encoder, SimpleImputer(keep_empty_features=True), StandardScaler())


def prepare_dataset(frame, config):
    if config["task"] == "forecasting":
        from ml_pipeline.forecasting_runtime import validate_forecasting_frame
        values, _ = validate_forecasting_frame(frame, config)
        return pd.DataFrame(index=frame.index), values, None
    target, features = config["target"], config["features"]
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("build_dataset must return a DataFrame with unique columns")
    if not features or len(set(features)) != len(features) or target in features:
        raise ValueError("Select unique features excluding the target")
    missing = set(features + [target]) - set(frame.columns)
    if missing:
        raise ValueError(f"Prepared dataset is missing columns: {sorted(missing)}")
    if len(frame) < 5 or frame[target].isna().any():
        raise ValueError("Need at least five rows and no missing targets. Resolve missing targets explicitly.")
    X = frame[features].replace([np.inf, -np.inf], np.nan).copy()
    for column in X.select_dtypes(exclude=["number"]).columns:
        X[column] = X[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
    labels = None
    if config["task"] == "classification":
        labels = LabelEncoder()
        y = labels.fit_transform(frame[target])
        if len(labels.classes_) < 2:
            raise ValueError("Classification requires at least two classes")
    elif config["task"] == "regression":
        y = pd.to_numeric(frame[target], errors="raise").to_numpy(dtype=np.float32)
        if not np.isfinite(y).all():
            raise ValueError("Regression targets must be finite numbers")
    else:
        raise ValueError("Task must be classification or regression")
    return X, y, labels


def split_indices(frame, y, config):
    if config.get("_split_manifest"):
        path = Path(config["_split_manifest"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != config["_split_sha256"]:
            raise ValueError("Frozen split manifest changed")
        config = {**config, "_frozen_splits": read_json(path)}
    if config.get("_frozen_splits") is not None:
        splits = [(np.asarray(item["train"], dtype=int), np.asarray(item["validation"], dtype=int))
                  for item in config["_frozen_splits"]]
        for train, valid in splits:
            if min(len(train), len(valid)) < 2 or set(train) & set(valid) or min(train.min(), valid.min()) < 0 or max(train.max(), valid.max()) >= len(frame):
                raise ValueError("Invalid frozen development indices")
        return splits
    seed = int(config.get("seed", 42))
    classification = config["task"] == "classification"
    strategy = config.get("split", "holdout")
    label_end_column = config.get("label_end_column")
    if label_end_column and strategy != "temporal":
        raise ValueError("Forward-looking label windows require a temporal split")
    indices = np.arange(len(y))
    fraction = float(config.get("validation_fraction", 0.2))
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    if strategy == "kfold":
        folds = int(config.get("folds", 5))
        if classification and np.bincount(y).min() < folds:
            raise ValueError("Each class needs at least as many rows as folds")
        splitter = (StratifiedKFold if classification else KFold)(folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(indices, y))
    elif strategy in ("group", "temporal"):
        column = config.get("split_column")
        if not column or column == config["target"] or column not in frame or frame[column].isna().any():
            raise ValueError("Group/time splitting requires a non-target column with no missing values")
        if strategy == "group":
            splits = list(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(
                indices, y, groups=frame[column]))
        else:
            times = pd.to_datetime(frame[column], errors="raise", utc=True)
            order = np.argsort(times.to_numpy(), kind="stable")
            boundary = times.iloc[order[min(len(y) - 1, int(len(y) * (1 - fraction)))]]
            training = times < boundary
            if label_end_column:
                if label_end_column not in frame or label_end_column in config["features"]:
                    raise ValueError("Label end column must exist and must not be a feature")
                label_ends = pd.to_datetime(frame[label_end_column], errors="raise", utc=True)
                if label_ends.isna().any() or (label_ends < times).any():
                    raise ValueError("Label end times must be present and no earlier than observation times")
                training &= label_ends < boundary
            splits = [(indices[training.to_numpy()], indices[(times >= boundary).to_numpy()])]
    elif strategy == "holdout":
        splits = [train_test_split(indices, test_size=fraction, random_state=seed,
                                   stratify=y if classification else None)]
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
    for train, validation in splits:
        if len(train) < 2 or len(validation) < 2:
            raise ValueError("Each partition needs at least two rows")
        if classification and set(np.unique(y[train])) != set(np.unique(y)):
            raise ValueError("A training partition is missing a target class. Review the split.")
    return splits


def score_predictions(y, predictions, probabilities, task):
    predictions = np.asarray(predictions).reshape(-1)
    if task == "regression":
        return {"r2": float(r2_score(y, predictions)),
                "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
                "mae": float(mean_absolute_error(y, predictions))}
    scores = {"accuracy": float(accuracy_score(y, predictions)),
              "f1_macro": float(f1_score(y, predictions, average="macro", zero_division=0))}
    if probabilities is not None and np.asarray(probabilities).shape[1] == 2:
        scores.update(precision_positive=float(precision_score(y, predictions, zero_division=0)),
                      recall_positive=float(recall_score(y, predictions, zero_division=0)))
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        if probabilities.shape[1] == 2 and len(np.unique(y)) == 2:
            scores["roc_auc"] = float(roc_auc_score(y, probabilities[:, 1]))
            scores["pr_auc"] = float(average_precision_score(y, probabilities[:, 1]))
        elif probabilities.shape[1] > 2 and len(np.unique(y)) == probabilities.shape[1]:
            scores["roc_auc"] = float(roc_auc_score(y, probabilities, multi_class="ovr", average="macro"))
    return scores


def reference_scores(training_y, evaluation_y, task):
    if task == "regression":
        value = float(np.mean(training_y, dtype=np.float64))
        predictions = np.full(len(evaluation_y), value)
        probabilities, name = None, "training_mean"
    else:
        counts = np.bincount(training_y)
        value = int(counts.argmax())
        predictions = np.full(len(evaluation_y), value)
        probabilities = np.broadcast_to(counts / counts.sum(), (len(evaluation_y), len(counts)))
        name = "training_prior"
    return {"name": name, "prediction": value, "scores": score_predictions(evaluation_y, predictions, probabilities, task),
            "scope": "Constant reference estimated only from this fold's original training labels; no tuning."}


def _model_parameters(factory, config, name, width, classes):
    params = {"input_dim": width, "num_classes": classes, "output_dim": 1}
    seed = int(config.get("seed", 42))
    if name.startswith("catboost"):
        params["random_seed"] = seed
    elif not name.startswith(("mlp", "deep_mlp", "constant_")):
        params["random_state"] = seed
    if name.startswith(("tabicl", "tabfm")):
        params["device"] = config.get("device", "cpu")
    params.update(config.get("model_params", {}).get(name, {}))
    signature = inspect.signature(factory)
    if not any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        params = {key: value for key, value in params.items() if key in signature.parameters}
    return params


def _fit_method(name, augmentation, fold, train_X, validation_X, y, train, validation, config,
                output_dir, reference, trace_workspace, trace_parent, trace_run_id):
    import torch
    from ml_pipeline.data_augmentation.augmentations import AUGMENTATION_REGISTRY
    from ml_pipeline.pipelines_torch.base import GeneralPipeline, GeneralPipelineSklearn
    from ml_pipeline.pipelines_torch.benchmark import set_all_seeds
    from ml_pipeline.pipelines_torch.models import CLASSIFICATION_MODEL_REGISTRY, REGRESSION_MODEL_REGISTRY
    from ml_pipeline.utils.utils import save_model
    task, seed = config["task"], int(config.get("seed", 42))
    registry = CLASSIFICATION_MODEL_REGISTRY if task == "classification" else REGRESSION_MODEL_REGISTRY
    output_dir = Path(output_dir)
    try:
        with operation(trace_workspace, "model.benchmark", {
            "model": name, "augmentation": augmentation, "fold": fold,
            "parameters": config.get("model_params", {}).get(name, {}),
            "seed": seed, "device": config.get("device", "cpu"), "worker_pid": os.getpid()},
            parent_id=trace_parent, run_id=trace_run_id) as trace:
            set_all_seeds(seed)
            model = registry[name](**_model_parameters(registry[name], config, name,
                                                      train_X.shape[1], len(np.unique(y))))
            if config.get("strict_model_params") and hasattr(getattr(model, "model", None), "get_params"):
                estimator = model.model.estimator if isinstance(model.model, MultiOutputRegressor) else model.model
                unknown = set(config.get("model_params", {}).get(name, {})) - set(estimator.get_params())
                if unknown:
                    raise ValueError(f"Unsupported parameters for {name}: {sorted(unknown)}")
            X_fit, y_fit = AUGMENTATION_REGISTRY[augmentation](
                train_X, y[train], random_state=seed, max_factor=float(config.get("max_factor", 2)))
            if isinstance(model, torch.nn.Module):
                batch_size = min(int(config.get("batch_size", 32)), len(X_fit))
                if name.startswith("deep_mlp") and batch_size < 2:
                    raise ValueError("Batch normalization requires batch_size >= 2")
                pipeline = GeneralPipeline(
                    model=model, loss_fn="CrossEntropyLoss" if task == "classification" else "MSELoss",
                    optimizer_cls=torch.optim.Adam,
                    optimizer_params={"lr": float(config.get("learning_rate", 0.001))},
                    task_type=task, device=config.get("device", "cpu"),
                    epochs=int(config.get("epochs", 10)), batch_size=batch_size,
                    use_kfold=False, random_state=seed,
                    use_class_weights=config.get("use_class_weights", True) and not name.startswith("constant_"))
                pipeline._single_fit_internal(X_fit, y_fit, validation_X, y[validation])
                predictions = pipeline.predict(validation_X)
                probabilities = pipeline.predict_proba(validation_X) if task == "classification" else None
            else:
                pipeline = GeneralPipelineSklearn(model=model, task_type=task, use_kfold=False,
                                                  random_state=seed,
                                                  use_class_weights=config.get("use_class_weights", True) and not name.startswith("constant_"))
                pipeline.internal_fit(X_fit, y_fit, validation_X, y[validation])
                predictions = model.predict(validation_X)
                probabilities = model.predict_proba(validation_X) if task == "classification" else None
            scores = score_predictions(y[validation], predictions, probabilities, task)
            primary = config.get("metric")
            if primary and (primary not in scores or not np.isfinite(scores[primary])):
                raise ValueError("The primary metric is undefined on this validation fold")
            scores.update({f"reference_{key}": value for key, value in reference["scores"].items()})
            metadata = {"model_name": name, "augmentation_name": augmentation, "task_type": task,
                        "fold": fold, "seed": seed, "epochs": int(config.get("epochs", 10))}
            with (output_dir / ".checkpoint.lock").open("a") as lock:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX)
                checkpoint = save_model(model, path_start=str(output_dir.resolve()), metadata_core=metadata)
            if config.get("save_predictions"):
                prediction_frame = pd.DataFrame({"row": validation, "y_true": y[validation],
                                                 "y_pred": np.asarray(predictions).reshape(-1)})
                if probabilities is not None:
                    for index in range(np.asarray(probabilities).shape[1]):
                        prediction_frame[f"probability_{index}"] = np.asarray(probabilities)[:, index]
                prediction_frame.to_csv(output_dir / f"predictions_{checkpoint['checkpoint_id']}.csv", index=False)
            result = {"model": name, "augmentation": augmentation, "fold": fold,
                      "checkpoint_id": checkpoint["checkpoint_id"], **scores}
            trace.update(metrics=scores, checkpoint=checkpoint,
                         fit_shape=list(X_fit.shape), validation_rows=len(validation))

        return {"row": result}
    except Exception as exc:
        return {"failure": {"model": name, "augmentation": augmentation, "fold": fold,
                            "error": f"{type(exc).__name__}: {exc}"}}


def benchmark(frame, config, output_dir, preprocessor_factory=make_preprocessor, *,
              trace_workspace=None, trace_parent=None, trace_run_id=None):
    if config["task"] == "forecasting":
        from ml_pipeline.forecasting_runtime import benchmark_forecasting
        return benchmark_forecasting(frame, config, output_dir, trace_workspace=trace_workspace,
                                     trace_parent=trace_parent, trace_run_id=trace_run_id)
    import torch
    from ml_pipeline.data_augmentation.augmentations import AUGMENTATION_REGISTRY
    from ml_pipeline.pipelines_torch.base import GeneralPipeline, GeneralPipelineSklearn
    from ml_pipeline.pipelines_torch.benchmark import set_all_seeds
    from ml_pipeline.pipelines_torch.models import CLASSIFICATION_MODEL_REGISTRY, REGRESSION_MODEL_REGISTRY
    from ml_pipeline.utils.utils import save_model

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "runtime_source.json", {"playground_runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    task = config["task"]
    models, augmentations = config["models"], config.get("augmentations", ["none"])
    if not models or set(models) - set(tabular_models(task)):
        raise ValueError("Select registered tabular models")
    if not augmentations or set(augmentations) - set(AUGMENTATION_REGISTRY):
        raise ValueError("Select registered augmentations")
    if task == "regression" and augmentations != ["none"]:
        raise ValueError("The implemented imbalance augmentations require classification targets")
    X, y, labels = prepare_dataset(frame, config)
    if any(name in ("mixup", "mixup_smote") for name in augmentations) and len(np.unique(y)) != 2:
        raise ValueError("Mixup augmentations in this registry require binary targets")
    splits = split_indices(frame, y, config)
    write_json(output_dir / "splits.json", [{"train": a.tolist(), "validation": b.tolist()} for a, b in splits])
    split_statistics = []
    for train, validation in splits:
        item = {"train_rows": len(train), "validation_rows": len(validation),
                "purged_rows": len(y) - len(train) - len(validation),
                "reference": reference_scores(y[train], y[validation], task)}
        if task == "classification":
            item["train_class_counts"] = {str(int(c)): int(n) for c, n in zip(*np.unique(y[train], return_counts=True))}
            item["validation_class_counts"] = {str(int(c)): int(n) for c, n in zip(*np.unique(y[validation], return_counts=True))}
            item["validation_majority_accuracy"] = float(np.bincount(y[validation]).max() / len(validation))
            if len(labels.classes_) == 2:
                item["validation_positive_prevalence"] = float(np.mean(y[validation] == 1))
        split_statistics.append(item)
    write_json(output_dir / "dataset_summary.json", {"prepared_rows": len(y), "splits": split_statistics})
    write_json(output_dir / "config.json", config)
    if labels is not None:
        write_json(output_dir / "labels.json", {str(i): str(value) for i, value in enumerate(labels.classes_)})
    registry = CLASSIFICATION_MODEL_REGISTRY if task == "classification" else REGRESSION_MODEL_REGISTRY
    rows, failures = [], []
    for fold, (train, validation) in enumerate(splits, 1):
        seed = int(config.get("seed", 42))
        set_all_seeds(seed)
        with operation(trace_workspace, "preprocessing.fit_transform", {
            "fold": fold, "preprocessing": config.get("preprocessing", "standard"),
            "train_rows": len(train), "validation_rows": len(validation)},
            parent_id=trace_parent, run_id=trace_run_id) as trace:
            transformer = preprocessor_factory(config)
            train_X = np.asarray(transformer.fit_transform(X.iloc[train], y[train]), dtype=np.float32)
            validation_X = np.asarray(transformer.transform(X.iloc[validation]), dtype=np.float32)
            if train_X.ndim != 2 or train_X.shape[1] == 0 or not np.isfinite(train_X).all() or not np.isfinite(validation_X).all():
                raise ValueError("Preprocessing must produce a finite, nonempty numeric feature matrix")
            joblib.dump(transformer, output_dir / f"preprocessor_fold_{fold}.joblib")
            trace.update(train_shape=list(train_X.shape), validation_shape=list(validation_X.shape),
                         artifact=str(output_dir / f"preprocessor_fold_{fold}.joblib"))
        workers = int(config.get("model_workers", 1))
        if workers not in range(1, 4):
            raise ValueError("model_workers must be one to three")
        safe = all(name.startswith(("random_forest_", "lightgbm_", "xgboost_", "linear_", "constant_", "catboost_")) for name in models)
        requested_workers = workers
        if config.get("device", "cpu") != "cpu" or not safe:
            workers = 1
        write_json(output_dir / "parallelism.json", {"requested": requested_workers, "effective": workers,
                   "policy": "Parallel CPU sklearn fits; other approved families execute serially."})
        from joblib import Parallel, delayed
        with joblib.parallel_config(backend="loky", inner_max_num_threads=4):
            results = Parallel(n_jobs=workers)(delayed(_fit_method)(
                name, augmentation, fold, train_X, validation_X, y, train, validation, config, output_dir,
                split_statistics[fold - 1]["reference"], trace_workspace, trace_parent, trace_run_id)
                for name in models for augmentation in (["none"] if name.startswith("constant_") else augmentations))
        for result in results:
            if "row" in result:
                rows.append(result["row"])
            else:
                failures.append(result["failure"])
        if rows:
            pd.DataFrame(rows).to_csv(output_dir / "fold_metrics.csv", index=False)
        write_json(output_dir / "errors.json", failures)
    if not rows:
        raise RuntimeError("All methods failed. See errors.json and the execution log.")
    metrics = pd.DataFrame(rows)
    numeric = [c for c in metrics if c not in ("model", "augmentation", "fold", "checkpoint_id")]
    summary = metrics.groupby(["model", "augmentation"])[numeric].mean().reset_index()
    completed = metrics.groupby(["model", "augmentation"]).size()
    summary["completed_folds"] = [int(completed.loc[(row.model, row.augmentation)]) for row in summary.itertuples()]
    summary["expected_folds"] = len(splits)
    summary.to_csv(output_dir / "summary.csv", index=False)
    write_json(output_dir / "completed.json", {"folds": len(splits),
               "summary_sha256": hashlib.sha256((output_dir / "summary.csv").read_bytes()).hexdigest()})
    return summary


def run_experiment(build_dataset, preprocessor_factory=make_preprocessor):
    import hashlib
    config_path = Path(sys.argv[1]).resolve()
    config = read_json(config_path)
    workspace = config_path.parent
    parent = os.getenv("ML_PIPELINE_TRACE_PARENT")
    run_id = os.getenv("ML_PIPELINE_TRACE_RUN_ID", Path(config["output"]).name)
    with operation(workspace, "experiment.run", {"config": config}, parent_id=parent, run_id=run_id) as experiment:
        trace_parent = experiment["operation_id"]
        with operation(workspace, "data.load", {"manifest": "tables.json"},
                       parent_id=trace_parent, run_id=run_id) as trace:
            tables = {}
            for name, item in read_json(workspace / "tables.json").items():
                path = (workspace / item["path"]).resolve()
                if not path.is_relative_to(workspace / "data"):
                    raise ValueError("Dataset path must be inside the workspace data directory")
                with path.open("rb") as uploaded:
                    digest = hashlib.file_digest(uploaded, "sha256").hexdigest()
                if digest != item["sha256"]:
                    raise ValueError(f"Uploaded table changed since this workspace was created: {name}")
                tables[name] = pd.read_csv(path)
            trace["tables"] = {name: list(table.shape) for name, table in tables.items()}
        with operation(workspace, "data.prepare", {"callable": build_dataset.__name__},
                       parent_id=trace_parent, run_id=run_id) as trace:
            frame = build_dataset(tables, config)
            trace.update(shape=list(frame.shape), columns=list(frame.columns))
        output = (workspace / config["output"]).resolve()
        if not output.is_relative_to(workspace / "runs"):
            raise ValueError("Results must be inside the workspace runs directory")
        summary = benchmark(frame, config, output, preprocessor_factory, trace_workspace=workspace,
                            trace_parent=trace_parent, trace_run_id=run_id)
        experiment.update(output=str(output), summary=json.loads(summary.to_json(orient="records")))
        print(summary.to_string(index=False), flush=True)
