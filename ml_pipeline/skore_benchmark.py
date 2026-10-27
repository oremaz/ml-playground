"""Classical model comparison delegated to skrub and skore."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import check_cv

from ml_pipeline.benchmark_agent import write_json


def registry_estimator(name, **parameters):
    """Return a cloneable estimator without a pipeline-specific wrapper."""
    from ml_pipeline.pipelines_torch.models import MODEL_REGISTRY

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown registered model: {name}")
    if name.startswith(("hf_qlora", "llama_cpp")):
        raise ValueError(f"{name} needs its native training workflow rather than skore cloning")
    adapter = MODEL_REGISTRY[name](**parameters)
    estimator = getattr(adapter, "model", adapter)
    if not isinstance(estimator, BaseEstimator):
        raise ValueError(f"{name} needs its native training workflow rather than skore cloning")
    return clone(estimator)


def evaluate_models(models, X, y, *, splitter=5, groups=None, preprocessing="skrub",
                    n_jobs=1, pos_label=None, output_dir=None):
    """Evaluate named estimators on one materialized set of development splits.

    Integer splitters retain skore's unshuffled (stratified) fold convention.
    Supply an explicit seeded splitter or frozen indices for another protocol.
    """
    try:
        import skore
    except ImportError as error:
        raise ImportError("Install ml-pipeline[evaluation] for skrub/skore comparisons") from error
    if not isinstance(models, dict) or not models or not all(isinstance(name, str) for name in models):
        raise ValueError("models must be a nonempty mapping of names to estimators")
    if preprocessing not in {"skrub", "none"}:
        raise ValueError("preprocessing must be skrub or none")
    if len(X) != len(y):
        raise ValueError("Features and targets must be aligned")
    group_values = np.asarray(groups) if groups is not None else None
    if group_values is not None and (group_values.ndim != 1 or len(group_values) != len(y)
                                     or pd.isna(group_values).any()):
        raise ValueError("Groups must be one-dimensional, aligned and nonmissing")
    estimators = {name: clone(estimator) for name, estimator in models.items()}
    tasks = {is_classifier(estimator) for estimator in estimators.values()}
    if len(tasks) != 1:
        raise ValueError("Compare classifiers or regressors, not both in the same report")
    cv = check_cv(splitter, y, classifier=tasks.pop())
    splits = [(np.asarray(train), np.asarray(valid)) for train, valid in cv.split(X, y, groups)]
    if not splits:
        raise ValueError("At least one development split is required")
    for train, valid in splits:
        for indices in (train, valid):
            if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                    or not len(indices) or indices.min() < 0 or indices.max() >= len(y)
                    or len(np.unique(indices)) != len(indices)):
                raise ValueError("Splits must contain nonempty, unique, in-range integer row indices")
        if np.intersect1d(train, valid).size:
            raise ValueError("Training and validation rows overlap")
        if groups is not None:
            if set(group_values[train]) & set(group_values[valid]):
                raise ValueError("Group splits must be aligned and disjoint")
    if preprocessing == "skrub":
        from skrub import tabular_pipeline

        estimators = {name: tabular_pipeline(estimator) for name, estimator in estimators.items()}
    destination = Path(output_dir) if output_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
        write_json(destination / "protocol.json", {
            "backend": "skore", "preprocessing": preprocessing, "n_jobs": n_jobs,
            "skore_version": skore.__version__, "sklearn_version": sklearn.__version__,
            "models": list(estimators),
            "positive_label": pos_label.item() if isinstance(pos_label, np.generic) else pos_label,
            "splits": [{"train": a.tolist(), "validation": b.tolist()} for a, b in splits],
        })
    candidates = next(iter(estimators.values())) if len(estimators) == 1 else estimators
    report = skore.evaluate(candidates, X, y, splitter=splits, n_jobs=n_jobs, pos_label=pos_label)
    if destination is not None:
        report.metrics.summarize().frame().to_csv(destination / "metrics.csv")
        state = ({name: child.to_dict() for name, child in report.reports_.items()}
                 if isinstance(report, skore.ComparisonReport) else report.to_dict())
        joblib.dump({"report_class": type(report).__name__, "state": state, "n_jobs": n_jobs},
                    destination / "report.joblib")
    return report


def load_report(path):
    """Restore a locally saved skore comparison, including cached results."""
    import skore

    saved = joblib.load(path)
    if saved["report_class"] not in {"ComparisonReport", "CrossValidationReport"}:
        raise ValueError("Unsupported saved skore report type")
    if saved["report_class"] == "ComparisonReport":
        reports = {name: skore.CrossValidationReport.from_dict(state)
                   for name, state in saved["state"].items()}
        return skore.compare(reports, n_jobs=saved["n_jobs"])
    return getattr(skore, saved["report_class"]).from_dict(saved["state"])
