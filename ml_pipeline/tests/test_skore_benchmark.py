"""Real skore comparisons preserve folds, preprocessing boundaries and cached fits."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold

pytest.importorskip("skore")
from skrub import tabular_pipeline

from ml_pipeline.skore_benchmark import evaluate_models, load_report, registry_estimator


class CountingRegressor(RegressorMixin, BaseEstimator):
    fits = 0

    def fit(self, X, y):
        type(self).fits += 1
        self.mean_ = np.mean(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_shared_generator_splits_and_train_fitted_skrub_match_independent_fits():
    X = pd.DataFrame({"number": [0., 1., np.nan, 3., 4., 5., 6., 1000.],
                      "category": ["a", "a", "b", "b", "a", "b", "c", "unseen"]})
    y = np.array([0., 1., 1., 3., 4., 5., 6., 20.])
    splits = list(KFold(2, shuffle=True, random_state=19).split(X))
    report = evaluate_models({"Ridge": Ridge(), "Control": Ridge(alpha=2)}, X, y,
                             splitter=iter(splits), n_jobs=1)
    predictions = report.reports_["Ridge"].get_predictions(data_source="test")
    for (train, valid), actual in zip(splits, predictions):
        expected = tabular_pipeline(Ridge()).fit(X.iloc[train], y[train]).predict(X.iloc[valid])
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("comparisons", [1, 2])
def test_saved_report_reloads_without_fitting_again(tmp_path, comparisons):
    X, y = np.arange(48).reshape(24, 2), np.arange(24, dtype=float)
    CountingRegressor.fits = 0
    models = {f"Mean{i}": CountingRegressor() for i in range(comparisons)}
    report = evaluate_models(models, X, y, splitter=3,
                             preprocessing="none", output_dir=tmp_path / "run")
    summary = report.metrics.summarize().frame()
    restored = load_report(tmp_path / "run/report.joblib")
    pd.testing.assert_frame_equal(restored.metrics.summarize().frame(), summary)
    assert CountingRegressor.fits == 3 * comparisons
    assert (tmp_path / "run/protocol.json").is_file()
    assert (tmp_path / "run/metrics.csv").is_file()


def test_group_disjointness_and_overlap_are_enforced():
    X, y, groups = np.arange(24).reshape(12, 2), np.arange(12), np.repeat([0, 1, 2], 4)
    report = evaluate_models({"Ridge": Ridge()}, X, y, splitter=GroupKFold(3),
                             groups=groups, preprocessing="none")
    assert len(report.reports_) == 3
    with pytest.raises(ValueError, match="Group splits"):
        evaluate_models({"Ridge": Ridge()}, X, y,
                        splitter=[(np.array([0, 4]), np.array([1, 8]))], groups=groups)
    with pytest.raises(ValueError, match="overlap"):
        evaluate_models({"Ridge": Ridge()}, X, y,
                        splitter=[(np.array([0, 1]), np.array([1, 2]))])
    with pytest.raises(ValueError, match="nonmissing"):
        evaluate_models({"Ridge": Ridge()}, X, y, splitter=GroupKFold(3), groups=[np.nan] * len(y))


def test_registry_conversion_keeps_requested_parameters_and_kumo_clones():
    model = registry_estimator("linear_regressor", alpha=3)
    assert isinstance(model, Ridge) and model.alpha == 3
    kumo = registry_estimator("kumo_classifier", size="medium", num_estimators=2)
    assert clone(kumo).size == "medium"
    with pytest.raises(ValueError, match="native training"):
        registry_estimator("hf_qlora_classifier")
