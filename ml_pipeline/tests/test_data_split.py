import numpy as np
import pytest

from ml_pipeline.utils.data_split import grouped_holdout_indices


def test_grouped_holdout_preserves_budget_coverage_and_reproducibility():
    groups = ["a", "a", "b", "b", "b", "c", "d", "e", "f"]
    training, holdout = grouped_holdout_indices(groups, 3, seed=42)
    assert len(holdout) == 3
    assert len(training) == 6
    assert set(training) | set(holdout) == set(range(len(groups)))
    assert not set(groups[i] for i in training) & set(groups[i] for i in holdout)
    repeated = grouped_holdout_indices(groups, 3, seed=42)
    for first, second in zip((training, holdout), repeated):
        np.testing.assert_array_equal(first, second)


def test_grouped_holdout_handles_exact_budget_and_rejects_impossible_split():
    for seed in range(10):
        training, holdout = grouped_holdout_indices(["a", "a", "b", "b", "b"], 3, seed=seed)
        assert set(holdout) == {2, 3, 4}
    with pytest.raises(ValueError, match="cannot keep duplicate groups disjoint"):
        grouped_holdout_indices(["a", "a", "b", "b"], 1)
