"""Tests for ComputeScore utility class."""
import numpy as np
import pytest
from pipelines_torch.base import ComputeScore


class TestF1:
    def test_binary(self):
        targets = np.array([0, 0, 1, 1])
        preds = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.3, 0.7]])
        result = ComputeScore.f1(targets, preds, "classification")
        assert result is not None
        assert result == pytest.approx(1.0)

    def test_binary_imperfect(self):
        targets = np.array([0, 0, 1, 1])
        preds = np.array([[0.9, 0.1], [0.2, 0.8], [0.2, 0.8], [0.3, 0.7]])
        result = ComputeScore.f1(targets, preds, "classification")
        assert result is not None
        assert 0.0 < result < 1.0

    def test_multiclass(self):
        targets = np.array([0, 1, 2, 0, 1, 2])
        preds = np.eye(3)[targets]  # perfect predictions
        result = ComputeScore.f1(targets, preds, "classification")
        assert result is not None
        assert result == pytest.approx(1.0)

    def test_1d_binary_probs(self):
        targets = np.array([0, 0, 1, 1])
        preds = np.array([0.1, 0.2, 0.8, 0.9])
        result = ComputeScore.f1(targets, preds, "classification")
        assert result is not None
        assert result == pytest.approx(1.0)

    def test_returns_none_for_regression(self):
        assert ComputeScore.f1(np.array([1.0]), np.array([1.0]), "regression") is None


class TestR2:
    def test_perfect(self):
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        preds = np.array([1.0, 2.0, 3.0, 4.0])
        result = ComputeScore.r2(targets, preds, "regression")
        assert result is not None
        assert result == pytest.approx(1.0)

    def test_imperfect(self):
        targets = np.array([1.0, 2.0, 3.0, 4.0])
        preds = np.array([1.1, 2.2, 2.8, 4.1])
        result = ComputeScore.r2(targets, preds, "regression")
        assert result is not None
        assert 0.0 < result < 1.0

    def test_returns_none_for_classification(self):
        assert ComputeScore.r2(np.array([0, 1]), np.array([0, 1]), "classification") is None


class TestRocAuc:
    def test_binary_2d(self):
        targets = np.array([0, 0, 1, 1])
        preds = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        result = ComputeScore.roc_auc(targets, preds, "classification")
        assert result is not None
        assert result == pytest.approx(1.0)

    def test_binary_1d(self):
        targets = np.array([0, 0, 1, 1])
        preds = np.array([0.1, 0.2, 0.8, 0.9])
        result = ComputeScore.roc_auc(targets, preds, "classification")
        assert result is not None
        assert result == pytest.approx(1.0)

    def test_multiclass(self):
        targets = np.array([0, 1, 2, 0, 1, 2])
        preds = np.eye(3)[targets]
        result = ComputeScore.roc_auc(targets, preds, "classification")
        assert result is not None
        assert result == pytest.approx(1.0)

    def test_returns_none_for_regression(self):
        assert ComputeScore.roc_auc(np.array([1.0]), np.array([1.0]), "regression") is None

    def test_single_class_graceful(self):
        targets = np.array([0, 0, 0, 0])
        preds = np.array([0.1, 0.2, 0.3, 0.4])
        result = ComputeScore.roc_auc(targets, preds, "classification")
        # Single class: sklearn returns nan or raises; either way our wrapper handles it
        assert result is None or (isinstance(result, float) and np.isnan(result))


class TestPrAuc:
    def test_binary_2d(self):
        targets = np.array([0, 0, 1, 1])
        preds = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        result = ComputeScore.pr_auc(targets, preds, "classification")
        assert result is not None
        assert result == pytest.approx(1.0)

    def test_binary_1d(self):
        targets = np.array([0, 0, 1, 1])
        preds = np.array([0.1, 0.2, 0.8, 0.9])
        result = ComputeScore.pr_auc(targets, preds, "classification")
        assert result is not None
        assert result > 0.5

    def test_multiclass(self):
        targets = np.array([0, 1, 2, 0, 1, 2])
        preds = np.eye(3)[targets]
        result = ComputeScore.pr_auc(targets, preds, "classification")
        assert result is not None
        assert result == pytest.approx(1.0)

    def test_returns_none_for_regression(self):
        assert ComputeScore.pr_auc(np.array([1.0]), np.array([1.0]), "regression") is None
