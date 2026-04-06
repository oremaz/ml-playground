"""Shared fixtures for ml_pipeline tests."""
import os
import sys
import shutil
import tempfile

import numpy as np
import pytest
import torch

# Ensure ml_pipeline is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def synthetic_classification_data():
    """Small synthetic binary classification dataset (50 samples, 5 features)."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 5).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    return X, y


@pytest.fixture
def synthetic_multiclass_data():
    """Small synthetic 3-class classification dataset (90 samples, 5 features)."""
    rng = np.random.RandomState(42)
    X = rng.randn(90, 5).astype(np.float32)
    y = np.repeat([0, 1, 2], 30).astype(np.int64)
    rng.shuffle(y)
    return X, y


@pytest.fixture
def synthetic_regression_data():
    """Small synthetic regression dataset (50 samples, 5 features)."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 5).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 1] + rng.randn(50) * 0.1).astype(np.float32)
    return X, y


@pytest.fixture
def tmp_results_dir(tmp_path):
    """Temporary directory for checkpoint/results storage."""
    results_dir = tmp_path / "results" / "test_exp"
    results_dir.mkdir(parents=True)
    return str(results_dir)


@pytest.fixture
def simple_mlp():
    """A tiny 2-layer MLP for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(5, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )


@pytest.fixture
def simple_regression_mlp():
    """A tiny MLP for regression testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(5, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 1),
    )
