"""Tests for BenchmarkRunner."""
import numpy as np
import pytest
import torch
from unittest.mock import patch
from pipelines_torch.benchmark import BenchmarkRunner, set_all_seeds


class TestSetAllSeeds:
    def test_deterministic(self):
        set_all_seeds(42)
        a = torch.randn(3)
        set_all_seeds(42)
        b = torch.randn(3)
        torch.testing.assert_close(a, b)


class TestEpochNormalization:
    def _configs(self, n=2):
        return [
            {"name": f"model_{i}", "class": torch.nn.Linear, "params": {"in_features": 5, "out_features": 2}}
            for i in range(n)
        ]

    def test_scalar_epochs(self):
        runner = BenchmarkRunner(
            model_configs=self._configs(),
            augmentations=[None],
            epochs=20,
        )
        assert runner._resolve_epochs("model_0", 0) == 20
        assert runner._resolve_epochs("model_1", 1) == 20

    def test_list_epochs(self):
        runner = BenchmarkRunner(
            model_configs=self._configs(),
            augmentations=[None],
            epochs=[10, 30],
        )
        assert runner._resolve_epochs("model_0", 0) == 10
        assert runner._resolve_epochs("model_1", 1) == 30

    def test_dict_epochs(self):
        runner = BenchmarkRunner(
            model_configs=self._configs(),
            augmentations=[None],
            epochs={"model_0": 15, "model_1": 25},
        )
        assert runner._resolve_epochs("model_0", 0) == 15
        assert runner._resolve_epochs("model_1", 1) == 25

    def test_list_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length must match"):
            BenchmarkRunner(
                model_configs=self._configs(2),
                augmentations=[None],
                epochs=[10],
            )

    def test_dict_keys_mismatch_raises(self):
        with pytest.raises(ValueError, match="exactly match"):
            BenchmarkRunner(
                model_configs=self._configs(2),
                augmentations=[None],
                epochs={"model_0": 10, "wrong_name": 20},
            )

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="must be int"):
            BenchmarkRunner(
                model_configs=self._configs(),
                augmentations=[None],
                epochs="bad",
            )


class TestResolveEpochs:
    def test_non_positive_raises(self):
        configs = [{"name": "m", "class": torch.nn.Linear, "params": {"in_features": 5, "out_features": 2}}]
        runner = BenchmarkRunner(
            model_configs=configs,
            augmentations=[None],
            epochs={"m": -1},
        )
        with pytest.raises(ValueError, match="positive int"):
            runner._resolve_epochs("m", 0)


class TestBenchmarkRunSmoke:
    def test_run_tiny_model(self, synthetic_classification_data, tmp_path):
        """Smoke test: run benchmark with 1 tiny model, no augmentation."""
        X, y = synthetic_classification_data

        class TinyMLP(torch.nn.Module):
            def __init__(self, input_dim=5, num_classes=2):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(4, num_classes),
                )
            def forward(self, x):
                return self.net(x)

        model_configs = [
            {"name": "TinyMLP", "class": TinyMLP, "params": {"input_dim": 5, "num_classes": 2}},
        ]

        results_dir = str(tmp_path / "results")
        import os
        os.makedirs(os.path.join(results_dir, "smoke_test"), exist_ok=True)

        with patch("utils.utils.RESULTS_DIR_OUT", results_dir):
            runner = BenchmarkRunner(
                model_configs=model_configs,
                augmentations=[None],
                task_type="classification",
                device="cpu",
                epochs=2,
                batch_size=16,
                use_kfold=False,
                path_start="smoke_test",
                use_mixed_precision=False,
            )
            # Patch save_path to use tmp dir
            runner.save_path = os.path.join(results_dir, "smoke_test")
            runner.run(X, y)
