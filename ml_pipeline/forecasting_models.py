"""Lazy zero-shot forecasting adapters with a common median/quantile contract."""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np


QUANTILES = (0.1, 0.5, 0.9)


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _padded(contexts):
    width = max(map(len, contexts))
    values = np.full((len(contexts), width), np.nan, dtype=np.float32)
    for index, context in enumerate(contexts):
        values[index, -len(context):] = context
    return values


@dataclass
class Forecast:
    point: np.ndarray
    quantiles: np.ndarray | None = None
    levels: tuple = QUANTILES

    def validate(self, batch, horizon):
        if self.point.shape != (batch, horizon) or not np.isfinite(self.point).all():
            raise ValueError(f"Invalid forecast shape/values: expected {(batch, horizon)}, got {self.point.shape}")
        if self.quantiles is not None:
            if self.quantiles.shape != (batch, horizon, len(self.levels)) or not np.isfinite(self.quantiles).all():
                raise ValueError("Invalid quantile forecast shape/values")
            if np.any(np.diff(self.quantiles, axis=-1) < -1e-5):
                raise ValueError("Forecast quantiles cross; refusing invalid uncertainty intervals")
        return self


class NaiveForecaster:
    """Repeat the latest observed value, or the latest observed seasonal cycle."""

    def __init__(self, season_length=1, device="cpu"):
        if isinstance(season_length, bool) or not isinstance(season_length, int) or season_length < 1:
            raise ValueError("season_length must be a positive integer")
        self.season_length = season_length

    def forecast(self, contexts, horizon):
        points = []
        for context in contexts:
            values = np.asarray(context, dtype=np.float32)
            if len(values) < self.season_length or not np.isfinite(values).any():
                raise ValueError("Not enough observed history for the seasonal naive forecast")
            # Fill only from earlier observations; never interpolate from future values.
            indices = np.maximum.accumulate(np.where(np.isfinite(values), np.arange(len(values)), -1))
            filled = values[np.maximum(indices, 0)]
            pattern = filled[-self.season_length:]
            if not np.isfinite(pattern).all():
                raise ValueError("Seasonal cycle contains unobserved leading history")
            points.append(np.resize(pattern, horizon))
        return Forecast(np.asarray(points)).validate(len(contexts), horizon)

    def descriptor(self):
        return {"family": "naive", "season_length": self.season_length, "point_statistic": "last observed seasonal cycle"}

    def close(self):
        pass


class FoundationForecaster:
    model_id = ""
    family = ""

    def __init__(self, model_id=None, revision=None, device="cpu", local_files_only=False, batch_size=16):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 256:
            raise ValueError("batch_size must be an integer from 1 to 256")
        if device != "cpu" and not str(device).startswith("cuda"):
            raise ValueError("Forecasting device must be cpu or cuda")
        self.model_id = model_id or type(self).model_id
        self.revision, self.device = revision, device
        self.local_files_only, self.batch_size = local_files_only, batch_size
        self.model = None
        self.resolved_revision = None

    def _checkpoint(self):
        if Path(self.model_id).is_dir():
            return self.model_id
        from huggingface_hub import snapshot_download
        path = snapshot_download(self.model_id, revision=self.revision, local_files_only=self.local_files_only,
                                 allow_patterns=["*.json", "*.safetensors"])
        self.resolved_revision = Path(path).name
        return path

    def forecast(self, contexts, horizon):
        import torch
        if self.model is None:
            self._load()
        points, quantiles = [], []
        with torch.inference_mode():
            for start in range(0, len(contexts), self.batch_size):
                batch = [np.asarray(item, dtype=np.float32) for item in contexts[start:start + self.batch_size]]
                result = self._forecast(batch, horizon).validate(len(batch), horizon)
                points.append(result.point)
                quantiles.append(result.quantiles)
        q = np.concatenate(quantiles) if all(item is not None for item in quantiles) else None
        return Forecast(np.concatenate(points), q).validate(len(contexts), horizon)

    def descriptor(self):
        return {"family": self.family, "model_id": self.model_id, "requested_revision": self.revision,
                "resolved_revision": self.resolved_revision, "device": self.device,
                "batch_size": self.batch_size, "point_statistic": "median", "mode": "zero-shot; no task fitting"}

    def close(self):
        self.model = None
        import gc
        import torch
        gc.collect()
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()


class TimestralForecaster(FoundationForecaster):
    model_id = "mistralai/Timestral-209M"
    family = "timestral"

    def _load(self):
        from timestral import TimestralForecaster as NativeTimestral
        self.model = NativeTimestral.from_pretrained(self._checkpoint(), device=self.device)

    def _forecast(self, contexts, horizon):
        result = self.model.forecast(_padded(contexts)[:, None, :], prediction_length=horizon)
        quantiles = np.stack([_numpy(result.quantile(q))[:, 0, :] for q in QUANTILES], axis=-1)
        return Forecast(_numpy(result.median)[:, 0, :], quantiles)


class TotoForecaster(FoundationForecaster):
    model_id = "Datadog/Toto-2.0-313m"
    family = "toto-2.0"

    def _load(self):
        from toto2 import Toto2Model
        self.model = Toto2Model.from_pretrained(self._checkpoint()).to(self.device).eval()

    def _forecast(self, contexts, horizon):
        import torch
        values = torch.as_tensor(_padded(contexts)[:, None, :], device=self.device)
        mask = torch.isfinite(values)
        inputs = {"target": torch.nan_to_num(values), "target_mask": mask,
                  "series_ids": torch.zeros((len(contexts), 1), dtype=torch.long, device=self.device)}
        result = self.model.forecast(inputs, horizon=horizon, decode_block_size=None,
                                     has_missing_values=not bool(mask.all()))
        quantiles = _numpy(result)[[0, 4, 8], :, 0, :].transpose(1, 2, 0)
        return Forecast(quantiles[:, :, 1], quantiles)


class TimesFMForecaster(FoundationForecaster):
    model_id = "google/timesfm-3.0-pytorch"
    family = "timesfm-3.0"

    def _load(self):
        from timesfm3 import TimesFM3Evaluator, ModelConfig
        self.model = TimesFM3Evaluator(ModelConfig(checkpoint_path=self._checkpoint(), device=self.device,
                                                  per_core_batch_size=self.batch_size))

    def _forecast(self, contexts, horizon):
        outputs = list(self.model.predict_batch(contexts, horizon=horizon, return_quantiles=True,
                                               use_symmetric_averaging=False))
        quantiles = np.stack([_numpy(item.quantiles)[:, [0, 4, 8]] for item in outputs])
        return Forecast(quantiles[:, :, 1], quantiles)


class ChronosForecaster(FoundationForecaster):
    model_id = "amazon/chronos-2"
    family = "chronos-2"

    def _load(self):
        from chronos import Chronos2Pipeline
        self.model = Chronos2Pipeline.from_pretrained(self._checkpoint(), device_map=self.device)

    def _forecast(self, contexts, horizon):
        import torch
        values = [torch.as_tensor(item) for item in contexts]
        quantiles, _ = self.model.predict_quantiles(values, prediction_length=horizon, quantile_levels=list(QUANTILES))
        quantiles = np.stack([_numpy(item).reshape(horizon, len(QUANTILES)) for item in quantiles])
        return Forecast(quantiles[:, :, 1], quantiles)


FORECASTING_MODEL_REGISTRY = {
    "naive_forecaster": NaiveForecaster,
    "seasonal_naive_forecaster": partial(NaiveForecaster, season_length=7),
    "timestral_209m": TimestralForecaster,
    **{f"toto_2_0_{size}": partial(TotoForecaster, model_id=f"Datadog/Toto-2.0-{hub_size}")
       for size, hub_size in [("4m", "4m"), ("22m", "22m"), ("313m", "313m"), ("1b", "1B"), ("2_5b", "2.5B")]},
    "timesfm_3_0": TimesFMForecaster,
    "chronos_2": ChronosForecaster,
}


def forecasting_catalog():
    return {name: {"family": "baseline" if "naive" in name else "time-series foundation model",
                  "task": "forecasting", "mode": "zero-shot", "input": "past numeric target history per series",
                  "output": "median point forecast; foundation models also return 0.1/0.5/0.9 quantiles",
                  "covariates": "not exposed in this common univariate benchmark",
                  "license": "non-commercial pretrained weights" if name == "timesfm_3_0" else
                             "Modified MIT pretrained weights" if name == "timestral_209m" else "Apache-2.0"}
            for name in FORECASTING_MODEL_REGISTRY}
