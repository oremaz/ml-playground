"""sklearn interfaces to NVIDIA's frozen Kumo Tabular checkpoints."""

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


class _KumoBase(BaseEstimator):
    def __init__(self, size="small", device="cpu", num_estimators=8, random_state=42,
                 batch_size=512, max_context_rows=None):
        self.size = size
        self.device = device
        self.num_estimators = num_estimators
        self.random_state = random_state
        self.batch_size = batch_size
        self.max_context_rows = max_context_rows

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.categorical = True
        tags.input_tags.string = True
        return tags

    @staticmethod
    def _frame(X):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if frame.ndim != 2 or frame.shape[1] == 0:
            raise ValueError("Kumo requires a two-dimensional feature table")
        if not frame.columns.is_unique or not all(isinstance(c, str) for c in frame.columns):
            frame = frame.copy(deep=False)
            frame.columns = frame.columns.map(str)
        if not frame.columns.is_unique:
            raise ValueError("Feature column names must be unique")
        numeric = frame.select_dtypes(include=np.number)
        values = numeric.to_numpy()
        if values.dtype == object:
            values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError("Features may contain missing values, but not infinity")
        return frame

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise ValueError("Kumo's context interface does not support sample weights")
        if self.size not in {"small", "medium", "large"}:
            raise ValueError("Kumo size must be small, medium or large")
        for name in ("batch_size", "num_estimators", "max_context_rows"):
            value = getattr(self, name)
            if name == "max_context_rows" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        frame, target = self._frame(X), np.asarray(y)
        if target.ndim != 1 or len(target) != len(frame) or len(frame) < 2:
            raise ValueError("Provide at least two aligned rows and a single target")
        if pd.isna(target).any():
            raise ValueError("Training targets must not be missing")
        if self._task == "classification":
            self.label_encoder_ = LabelEncoder().fit(target)
            self.classes_ = self.label_encoder_.classes_
            if len(self.classes_) < 2:
                raise ValueError("Classification requires at least two classes")
            target = self.label_encoder_.transform(target)
        else:
            target = np.asarray(target, dtype=np.float64)
            if not np.isfinite(target).all():
                raise ValueError("Regression targets must be finite")
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        indices = np.arange(len(frame))
        if self.max_context_rows is not None and len(indices) > self.max_context_rows:
            rng = np.random.default_rng(self.random_state)
            if self._task == "classification":
                from sklearn.model_selection import train_test_split
                indices, _ = train_test_split(
                    indices, train_size=self.max_context_rows, stratify=target,
                    random_state=self.random_state,
                )
                indices.sort()
            else:
                indices = np.sort(rng.choice(indices, self.max_context_rows, replace=False))
        self.context_indices_ = indices
        self.X_context_ = frame.iloc[indices].copy()
        self.y_context_ = target[indices].copy()
        self._model = None
        self._context = None
        return self

    def _load(self):
        import torch
        try:
            import sdm
        except ImportError as error:
            raise ImportError("Install ml-pipeline[kumo] in a Torch 2.7+ environment") from error
        saved_version = getattr(self, "model_metadata_", {}).get("sdm_version")
        if saved_version is not None and saved_version != sdm.__version__:
            raise ValueError("SDM version changed since this checkpoint; restore its recorded environment")
        if self._model is not None:
            return sdm
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if self.device in (None, "auto") else torch.device(self.device)
        schema = sdm.infer_stypes(self.X_context_)
        if any(str(value) not in {"numerical", "categorical"} for value in schema.values()):
            raise ValueError("Kumo features must be numeric or categorical; encode dates explicitly")
        features = sdm.TableTensor.from_pandas(self.X_context_, schema, device=self._device)
        target = sdm.TableTensor.from_pandas(
            pd.DataFrame({"target": self.y_context_}),
            {"target": "categorical" if self._task == "classification" else "numerical"},
            device=self._device,
        )
        self._model = sdm.models.KumoTabular(task=self._task, size=self.size, device=self._device)
        self._model.eval().requires_grad_(False)
        self._context = (features, target, schema)
        self.model_metadata_ = {
            "model_id": "nvidia/Kumo-Tabular", "size": self.size,
            "checkpoint_revision": "v1.0.1" if sdm.__version__ == "0.2.1" else None,
            "sdm_version": sdm.__version__,
            "num_estimators": self.num_estimators, "context_rows": len(self.X_context_),
            "regression_statistic": "median" if self._task == "regression" else None,
        }
        return sdm

    def _predict(self, X):
        import torch
        check_is_fitted(self, "X_context_")
        frame = self._frame(X)
        if list(frame.columns) != self.feature_names_in_.tolist():
            raise ValueError("Prediction columns must match fitted names and order")
        if not len(frame):
            return np.empty((0, len(self.classes_))) if self._task == "classification" else np.empty(0)
        sdm = self._load()
        features, target, schema = self._context
        predictions = []
        for start in range(0, len(frame), self.batch_size):
            query = sdm.TableTensor.from_pandas(
                frame.iloc[start:start + self.batch_size], schema, device=self._device,
            )
            generator = torch.Generator(device=self._device).manual_seed(self.random_state)
            with torch.inference_mode(), torch.amp.autocast(
                self._device.type, dtype=torch.float16, enabled=self._device.type == "cuda",
            ):
                result = self._model(
                    x_context=features, y_context=target, x_query=query,
                    num_estimators=self.num_estimators, estimator_batch_size=1, generator=generator,
                )
            if self._task == "classification":
                columns = list(result.columns[sdm.Stype.numerical])
                order = [columns.index(str(i)) for i in range(len(self.classes_))]
                values = result.numerical[:, order]
            else:
                values = result[:, "q500"].numerical.squeeze(-1)
            predictions.append(values.detach().float().cpu().numpy())
        values = np.concatenate(predictions, axis=0)
        if not np.isfinite(values).all():
            raise ValueError("Kumo returned non-finite predictions")
        if self._task == "classification":
            if (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(1), 1, atol=1e-5):
                raise ValueError("Kumo returned invalid class probabilities")
        return values

    def __getstate__(self):
        state = super().__getstate__()
        state.update(_model=None, _context=None)
        return state


class KumoTabularClassifier(ClassifierMixin, _KumoBase):
    _task = "classification"

    def predict_proba(self, X):
        return self._predict(X)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]


class KumoTabularRegressor(RegressorMixin, _KumoBase):
    _task = "regression"

    def predict(self, X):
        return self._predict(X)
