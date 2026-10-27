"""Tabular estimators and MLPs."""

import logging
import inspect
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger("ml_pipeline.pipelines_torch.models")


class TorchMLP(nn.Module):
    """
    Configurable MLP for classification or regression.
    """

    __module__ = "ml_pipeline.pipelines_torch.models"
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64], output_dim: int = 1, dropout: float = 0.3, batchnorm: bool = False):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


class TorchMLPClassifier(TorchMLP):

    __module__ = "ml_pipeline.pipelines_torch.models"
    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64], num_classes: int = 2, dropout: float = 0.3, batchnorm: bool = False):
        super().__init__(input_dim, hidden_dims, num_classes, dropout, batchnorm)


class TorchMLPRegressor(TorchMLP):

    __module__ = "ml_pipeline.pipelines_torch.models"
    def __init__(self, input_dim: int, output_dim: int = 1, hidden_dims: list = [256, 128, 64], dropout: float = 0.3, batchnorm: bool = False):
        super().__init__(input_dim, hidden_dims, output_dim, dropout, batchnorm)


class SklearnModelWrapper:
    """Base wrapper making any sklearn model compatible with GeneralPipeline."""

    __module__ = "ml_pipeline.pipelines_torch.models"

    _PIPELINE_ONLY_KWARGS = {
        "input_dim",
        "num_classes",
        "output_dim",
        "device",
        "task_type",
    }

    def __init__(self, sklearn_cls, **kwargs):
        kwargs = self._filter_sklearn_kwargs(sklearn_cls, kwargs)
        self.model = sklearn_cls(**kwargs)

    @classmethod
    def _filter_sklearn_kwargs(cls, sklearn_cls, kwargs):
        model_name = getattr(sklearn_cls, "__name__", str(sklearn_cls))
        valid_params = set(sklearn_cls().get_params().keys())
        filtered = {
            key: value
            for key, value in kwargs.items()
            if key in valid_params
        }
        ignored = sorted(set(kwargs) - set(filtered))
        unexpected = [key for key in ignored if key not in cls._PIPELINE_ONLY_KWARGS]
        if unexpected:
            logger.warning("Ignoring unsupported %s kwargs: %s", model_name, unexpected)
        return filtered

    def to(self, device):
        return self

    def eval(self):
        pass

    def train(self):
        pass

    def fit(self, X, y, *args, **kwargs):
        X, y = self._to_numpy(X), self._to_numpy(y)
        sample_weight = kwargs.get('sample_weight')
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)

    def predict(self, X):
        X = self._to_numpy(X)
        predictions = self.model.predict(X)
        logger.debug("[%s] predict shape=%s", self.__class__.__name__, np.asarray(predictions).shape)
        return predictions

    def predict_proba(self, X):
        X = self._to_numpy(X)
        probs = self.model.predict_proba(X)
        logger.debug("[%s] predict_proba shape=%s", self.__class__.__name__, np.asarray(probs).shape)
        return probs

    def __call__(self, X):
        X = self._to_numpy(X)
        if hasattr(self.model, 'predict_proba'):
            return torch.tensor(self.model.predict_proba(X), dtype=torch.float32)
        return torch.tensor(self.model.predict(X), dtype=torch.float32)

    @staticmethod
    def _to_numpy(data):
        return data.cpu().numpy() if isinstance(data, torch.Tensor) else data


class SklearnRandomForestClassifierWrapper(SklearnModelWrapper):

    __module__ = "ml_pipeline.pipelines_torch.models"
    def __init__(self, **kwargs):
        super().__init__(RandomForestClassifier, **kwargs)


class SklearnRandomForestRegressorWrapper(SklearnModelWrapper):

    __module__ = "ml_pipeline.pipelines_torch.models"
    def __init__(self, **kwargs):
        super().__init__(RandomForestRegressor, **kwargs)


class ConstantClassifierWrapper(SklearnModelWrapper):

    __module__ = "ml_pipeline.pipelines_torch.models"
    def __init__(self, **kwargs):
        from sklearn.dummy import DummyClassifier
        super().__init__(DummyClassifier, strategy="prior", **kwargs)


class ConstantRegressorWrapper(SklearnModelWrapper):

    __module__ = "ml_pipeline.pipelines_torch.models"
    def __init__(self, **kwargs):
        from sklearn.dummy import DummyRegressor
        super().__init__(DummyRegressor, **kwargs)


    def fit(self, X, y, *args, **kwargs):
        self.model.fit(self._to_numpy(X), np.asarray(self._to_numpy(y), dtype=np.float64))


class LinearClassifierWrapper(SklearnModelWrapper):

    __module__ = "ml_pipeline.pipelines_torch.models"
    def __init__(self, **kwargs):
        from sklearn.linear_model import LogisticRegression
        kwargs.setdefault("max_iter", 1000)
        super().__init__(LogisticRegression, **kwargs)


class LinearRegressorWrapper(SklearnModelWrapper):

    __module__ = "ml_pipeline.pipelines_torch.models"
    def __init__(self, **kwargs):
        from sklearn.linear_model import Ridge
        super().__init__(Ridge, **kwargs)


class XGBoostClassifierWrapper(SklearnModelWrapper):

    __module__ = "ml_pipeline.pipelines_torch.models"
    def __init__(self, **kwargs):
        super().__init__(xgb.XGBClassifier, **kwargs)


class XGBoostRegressorWrapper(SklearnModelWrapper):

    __module__ = "ml_pipeline.pipelines_torch.models"
    def __init__(self, **kwargs):
        super().__init__(xgb.XGBRegressor, **kwargs)


class LightGBMClassifierWrapper(SklearnModelWrapper):

    __module__ = "ml_pipeline.pipelines_torch.models"
    def __init__(self, **kwargs):
        super().__init__(lgb.LGBMClassifier, **kwargs)


class LightGBMRegressorWrapper(SklearnModelWrapper):
    """LightGBM regressor wrapped in MultiOutputRegressor for multi-target support."""

    __module__ = "ml_pipeline.pipelines_torch.models"

    def __init__(self, **kwargs):
        super().__init__(lgb.LGBMRegressor, **kwargs)
        # Re-wrap the inner model with MultiOutputRegressor
        self.model = MultiOutputRegressor(estimator=self.model)

    def fit(self, X, y, *args, **kwargs):
        X, y = self._to_numpy(X), self._to_numpy(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.model.fit(X, y)

    def predict(self, X):
        X = self._to_numpy(X)
        preds = self.model.predict(X)
        logger.debug("[%s] predict shape=%s", self.__class__.__name__, np.asarray(preds).shape)
        return torch.tensor(preds, dtype=torch.float32)


class CatBoostClassifierWrapper(SklearnModelWrapper):
    """CatBoost classifier (ordered boosting, oblivious trees)."""

    __module__ = "ml_pipeline.pipelines_torch.models"

    _DEFAULTS = dict(
        iterations=500, learning_rate=0.05, depth=6,
        random_seed=42, verbose=False, allow_writing_files=False,
    )

    def __init__(self, **kwargs):
        from catboost import CatBoostClassifier

        params = dict(self._DEFAULTS)
        params.update({k: v for k, v in kwargs.items() if k not in self._PIPELINE_ONLY_KWARGS})
        self.model = CatBoostClassifier(**params)


class CatBoostRegressorWrapper(SklearnModelWrapper):
    """CatBoost regressor wrapped in MultiOutputRegressor for multi-target support."""

    __module__ = "ml_pipeline.pipelines_torch.models"

    _DEFAULTS = dict(
        iterations=500, learning_rate=0.05, depth=6,
        random_seed=42, verbose=False, allow_writing_files=False,
    )

    def __init__(self, **kwargs):
        from catboost import CatBoostRegressor

        params = dict(self._DEFAULTS)
        params.update({k: v for k, v in kwargs.items() if k not in self._PIPELINE_ONLY_KWARGS})
        self.model = MultiOutputRegressor(estimator=CatBoostRegressor(**params))

    def fit(self, X, y, *args, **kwargs):
        X, y = self._to_numpy(X), self._to_numpy(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.model.fit(X, y)

    def predict(self, X):
        X = self._to_numpy(X)
        preds = self.model.predict(X)
        logger.debug("[%s] predict shape=%s", self.__class__.__name__, np.asarray(preds).shape)
        return torch.tensor(preds, dtype=torch.float32)


def _stacking_base_classifiers(random_state=42, n_estimators=300):
    from sklearn.linear_model import LogisticRegression

    return [
        ("lightgbm", lgb.LGBMClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            random_state=random_state, verbosity=-1)),
        ("xgboost", xgb.XGBClassifier(
            n_estimators=n_estimators, eval_metric="mlogloss",
            random_state=random_state, verbosity=0)),
        ("logreg", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ]


class StackingEnsembleClassifierWrapper(SklearnModelWrapper):
    """LightGBM + XGBoost + logistic base learners stacked out-of-fold
    with a logistic meta-learner over their predicted probabilities."""

    __module__ = "ml_pipeline.pipelines_torch.models"

    def __init__(self, cv: int = 5, random_state: int = 42, n_estimators: int = 300,
                 n_jobs: int = 8, **kwargs):
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression

        # n_jobs bounded by default: -1 on many-core shared hosts spawns a
        # loky worker storm that can get OOM-killed
        self.model = StackingClassifier(
            estimators=_stacking_base_classifiers(random_state, n_estimators),
            final_estimator=LogisticRegression(max_iter=3000),
            cv=cv, stack_method="predict_proba", n_jobs=n_jobs,
        )


class StackingEnsembleRegressorWrapper(SklearnModelWrapper):
    """Per-target stacking of LightGBM + XGBoost + ridge with a RidgeCV
    meta-learner, wrapped in MultiOutputRegressor for multi-target support."""

    __module__ = "ml_pipeline.pipelines_torch.models"

    def __init__(self, cv: int = 5, random_state: int = 42, n_estimators: int = 400,
                 n_jobs: int = 8, **kwargs):
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import Ridge, RidgeCV

        base = StackingRegressor(
            estimators=[
                ("lightgbm", lgb.LGBMRegressor(
                    n_estimators=n_estimators, random_state=random_state, verbosity=-1)),
                ("xgboost", xgb.XGBRegressor(
                    n_estimators=n_estimators, random_state=random_state, verbosity=0)),
                ("ridge", Ridge(alpha=10.0)),
            ],
            final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
            cv=cv, n_jobs=n_jobs,
        )
        self.model = MultiOutputRegressor(estimator=base)

    def fit(self, X, y, *args, **kwargs):
        X, y = self._to_numpy(X), self._to_numpy(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.model.fit(X, y)

    def predict(self, X):
        X = self._to_numpy(X)
        preds = self.model.predict(X)
        logger.debug("[%s] predict shape=%s", self.__class__.__name__, np.asarray(preds).shape)
        return torch.tensor(preds, dtype=torch.float32)


class TabICLClassifierWrapper(SklearnModelWrapper):

    __module__ = "ml_pipeline.pipelines_torch.models"
    def __init__(self, **kwargs):
        from tabicl import TabICLClassifier
        kwargs.setdefault("checkpoint_version", "tabicl-classifier-v2-20260212.ckpt")
        super().__init__(TabICLClassifier, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        kwargs.pop('sample_weight', None)
        super().fit(X, y, *args, **kwargs)


class TabICLRegressorWrapper(SklearnModelWrapper):

    __module__ = "ml_pipeline.pipelines_torch.models"
    def __init__(self, **kwargs):
        from tabicl import TabICLRegressor
        kwargs.setdefault("checkpoint_version", "tabicl-regressor-v2-20260212.ckpt")
        super().__init__(TabICLRegressor, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        kwargs.pop('sample_weight', None)
        super().fit(X, y, *args, **kwargs)


class TabFMClassifierWrapper(SklearnModelWrapper):
    """Google TabFM zero-shot tabular foundation model (PyTorch backend).

    Predictions come from in-context learning over the training rows; fit only
    prepares encoders and the ensemble configuration.
    """

    __module__ = "ml_pipeline.pipelines_torch.models"

    def __init__(self, device: Optional[str] = None, **kwargs):
        from tabfm import TabFMClassifier
        from tabfm import tabfm_v1_0_0_pytorch

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        backbone = tabfm_v1_0_0_pytorch.load(device=dev)
        valid_params = inspect.signature(TabFMClassifier).parameters
        params = {k: v for k, v in kwargs.items() if k in valid_params}
        ignored = sorted(set(kwargs) - set(params) - self._PIPELINE_ONLY_KWARGS)
        if ignored:
            logger.warning("Ignoring unsupported TabFMClassifier kwargs: %s", ignored)
        self.model = TabFMClassifier(model=backbone, **params)

    def fit(self, X, y, *args, **kwargs):
        kwargs.pop('sample_weight', None)
        super().fit(X, y, *args, **kwargs)


class TabFMRegressorWrapper(SklearnModelWrapper):
    """Google TabFM zero-shot tabular foundation model for regression (PyTorch backend)."""

    __module__ = "ml_pipeline.pipelines_torch.models"

    def __init__(self, device: Optional[str] = None, **kwargs):
        from tabfm import TabFMRegressor
        from tabfm import tabfm_v1_0_0_pytorch

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        backbone = tabfm_v1_0_0_pytorch.load(model_type="regression", device=dev)
        valid_params = inspect.signature(TabFMRegressor).parameters
        params = {k: v for k, v in kwargs.items() if k in valid_params}
        ignored = sorted(set(kwargs) - set(params) - self._PIPELINE_ONLY_KWARGS)
        if ignored:
            logger.warning("Ignoring unsupported TabFMRegressor kwargs: %s", ignored)
        self.model = TabFMRegressor(model=backbone, **params)

    def fit(self, X, y, *args, **kwargs):
        kwargs.pop('sample_weight', None)
        super().fit(X, y, *args, **kwargs)

