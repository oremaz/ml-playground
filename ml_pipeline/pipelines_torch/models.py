"""Model registries and stable imports used by consumers and saved checkpoints."""

from typing import Callable, Dict

from ml_pipeline.forecasting_models import FORECASTING_MODEL_REGISTRY
from .hf_models import HuggingFaceQLoRAWrapper
from .llama_models import LlamaCppClassifier
from .kumo_models import KumoTabularClassifier, KumoTabularRegressor
from .tabular_models import (
    TorchMLP,
    TorchMLPClassifier,
    TorchMLPRegressor,
    SklearnModelWrapper,
    SklearnRandomForestClassifierWrapper,
    SklearnRandomForestRegressorWrapper,
    ConstantClassifierWrapper,
    ConstantRegressorWrapper,
    LinearClassifierWrapper,
    LinearRegressorWrapper,
    XGBoostClassifierWrapper,
    XGBoostRegressorWrapper,
    LightGBMClassifierWrapper,
    LightGBMRegressorWrapper,
    CatBoostClassifierWrapper,
    CatBoostRegressorWrapper,
    _stacking_base_classifiers,
    StackingEnsembleClassifierWrapper,
    StackingEnsembleRegressorWrapper,
    TabICLClassifierWrapper,
    TabICLRegressorWrapper,
    TabFMClassifierWrapper,
    TabFMRegressorWrapper,
)


CLASSIFICATION_MODEL_REGISTRY: Dict[str, Callable] = {
    "constant_classifier": ConstantClassifierWrapper,
    "linear_classifier": LinearClassifierWrapper,
    "mlp_classifier": TorchMLPClassifier,
    "deep_mlp_classifier": lambda input_dim, num_classes=2: TorchMLPClassifier(input_dim, [512, 256, 128, 64], num_classes=num_classes, dropout=0.3, batchnorm=True),
    "random_forest_classifier": SklearnRandomForestClassifierWrapper,
    "xgboost_classifier": XGBoostClassifierWrapper,
    "lightgbm_classifier": LightGBMClassifierWrapper,
    "catboost_classifier": CatBoostClassifierWrapper,
    "stacking_classifier": StackingEnsembleClassifierWrapper,
    "hf_qlora_classifier": lambda **kwargs: HuggingFaceQLoRAWrapper(task_type='classification', **kwargs),
    "llama_cpp_classifier": lambda **kwargs: LlamaCppClassifier(task_type='classification', **kwargs),
    "tabicl_classifier": TabICLClassifierWrapper,
    "tabfm_classifier": TabFMClassifierWrapper,
    "kumo_classifier": KumoTabularClassifier,
}

REGRESSION_MODEL_REGISTRY: Dict[str, Callable] = {
    "constant_regressor": ConstantRegressorWrapper,
    "linear_regressor": LinearRegressorWrapper,
    "mlp_regressor": TorchMLPRegressor,
    "deep_mlp_regressor": lambda input_dim, output_dim: TorchMLPRegressor(input_dim, output_dim, [512, 256, 128, 64], dropout=0.3, batchnorm=True),
    "random_forest_regressor": SklearnRandomForestRegressorWrapper,
    "xgboost_regressor": XGBoostRegressorWrapper,
    "lightgbm_regressor": LightGBMRegressorWrapper,
    "catboost_regressor": CatBoostRegressorWrapper,
    "stacking_regressor": StackingEnsembleRegressorWrapper,
    "hf_qlora_regressor": lambda **kwargs: HuggingFaceQLoRAWrapper(task_type='regression', **kwargs),
    "llama_cpp_regressor": lambda **kwargs: LlamaCppClassifier(task_type='regression', **kwargs),
    "tabicl_regressor": TabICLRegressorWrapper,
    "tabfm_regressor": TabFMRegressorWrapper,
    "kumo_regressor": KumoTabularRegressor,
}

# Combined registry for backward compatibility
MODEL_REGISTRY: Dict[str, Callable] = {
    **CLASSIFICATION_MODEL_REGISTRY,
    **REGRESSION_MODEL_REGISTRY,
    **FORECASTING_MODEL_REGISTRY,
}
