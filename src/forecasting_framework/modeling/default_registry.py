from forecasting_framework.modeling.model_registry import ModelRegistry
from forecasting_framework.modeling.xgb_model import XGBoostModel


def build_default_registry() -> ModelRegistry:
    reg = ModelRegistry()
    reg.register("xgboost", lambda **kwargs: XGBoostModel(**kwargs))
    return reg