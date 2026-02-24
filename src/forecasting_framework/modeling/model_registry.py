from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from forecasting_framework.modeling.base_model import BaseModel


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable[..., BaseModel]


class ModelRegistry:
    """
    model_name -> builder(**kwargs)
    """

    def __init__(self) -> None:
        self._models: Dict[str, ModelSpec] = {}

    def register(self, name: str, builder: Callable[..., BaseModel]) -> None:
        key = name.strip().lower()
        if key in self._models:
            raise ValueError(f"Model '{name}' is already registered.")
        self._models[key] = ModelSpec(name=key, builder=builder)

    def create(self, name: str, **kwargs) -> BaseModel:
        key = name.strip().lower()
        if key not in self._models:
            raise KeyError(f"Unknown model '{name}'. Available: {self.list_models()}")
        return self._models[key].builder(**kwargs)

    def list_models(self) -> List[str]:
        return sorted(self._models.keys())