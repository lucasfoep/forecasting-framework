from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd


@dataclass
class FitContext:
    """
    Metadata that can be useful for fit-time logic (time column, target column, etc.)
    """
    target_col: str
    time_col: str
    id_col: Optional[str] = None


class BaseModel(ABC):
    """
    Minimal interface all models must implement.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, ctx: FitContext) -> "BaseModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        return joblib.load(Path(path))