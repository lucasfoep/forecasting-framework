from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from xgboost import XGBRegressor

from forecasting_framework.modeling.base_model import BaseModel, FitContext


@dataclass
class XGBoostModel(BaseModel):
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    random_state: int = 42

    def __post_init__(self) -> None:
        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, ctx: FitContext) -> "XGBoostModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self._model.predict(X)
        return pd.Series(preds, index=X.index, name="y_pred")