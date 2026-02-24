from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from forecasting_framework.modeling.base_model import FitContext
from forecasting_framework.utils.metrics import mae, rmse


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    metrics: Dict[str, float]


class WalkForwardBacktester:
    """
    Expanding-window walk-forward backtest.

    v1 supports a single series (id_col=None).
    v1 supports horizon=1 cleanly.
    """

    def __init__(self, time_col: str, target_col: str, id_col: Optional[str] = None) -> None:
        if id_col is not None:
            raise NotImplementedError("Multi-series (id_col) will be added in v2.")
        self.time_col = time_col
        self.target_col = target_col
        self.id_col = id_col

    def run(
        self,
        df: pd.DataFrame,
        model,
        train_size: int,
        step_size: int,
        horizon: int,
        dropna: bool = True,
    ) -> BacktestResult:
        if horizon != 1:
            raise NotImplementedError("v1 supports horizon=1. Multi-step horizon will be added in v2.")

        if dropna:
            df = df.dropna().copy()

        df = df.sort_values(self.time_col).reset_index(drop=True)

        n = len(df)
        if n < train_size + horizon:
            raise ValueError(f"Not enough rows ({n}) for train_size={train_size} and horizon={horizon}.")

        preds: List[Dict[str, object]] = []

        feature_cols = [c for c in df.columns if c not in {self.time_col, self.target_col}]
        ctx = FitContext(target_col=self.target_col, time_col=self.time_col, id_col=self.id_col)

        start = train_size
        while start < n:
            train_df = df.iloc[:start]
            test_df = df.iloc[start : start + horizon]  # horizon=1

            if len(test_df) == 0:
                break

            X_train = train_df[feature_cols]
            y_train = train_df[self.target_col]
            X_test = test_df[feature_cols]
            y_test = test_df[self.target_col]

            fitted = model.fit(X_train, y_train, ctx=ctx)
            y_pred = fitted.predict(X_test)

            preds.append(
                {
                    self.time_col: test_df.iloc[0][self.time_col],
                    "y_true": float(y_test.iloc[0]),
                    "y_pred": float(y_pred.iloc[0]),
                }
            )

            start += step_size

        pred_df = pd.DataFrame(preds).sort_values(self.time_col).reset_index(drop=True)
        m = {
            "mae": mae(pred_df["y_true"], pred_df["y_pred"]),
            "rmse": rmse(pred_df["y_true"], pred_df["y_pred"]),
            "n_predictions": float(len(pred_df)),
        }
        return BacktestResult(predictions=pred_df, metrics=m)