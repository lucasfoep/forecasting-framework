from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = y_true.astype(float).to_numpy()
    y_pred = y_pred.astype(float).to_numpy()
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = y_true.astype(float).to_numpy()
    y_pred = y_pred.astype(float).to_numpy()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))