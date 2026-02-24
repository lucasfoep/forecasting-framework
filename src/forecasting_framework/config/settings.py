from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BacktestConfig:
    """
    Defines how walk-forward backtesting should be executed.
    - train_size: number of rows in the initial training window
    - step_size: how many rows to move forward each iteration
    - horizon: how many steps ahead to predict (v1 supports horizon=1 cleanly)
    """
    time_col: str = "ds"
    target_col: str = "y"
    id_col: Optional[str] = None

    train_size: int = 60
    step_size: int = 1
    horizon: int = 1

    dropna: bool = True


@dataclass(frozen=True)
class XGBConfig:
    """
    XGBoost hyperparameters live here so the pipeline is reproducible.
    Keep it small for v1.
    """
    name: str = "xgboost"
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    random_state: int = 42


@dataclass(frozen=True)
class RunConfig:
    model: XGBConfig = XGBConfig()
    backtest: BacktestConfig = BacktestConfig()