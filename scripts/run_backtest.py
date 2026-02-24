from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_framework.config.settings import RunConfig
from forecasting_framework.modeling.default_registry import build_default_registry
from forecasting_framework.pipeline.backtester import WalkForwardBacktester


def make_synthetic_data(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ds = pd.date_range("2020-01-01", periods=n, freq="D")
    base = np.sin(np.linspace(0, 10 * np.pi, n)) + rng.normal(0, 0.15, size=n)
    y = base + rng.normal(0, 0.1, size=n)

    df = pd.DataFrame({"ds": ds, "y": y})
    df["lag_1"] = df["y"].shift(1)
    df["lag_7"] = df["y"].shift(7)
    df["roll_7_mean"] = df["y"].rolling(7).mean()
    df["roll_14_mean"] = df["y"].rolling(14).mean()
    df["dow"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run walk-forward backtest (XGBoost).")
    p.add_argument("--csv", type=str, default=None, help="Path to CSV dataset. If omitted, uses synthetic data.")
    p.add_argument("--time_col", type=str, default=None, help="Time column name (overrides config).")
    p.add_argument("--target_col", type=str, default=None, help="Target column name (overrides config).")

    p.add_argument("--train_size", type=int, default=None, help="Initial training window size.")
    p.add_argument("--step_size", type=int, default=None, help="Step size for walk-forward.")
    p.add_argument("--horizon", type=int, default=None, help="Forecast horizon (v1 supports 1).")

    p.add_argument("--n_estimators", type=int, default=None)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig()

    # Load data
    if args.csv:
        path = Path(args.csv)
        df = pd.read_csv(path)

        # If time column looks like dates, parse it
        tcol = args.time_col or cfg.backtest.time_col
        if tcol in df.columns:
            df[tcol] = pd.to_datetime(df[tcol])
    else:
        df = make_synthetic_data()

    # Override config from CLI if provided
    time_col = args.time_col or cfg.backtest.time_col
    target_col = args.target_col or cfg.backtest.target_col

    train_size = args.train_size or cfg.backtest.train_size
    step_size = args.step_size or cfg.backtest.step_size
    horizon = args.horizon or cfg.backtest.horizon

    model_kwargs = {
        "n_estimators": args.n_estimators or cfg.model.n_estimators,
        "max_depth": args.max_depth or cfg.model.max_depth,
        "learning_rate": args.learning_rate or cfg.model.learning_rate,
        "subsample": cfg.model.subsample,
        "colsample_bytree": cfg.model.colsample_bytree,
        "reg_lambda": cfg.model.reg_lambda,
        "random_state": cfg.model.random_state,
    }

    reg = build_default_registry()
    model = reg.create(cfg.model.name, **model_kwargs)

    bt = WalkForwardBacktester(time_col=time_col, target_col=target_col, id_col=cfg.backtest.id_col)

    result = bt.run(
        df=df,
        model=model,
        train_size=train_size,
        step_size=step_size,
        horizon=horizon,
        dropna=cfg.backtest.dropna,
    )

    print("Backtest metrics:")
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")

    print("\nSample predictions:")
    print(result.predictions.head(10))


if __name__ == "__main__":
    main()