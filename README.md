\# forecasting-framework



A small, production-shaped forecasting/backtesting framework focused on \*\*time-aware evaluation\*\* (walk-forward / expanding window) and \*\*modular model interfaces\*\*.



\## What it demonstrates

\- \*\*Walk-forward backtesting\*\* (expanding window) for time series

\- Clean \*\*model interface\*\* (`fit` / `predict`) + \*\*registry pattern\*\*

\- Reproducible project setup (src layout + `pyproject.toml`)

\- A runnable demo using a synthetic dataset (no proprietary data)



\## Quickstart

```bash

python -m venv .venv

\# Windows PowerShell:

.\\.venv\\Scripts\\Activate.ps1



python -m pip install -r requirements.txt

python -m pip install -e .

python scripts/run\_backtest.py

