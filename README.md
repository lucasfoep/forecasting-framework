# Forecasting Framework

A modular time-series forecasting framework built to support reproducible modeling and disciplined walk-forward validation.

The framework simulates real-world forecasting deployment scenarios, enforcing forward-only prediction and realistic out-of-sample evaluation.

---

## Why This Framework Exists

Many forecasting workflows rely on static train/test splits or standard cross-validation techniques that introduce temporal leakage or unrealistic evaluation assumptions.

This framework enforces strict temporal discipline to better approximate real production forecasting behavior, where only historical data is available at prediction time.

---

## Objective

Provide a structured pipeline for:

- Training forecasting models without data leakage  
- Expanding-window (walk-forward) validation  
- Clear separation between data preparation, modeling, and evaluation  
- Reproducible experimentation  

---

## Design Principles

- Strict forward-only validation (no future leakage)
- Modular architecture
- Explicit train/test window control
- Config-driven experimentation
- Transparent and window-level performance evaluation

---

## Validation Strategy

Expanding-window validation:

Train: months 1…n  
Test: month n+1  

Then expand:

Train: months 1…n+1  
Test: month n+2  

Each validation window simulates a real forecasting cycle where only historical information is available at prediction time.

This approach mirrors production forecasting environments and prevents look-ahead bias.

---

## Model Implemented

- **XGBoost Regressor**

The framework prioritizes structural integrity and validation rigor over model complexity or hyperparameter tuning.

---

## Evaluation Metric

Model performance is evaluated using:

- **MAPE (Mean Absolute Percentage Error)**

MAPE is computed independently for each validation window and aggregated across windows to summarize out-of-sample forecasting performance.

---

## Project Structure

```
forecasting-framework/
│
├── pyproject.toml
├── requirements.txt
├── README.md
├── scripts/
│   └── run_backtest.py
└── src/
    └── forecasting_framework/
        ├── config/
        │   └── settings.py
        ├── data/
        ├── models/
        │   └── xgboost_model.py
        ├── validation/
        │   └── walk_forward.py
        └── registry.py
```

---

## Structure Rationale

- `src/` layout ensures the framework behaves like an installed package.
- `registry.py` allows models to be swapped without modifying validation logic.
- `validation/` enforces forward-only evaluation.
- `config/` centralizes experiment configuration.
- `scripts/` contains runnable entry points, separate from library code.

This separation supports reproducibility, extensibility, and production-aligned experimentation.
