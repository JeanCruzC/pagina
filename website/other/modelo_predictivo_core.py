"""Predictive modelling utilities for demo purposes.

This module groups together common data science imports so that other
components can rely on a single place for lightweight experimentation.
The implementations intentionally remain simple.
"""
from __future__ import annotations

from typing import Iterable, Dict, Any

import numpy as np
import pandas as pd
import seaborn as sns  # noqa: F401 - imported for convenience
import statsmodels.api as sm

try:  # pragma: no cover - optional dependency
    import pmdarima as pm  # noqa: F401
except Exception:  # pragma: no cover
    pm = None  # type: ignore

from sklearn.metrics import mean_squared_error  # noqa: F401
from sklearn.model_selection import train_test_split  # noqa: F401

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor  # noqa: F401
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore

import plotly.graph_objects as go


def simple_forecast(series: Iterable[float]) -> Dict[str, Any]:
    """Return a one-step forecast using a minimal ARIMA model."""
    arr = np.array(list(series), dtype=float)
    if arr.size == 0:
        return {"forecast": [], "figure": go.Figure().to_dict()}

    model = sm.tsa.ARIMA(arr, order=(1, 0, 0)).fit()
    forecast = model.forecast(steps=1)

    fig = go.Figure(data=[go.Scatter(y=arr, mode="lines", name="data")])
    fig.add_trace(go.Scatter(x=[len(arr)], y=forecast.tolist(), mode="markers", name="forecast"))
    fig.update_layout(title="Simple Forecast", xaxis_title="Index", yaxis_title="Value")
    return {"forecast": forecast.tolist(), "figure": fig.to_dict()}
