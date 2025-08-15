"""Extended time series utilities with forecasting helpers."""
from __future__ import annotations

from typing import Iterable, Dict, Any

import numpy as np
import pandas as pd
import seaborn as sns  # noqa: F401
import statsmodels.api as sm

try:  # pragma: no cover - optional dependency
    import pmdarima as pm  # noqa: F401
except Exception:  # pragma: no cover
    pm = None  # type: ignore

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor  # noqa: F401
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore

import plotly.graph_objects as go


def run_full_analysis(series: Iterable[float]) -> Dict[str, Any]:
    """Return forecast metrics and a Plotly figure for ``series``."""
    arr = np.array(list(series), dtype=float)
    if arr.size < 5:
        return {"metrics": {}, "figure": go.Figure().to_dict()}

    train, test = train_test_split(arr, test_size=0.2, shuffle=False)
    model = sm.tsa.ARIMA(train, order=(1, 0, 0)).fit()
    forecast = model.forecast(steps=len(test))
    mse = float(mean_squared_error(test, forecast))

    fig = go.Figure(
        data=[
            go.Scatter(y=arr, mode="lines", name="data"),
            go.Scatter(
                x=np.arange(len(train), len(arr)),
                y=forecast,
                mode="lines",
                name="forecast",
            ),
        ]
    )
    fig.update_layout(title="Full Time Series Forecast", xaxis_title="Index", yaxis_title="Value")
    return {"metrics": {"mse": mse}, "figure": fig.to_dict()}
