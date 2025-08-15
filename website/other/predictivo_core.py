"""Core forecasting utilities for the predictivo app.

This module exposes a small wrapper around statsmodels to generate
basic forecasts from a ``pandas.Series``.  It is intentionally light
weight and independent from any web framework so it can be reused by
both the Flask views and potential command line interfaces.
"""
from __future__ import annotations

from typing import Dict, Any

import warnings

import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


warnings.filterwarnings("ignore")


def detectar_frecuencia(series: pd.Series) -> str:
    """Infer a sensible frequency string for ``series``."""
    freq = pd.infer_freq(series.index)
    if freq:
        return freq
    dias = series.index.to_series().diff().dt.days.mean()
    if dias <= 1.5:
        return "D"
    if dias <= 8:
        return "W"
    return "MS"


def calcular_estacionalidad(series: pd.Series) -> int:
    """Return a crude seasonal period estimate based on length."""
    n = len(series)
    if n >= 24:
        return 12
    if n >= 18:
        return 6
    if n >= 12:
        return 4
    if n >= 8:
        return 2
    return 1


def forecast(series: pd.Series, steps: int = 6) -> Dict[str, Any]:
    """Generate forecast for ``series`` ``steps`` into the future.

    Parameters
    ----------
    series:
        Time indexed series with numeric values.
    steps:
        Number of periods to forecast.

    Returns
    -------
    dict
        ``forecast``: DataFrame with forecasted values for each model.
        ``figure``: Plotly figure with the resulting predictions.
    """

    if series.index.dtype != "datetime64[ns]":
        raise ValueError("Series index must be datetime")

    freq = detectar_frecuencia(series)
    m = calcular_estacionalidad(series)

    # Determine future index
    fechas = pd.date_range(
        series.index[-1] + pd.tseries.frequencies.to_offset(freq),
        periods=steps,
        freq=freq,
    )

    # Fit simple ARIMA(1,0,0)
    arima_model = ARIMA(series, order=(1, 0, 0)).fit()
    arima_fc = arima_model.forecast(steps)

    # Holt-Winters additive
    if m > 1:
        wes_model = ExponentialSmoothing(
            series, trend="add", seasonal="add", seasonal_periods=m
        ).fit()
    else:
        wes_model = ExponentialSmoothing(series, trend="add").fit()
    wes_fc = wes_model.forecast(steps)

    fc_df = pd.DataFrame({"ARIMA": arima_fc, "WES": wes_fc}, index=fechas)

    fig = go.Figure()
    for col in fc_df.columns:
        fig.add_trace(go.Scatter(x=fc_df.index, y=fc_df[col], name=col))
    fig.update_layout(title="Pronóstico", xaxis_title="Fecha", yaxis_title="Valor")

    return {"forecast": fc_df, "figure": fig}


__all__ = [
    "forecast",
    "detectar_frecuencia",
    "calcular_estacionalidad",
]
