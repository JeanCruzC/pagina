"""Utility functions for simple time series smoothing and forecasting.

This module exposes a :func:`timeseries_core` helper used by the web
application to generate a smoothed series and a naive forecast.  The function
returns a Plotly ``Figure`` alongside some basic metrics which are displayed in
the UI.

The implementation intentionally keeps the logic lightweight – it does not aim
to be a full fledged forecasting library but rather a demonstrative example
that produces deterministic results for testing purposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from math import sin
import plotly.graph_objects as go


@dataclass
class TimeSeriesResult:
    """Container for the objects returned by :func:`timeseries_core`."""

    figure: go.Figure
    metrics: Dict[str, float]


def timeseries_core(
    method: str = "rolling",
    window: int = 5,
    params: Dict[str, float] | None = None,
) -> TimeSeriesResult:
    """Generate a simple smoothed time series and naive forecast.

    Parameters
    ----------
    method:
        Smoothing method. Supported values are ``"rolling"`` for a moving
        average and ``"exp"`` for exponential smoothing.
    window:
        Size of the window used for smoothing and forecasting.
    params:
        Optional dictionary with extra parameters.  ``alpha`` is recognised when
        ``method`` is ``"exp"``.

    Returns
    -------
    TimeSeriesResult
        A container with the generated Plotly figure and a dictionary of
        metrics.
    """

    if params is None:
        params = {}

    # Deterministic toy data: a noisy upward trend.
    rng = list(range(30))
    series = [i / 30 + 0.1 * sin(i / 2) for i in rng]

    # Smoothing
    if method == "exp":
        alpha = float(params.get("alpha", 0.3))
        smooth: List[float] = []
        for x in series:
            if not smooth:
                smooth.append(x)
            else:
                smooth.append(alpha * x + (1 - alpha) * smooth[-1])
    else:
        smooth = []
        for i in range(len(series)):
            start = max(0, i + 1 - window)
            segment = series[start : i + 1]
            smooth.append(sum(segment) / len(segment))

    # Naive forecast: extend last smoothed value ``window`` steps ahead.
    forecast_index = list(range(len(series), len(series) + window))
    forecast = [smooth[-1]] * window

    # Build Plotly figure.
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rng, y=series, name="original"))
    fig.add_trace(go.Scatter(x=rng, y=smooth, name="smooth"))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast, name="forecast"))

    metrics = {
        "original_mean": sum(series) / len(series),
        "smooth_mean": sum(smooth) / len(smooth),
        "forecast_last": forecast[-1],
    }

    return TimeSeriesResult(fig, metrics)


__all__ = ["timeseries_core", "TimeSeriesResult"]

