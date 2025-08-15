"""Simplified predictive model core module.

This module exposes a :func:`run` function used by the Flask blueprint to
produce a basic forecast from an uploaded CSV/Excel file.  The implementation
is intentionally lightweight compared to the original Streamlit prototype but
keeps a similar interface: it returns a dictionary with ``metrics``,
``table`` and a binary ``file_bytes`` for download.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Optional imports used by extended forecasting strategies.  These libraries are
# heavy and may not always be installed in minimal environments, therefore they
# are loaded lazily and default to ``None`` when unavailable.
try:  # pragma: no cover - optional dependency
    from pmdarima import auto_arima
except Exception:  # pragma: no cover - optional dependency
    auto_arima = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sklearn.ensemble import RandomForestRegressor
except Exception:  # pragma: no cover - optional dependency
    RandomForestRegressor = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None  # type: ignore

_ = (np, auto_arima, RandomForestRegressor, XGBRegressor)


def detectar_frecuencia(series: pd.Series) -> str:
    """Infer the frequency of a time indexed series.

    Falls back to simple heuristics when ``pd.infer_freq`` fails.
    """

    try:
        freq = pd.infer_freq(series.index)
    except Exception:
        freq = None
    if freq:
        return freq
    dias = series.index.to_series().diff().dt.days.mean()
    if dias <= 1.5:
        return "D"
    if dias <= 8:
        return "W"
    return "MS"


def calcular_estacionalidad(series: pd.Series) -> int:
    """Return a seasonal period length based on available observations."""

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


def run(file_obj, steps_horizon: int = 6) -> Dict[str, Any]:
    """Execute a forecast from an uploaded file.

    Parameters
    ----------
    file_obj:
        File-like object containing two columns: date and value.
    steps_horizon:
        Number of future periods to forecast.

    Returns
    -------
    dict
        ``{"metrics": ..., "table": ..., "file_bytes": ...}``
    """

    # Load the dataset
    name = getattr(file_obj, "filename", "")
    if name.endswith(("xlsx", "xls")):
        raw = pd.read_excel(file_obj)
    else:
        raw = pd.read_csv(file_obj)

    raw.iloc[:, 0] = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    raw.set_index(raw.columns[0], inplace=True)
    series = raw.iloc[:, 0].ffill()

    # Basic analysis
    freq = detectar_frecuencia(series)
    m = calcular_estacionalidad(series)

    seasonal = "add" if m > 1 else None
    model = ExponentialSmoothing(
        series, trend="add", seasonal=seasonal, seasonal_periods=m if m > 1 else None
    ).fit()
    fc = model.forecast(steps_horizon)
    offset = pd.tseries.frequencies.to_offset(freq)
    dates = pd.date_range(series.index[-1] + offset, periods=steps_horizon, freq=freq)
    fc_df = pd.DataFrame({"Pronóstico": fc}, index=dates)

    # Prepare data for the template and download link
    table = [
        {"Fecha": idx.strftime("%Y-%m-%d"), "Pronóstico": float(val)}
        for idx, val in fc_df["Pronóstico"].items()
    ]

    buf = BytesIO()
    fc_df.to_csv(buf)
    file_bytes = buf.getvalue()

    return {
        "metrics": {"frecuencia": freq, "estacionalidad": m},
        "table": table,
        "file_bytes": file_bytes,
    }
