"""Core forecasting utilities without any web framework dependencies.

This module provides a small helper to load a time series from a CSV/XLSX
upload and generate a forecast using a combination of statistical and machine
learning models.  It intentionally avoids Streamlit so that it can be reused by
Flask views or other callers.
"""

from __future__ import annotations

from typing import Dict, Any

import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def detectar_frecuencia(series: pd.Series) -> str:
    """Infer a plausible frequency string for ``series`` index."""
    freq = pd.infer_freq(series.index)
    if freq:
        return freq
    dias = series.index.to_series().diff().dt.days.median()
    if dias and dias <= 1.5:
        return "D"
    if dias and dias <= 8:
        return "W"
    return "M"

def calcular_estacionalidad(series: pd.Series) -> int:
    """Return a seasonal period guess based on series length."""
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

def build_features(series: pd.Series) -> pd.DataFrame:
    """Build lag based features for machine learning models."""
    df = pd.DataFrame(index=series.index)
    df["Valor"] = series
    df["Mes"] = df.index.month
    df["DiaDelAnio"] = df.index.dayofyear
    df["Lag1"] = df["Valor"].shift(1)
    df["Lag2"] = df["Valor"].shift(2)
    df["MediaMovil3"] = df["Valor"].rolling(3).mean()
    return df.dropna()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_series(upload) -> pd.Series:
    """Return a ``pd.Series`` from an uploaded CSV or Excel file."""
    raw = (
        pd.read_excel(upload)
        if getattr(upload, "filename", "").lower().endswith((".xlsx", ".xls"))
        else pd.read_csv(upload)
    )
    raw.iloc[:, 0] = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    raw.set_index(raw.columns[0], inplace=True)
    series = raw.iloc[:, 0].ffill()
    return series

def generate_forecast(series: pd.Series, steps: int = 6) -> Dict[str, Any]:
    """Generate forecasts for ``series`` returning a DataFrame and figure."""
    freq = detectar_frecuencia(series)
    m = calcular_estacionalidad(series)

    feat_df = build_features(series)
    X, y = feat_df.drop(columns="Valor"), feat_df["Valor"]
    if len(X) == 0:
        raise ValueError("Serie insuficiente para generar características")

    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)

    fechas = pd.date_range(series.index[-1] + pd.tseries.frequencies.to_offset(freq),
                           periods=steps, freq=freq)
    hist = series.copy()
    feats = []
    for dt in fechas:
        lag1 = float(hist.iloc[-1])
        lag2 = float(hist.iloc[-2] if len(hist) > 1 else lag1)
        mv3 = float(hist.iloc[-3:].mean() if len(hist) >= 3 else lag1)
        feats.append([dt.month, dt.dayofyear, lag1, lag2, mv3])
        hist = pd.concat([hist, pd.Series(lag1, index=[dt])])
    Xf = pd.DataFrame(feats, columns=["Mes", "DiaDelAnio", "Lag1", "Lag2", "MediaMovil3"], index=fechas)

    rf_fc = rf.predict(Xf)

    if m > 1 and len(series) >= 2 * m:
        wes = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=m).fit()
    else:
        wes = ExponentialSmoothing(series, trend="add", seasonal=None).fit()
    wes_fc = wes.forecast(steps)

    fc_df = pd.DataFrame({"WES": wes_fc, "RandomForest": rf_fc}, index=fechas)
    fc_df["Stacking"] = fc_df.mean(axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="Histórico"))
    fig.add_trace(go.Scatter(x=fechas, y=fc_df["Stacking"], mode="lines", name="Pronóstico"))
    fig.update_layout(title="Pronóstico", xaxis_title="Fecha", yaxis_title="Valor")

    return {"forecast": fc_df, "figure": fig}

def forecast_from_file(upload, steps: int = 6) -> Dict[str, Any]:
    """Convenience wrapper that loads a series from ``upload`` and forecasts."""
    series = load_series(upload)
    return generate_forecast(series, steps)
