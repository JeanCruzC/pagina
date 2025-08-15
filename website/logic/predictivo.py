"""Utility helpers for the predictive model prototype."""

from __future__ import annotations

from functools import lru_cache
import pandas as pd


@lru_cache(maxsize=None)
def detectar_frecuencia(series: pd.Series) -> str:
    """Infer frequency string from a ``pandas`` series index."""
    freq = pd.infer_freq(series.index)
    if freq:
        return freq
    dias = series.index.to_series().diff().dt.days.mean()
    if dias <= 1.5:
        return "D"
    if dias <= 8:
        return "W"
    return "MS"


@lru_cache(maxsize=None)
def calcular_estacionalidad(series: pd.Series) -> int:
    """Return a simple seasonality estimate based on series length."""
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
    """Build a small feature matrix used in the experimental models."""
    df = pd.DataFrame(index=series.index)
    df["Valor"] = series
    df["Mes"] = df.index.month
    df["DiaDelAnio"] = df.index.dayofyear
    df["Lag1"] = df["Valor"].shift(1)
    df["Lag2"] = df["Valor"].shift(2)
    df["MediaMovil3"] = df["Valor"].rolling(3).mean()
    return df.dropna()

