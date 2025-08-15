"""Time-series helper functions extracted from a Streamlit prototype."""

from __future__ import annotations

from functools import lru_cache
import pandas as pd


@lru_cache(maxsize=None)
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with stripped and lower-cased column names."""
    new_df = df.copy()
    new_df.columns = new_df.columns.str.strip().str.lower()
    return new_df


@lru_cache(maxsize=None)
def week_number(date_series: pd.Series) -> pd.Series:
    """Helper returning ISO week numbers with caching for repeated calls."""
    return date_series.dt.isocalendar().week

