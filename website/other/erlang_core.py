"""Core utilities for Erlang related calculations and charts.

This module provides pure functions that operate on simple data structures
(lists, dictionaries) and always return serialisable results. Plotly is used
for visualisation so that the resulting figures can be easily converted to
JSON on the frontend.
"""
from __future__ import annotations

from pathlib import Path
from typing import IO, List, Dict, Any, Iterable

import numpy as np
import plotly.graph_objects as go
import pandas as pd


def load_demand_matrix(records: Iterable[Dict[str, Any]]) -> List[List[float]]:
    """Create a 7x24 demand matrix from iterable ``records``.

    Each record should provide values for day, hour and demand. The function
    tries to infer the appropriate keys in a case-insensitive manner so that it
    works with a variety of column names coming from Excel exports.

    Parameters
    ----------
    records:
        Iterable of dictionaries with demand data.

    Returns
    -------
    list of list of float
        A serialisable 7x24 matrix (list of lists).
    """
    matrix = [[0.0 for _ in range(24)] for _ in range(7)]

    day_key = time_key = demand_key = None
    records = list(records)
    if records:
        sample = records[0]
        for key in sample.keys():
            kl = key.lower()
            if day_key is None and ("día" in kl or "dia" in kl or kl == "day"):
                day_key = key
            elif time_key is None and ("horario" in kl or "hora" in kl or kl == "time"):
                time_key = key
            elif demand_key is None and ("erlang" in kl or "requeridos" in kl or "demand" in kl):
                demand_key = key
        if day_key is None or time_key is None or demand_key is None:
            for row in records:
                for key in row.keys():
                    kl = key.lower()
                    if day_key is None and ("día" in kl or "dia" in kl or kl == "day"):
                        day_key = key
                    elif time_key is None and ("horario" in kl or "hora" in kl or kl == "time"):
                        time_key = key
                    elif demand_key is None and ("erlang" in kl or "requeridos" in kl or "demand" in kl):
                        demand_key = key
                if day_key and time_key and demand_key:
                    break
    if day_key is None or time_key is None or demand_key is None:
        return matrix

    for row in records:
        try:
            day = int(row[day_key])
            if not 1 <= day <= 7:
                continue
            horario = str(row[time_key])
            hour = int(horario.split(":")[0]) if ":" in horario else int(float(horario))
            if not 0 <= hour <= 23:
                continue
            demand = float(row[demand_key])
            matrix[day - 1][hour] = demand
        except (ValueError, TypeError, KeyError):
            continue

    return matrix


def load_demand_from_excel(file_stream: IO | str | Path) -> List[List[float]]:
    """Load demand data from an Excel file-like object or path."""
    df = pd.read_excel(file_stream)
    return load_demand_matrix(df.to_dict("records"))


def analyze_demand_matrix(matrix: Iterable[Iterable[float]]) -> Dict[str, Any]:
    """Return basic metrics from a demand matrix.

    Parameters
    ----------
    matrix:
        7x24 matrix provided as list of lists or numpy array.

    Returns
    -------
    dict
        All values are JSON serialisable.
    """
    arr = np.array(list(map(list, matrix)), dtype=float)
    daily_demand = arr.sum(axis=1)
    hourly_demand = arr.sum(axis=0)
    active_days = [int(d) for d in range(7) if daily_demand[d] > 0]
    inactive_days = [int(d) for d in range(7) if daily_demand[d] == 0]
    working_days = len(active_days)
    active_hours = np.where(hourly_demand > 0)[0]
    first_hour = int(active_hours.min()) if active_hours.size else 8
    last_hour = int(active_hours.max()) if active_hours.size else 20
    operating_hours = last_hour - first_hour + 1
    peak_demand = float(arr.max()) if arr.size else 0.0
    avg_demand = float(arr[active_days].mean()) if active_days else 0.0
    daily_totals = arr.sum(axis=1)
    hourly_totals = arr.sum(axis=0)
    if daily_totals.size > 1:
        critical_days = list(np.argsort(daily_totals)[-2:].astype(int))
    else:
        critical_days = [int(np.argmax(daily_totals))]
    if np.any(hourly_totals > 0):
        peak_threshold = float(np.percentile(hourly_totals[hourly_totals > 0], 75))
    else:
        peak_threshold = 0.0
    peak_hours = [int(h) for h in np.where(hourly_totals >= peak_threshold)[0]]
    return {
        "daily_demand": daily_demand.tolist(),
        "hourly_demand": hourly_demand.tolist(),
        "active_days": active_days,
        "inactive_days": inactive_days,
        "working_days": working_days,
        "first_hour": first_hour,
        "last_hour": last_hour,
        "operating_hours": operating_hours,
        "peak_demand": peak_demand,
        "average_demand": avg_demand,
        "critical_days": critical_days,
        "peak_hours": peak_hours,
    }


def create_heatmap(matrix: Iterable[Iterable[float]], title: str, colorscale: str = "RdYlBu") -> Dict[str, Any]:
    """Return a Plotly heatmap figure as a serialisable dict."""
    arr = np.array(list(map(list, matrix)))
    fig = go.Figure(data=go.Heatmap(z=arr, colorscale=colorscale, colorbar=dict(title="Agentes")))
    fig.update_layout(title=title, xaxis_title="Hora", yaxis_title="Día")
    return fig.to_dict()


def generate_all_heatmaps(
    demand: Iterable[Iterable[float]],
    coverage: Iterable[Iterable[float]] | None = None,
    diff: Iterable[Iterable[float]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Generate heatmaps for demand, coverage and difference matrices."""
    maps: Dict[str, Dict[str, Any]] = {
        "demand": create_heatmap(demand, "Demanda por Hora y Día", "Reds")
    }
    if coverage is not None:
        maps["coverage"] = create_heatmap(coverage, "Cobertura por Hora y Día", "Blues")
    if diff is not None:
        maps["difference"] = create_heatmap(diff, "Diferencias por Hora y Día", "RdBu")
    return maps
