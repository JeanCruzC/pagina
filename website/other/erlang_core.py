"""Core utilities for Erlang related calculations and charts.

This module provides pure functions that operate on simple data structures
(lists, dictionaries) and always return serialisable results. Plotly is used
for visualisation so that the resulting figures can be easily converted to
JSON on the frontend.
"""
from __future__ import annotations

from typing import List, Dict, Any, Iterable

import numpy as np
import plotly.graph_objects as go
import pandas as pd
import math

from ..services import erlang as erlang_service
from ..logic.erlang import erlang_x_abandonment


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


def load_demand_from_excel(file_stream) -> List[List[float]]:
    """Load demand data from an Excel file-like object."""
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


# ---------------------------------------------------------------------------
# Erlang C helper functions
# ---------------------------------------------------------------------------

def erlang_b(traffic: float, agents: int) -> float:
    """Erlang B blocking probability."""
    agents = int(agents)
    if agents == 0:
        return 1.0
    if traffic == 0:
        return 0.0
    b = 1.0
    for i in range(1, agents + 1):
        b = (traffic * b) / (i + traffic * b)
    return b


def erlang_c(traffic: float, agents: int) -> float:
    """Erlang C waiting probability."""
    agents = int(agents)
    if agents <= 0:
        return 1.0
    if agents <= traffic:
        return 1.0
    eb = erlang_b(traffic, agents)
    rho = traffic / agents
    if rho >= 1:
        return 1.0
    return eb / (1 - rho + rho * eb)


def service_level_erlang_c(arrival_rate: float, aht: float, agents: int, awt: float) -> float:
    """Return service level for given parameters using Erlang C."""
    traffic = arrival_rate * aht
    agents = int(agents)
    if agents <= traffic:
        return 0.0
    pc = erlang_c(traffic, agents)
    if pc == 0:
        return 1.0
    exp_factor = math.exp(-(agents - traffic) * awt / aht)
    return 1 - pc * exp_factor


def waiting_time_erlang_c(arrival_rate: float, aht: float, agents: int) -> float:
    """Average speed of answer using Erlang C."""
    traffic = arrival_rate * aht
    agents = int(agents)
    if agents <= traffic:
        return float("inf")
    pc = erlang_c(traffic, agents)
    return (pc * aht) / (agents - traffic)


def occupancy_erlang_c(arrival_rate: float, aht: float, agents: int) -> float:
    """Agent occupancy ratio."""
    traffic = arrival_rate * aht
    agents = int(agents)
    if agents <= 0:
        return 1.0
    return min(traffic / agents, 1.0)


def required_agents_for_service_level(
    arrival_rate: float, aht: float, awt: float, sl_target: float, max_agents: int
) -> int:
    """Compute minimal agents to meet ``sl_target`` or ``max_agents`` if not reached."""
    for agents in range(1, int(max_agents) + 1):
        sl = service_level_erlang_c(arrival_rate, aht, agents, awt)
        if sl >= sl_target:
            return agents
    return int(max_agents)


def build_sensitivity_figure(
    arrival_rate: float,
    aht: float,
    awt: float,
    actual_agents: int,
    recommended_agents: int,
    lines: int | None = None,
    patience: float | None = None,
) -> go.Figure:
    """Create sensitivity chart with vertical reference lines.

    The chart plots Service Level and ASA against a range of agent counts and
    highlights the *Actual* and *Recomendado* values using vertical dashed
    lines so that the frontend can render identical visuals whether using the
    Python figure directly or its JSON representation in JavaScript.
    """

    rec = max(int(recommended_agents), 1)
    agent_min = max(1, int(rec * 0.7))
    agent_max = max(agent_min + 1, int(rec * 1.5))
    agent_range = list(range(agent_min, agent_max + 1))

    sl_data = [
        erlang_service.sla_x(arrival_rate, aht, a, awt, lines, patience)
        for a in agent_range
    ]
    asa_data = [waiting_time_erlang_c(arrival_rate, aht, a) for a in agent_range]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agent_range,
            y=sl_data,
            mode="lines+markers",
            name="Service Level",
            yaxis="y",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agent_range,
            y=asa_data,
            mode="lines+markers",
            name="ASA (seg)",
            yaxis="y2",
            line=dict(color="red"),
        )
    )

    fig.update_layout(
        title="Service Level y ASA vs Número de Agentes",
        xaxis_title="Número de Agentes",
        yaxis=dict(title="Service Level", side="left", range=[0, 1]),
        yaxis2=dict(title="ASA (segundos)", side="right", overlaying="y"),
        hovermode="x unified",
    )

    fig.add_vline(
        x=int(actual_agents),
        line_dash="dash",
        line_color="red",
        annotation_text="Actual",
    )
    fig.add_vline(
        x=int(recommended_agents),
        line_dash="dash",
        line_color="orange",
        annotation_text="Recomendado",
    )

    return fig


def calculate_erlang_metrics(
    calls: float,
    aht: float,
    awt: float,
    agents: int,
    sl_target: float = 0.8,
    lines: int | None = None,
    patience: float | None = None,
    interval_seconds: float = 3600.0,
) -> Dict[str, float]:
    """Compute Erlang metrics for the web interface."""

    arrival_rate = calls / interval_seconds if interval_seconds else 0.0

    service_level = erlang_service.sla_x(
        arrival_rate, aht, agents, awt, lines, patience
    )
    asa = waiting_time_erlang_c(arrival_rate, aht, agents)
    occ = occupancy_erlang_c(arrival_rate, aht, agents)
    required = erlang_service.agents_for_sla(
        sl_target, arrival_rate, aht, awt, lines, patience
    )

    metrics = {
        "service_level": service_level,
        "asa": asa,
        "occupancy": occ,
        "required_agents": int(required),
        "calls_per_agent_req": calls / required if required else 0.0,
        "sl_class": "success" if service_level >= 0.8 else "warning" if service_level >= 0.7 else "danger",
        "asa_class": "success" if asa <= 30 else "warning" if asa <= 60 else "danger",
        "occ_class": "success" if 0.7 <= occ <= 0.85 else "warning",
    }

    if lines and patience:
        abandon_rate = erlang_x_abandonment(
            arrival_rate, aht, agents, int(lines), patience
        )
        metrics.update(
            {
                "abandonment": abandon_rate,
                "abandon_class": "success"
                if abandon_rate <= 0.05
                else "warning"
                if abandon_rate <= 0.1
                else "danger",
            }
        )

    metrics["figure"] = build_sensitivity_figure(
        arrival_rate, aht, awt, agents, required, lines, patience
    )

    return metrics
