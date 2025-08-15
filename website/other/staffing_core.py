"""Staffing optimisation helpers."""
from __future__ import annotations

from typing import Dict, Any, Iterable, List

import plotly.graph_objects as go

from ..services.erlang import service_level_erlang_c, waiting_time_erlang_c, agents_for_sla


def staffing_optimizer(
    forecasts: Iterable[float],
    aht: float,
    interval_seconds: int,
    sl_target: float,
    awt: float = 20.0,
) -> Dict[str, Any]:
    """Return per-hour staffing recommendations."""
    hours = list(range(len(list(forecasts))))
    forecasts = list(forecasts)
    table: List[Dict[str, Any]] = []
    total_agent_hours = 0
    agent_series: List[float] = []

    for hour, forecast in zip(hours, forecasts):
        arrival_rate = forecast / interval_seconds
        agents_needed = agents_for_sla(sl_target, arrival_rate, aht, awt)
        sl = service_level_erlang_c(arrival_rate, aht, agents_needed, awt)
        asa = waiting_time_erlang_c(arrival_rate, aht, agents_needed)
        table.append(
            {
                "Hora": f"{hour:02d}:00",
                "Forecast": round(forecast, 2),
                "Agentes": agents_needed,
                "SL": round(sl, 3),
                "ASA": round(asa, 2),
            }
        )
        total_agent_hours += agents_needed
        agent_series.append(agents_needed)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=forecasts, name="Forecast"))
    fig.add_trace(go.Scatter(x=hours, y=agent_series, name="Agentes"))
    fig.update_layout(
        title="Forecast vs Agentes Necesarios",
        xaxis_title="Hora",
        yaxis_title="Cantidad",
    )

    summary = {
        "max_agents": max(agent_series) if agent_series else 0,
        "min_agents": min(agent_series) if agent_series else 0,
        "avg_agents": total_agent_hours / len(agent_series) if agent_series else 0,
        "total_agent_hours": total_agent_hours,
    }

    return {"table": table, "figure": fig.to_dict(), "summary": summary}


__all__ = ["staffing_optimizer"]
