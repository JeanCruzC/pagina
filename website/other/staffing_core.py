"""Staffing optimisation helpers."""
from __future__ import annotations

from typing import Dict, Any, Iterable, List


def _parse_hour(value: str | None) -> int:
    """Return hour integer from HH:MM strings."""
    if not value:
        return 0
    try:
        return int(value.split(":", 1)[0])
    except (ValueError, AttributeError):
        return 0

import plotly.graph_objects as go

from ..services.erlang import service_level_erlang_c, waiting_time_erlang_c, agents_for_sla


def staffing_optimizer(
    forecasts: Iterable[float],
    aht: float,
    interval_seconds: int,
    sl_target: float,
    awt: float = 20.0,
    start_time: str | None = None,
    end_time: str | None = None,
    pattern: str = "manual",
) -> Dict[str, Any]:
    """Return per-hour staffing recommendations."""
    predefined_patterns = {
        "call_center": (9, [100, 120, 140, 160, 180, 160, 140, 120, 100]),
        "ecommerce": (8, [80, 100, 130, 160, 180, 170, 160, 150, 140, 120, 100, 80]),
        "soporte": (0, [60] * 24),
    }

    if pattern != "manual" and pattern in predefined_patterns:
        start_hour, pattern_forecasts = predefined_patterns[pattern]
        forecasts = pattern_forecasts
    else:
        start_hour = _parse_hour(start_time)
        forecasts = list(forecasts)
        if end_time:
            end_hour = _parse_hour(end_time)
            length = end_hour - start_hour
            if length > len(forecasts):
                forecasts.extend([0] * (length - len(forecasts)))
    hours = [start_hour + i for i in range(len(forecasts))]
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
    fig.add_trace(go.Scatter(x=hours, y=forecasts, name="Forecast", yaxis="y"))
    fig.add_trace(
        go.Scatter(x=hours, y=agent_series, name="Agentes Requeridos", yaxis="y2")
    )
    fig.update_layout(
        title="Forecast vs Agentes Requeridos",
        xaxis_title="Hora",
        yaxis=dict(title="Forecast"),
        yaxis2=dict(title="Agentes", overlaying="y", side="right"),
    )

    peak_agents = max(agent_series) if agent_series else 0
    valley_agents = min(agent_series) if agent_series else 0
    avg_agents = total_agent_hours / len(agent_series) if agent_series else 0
    summary = {
        "Pico": peak_agents,
        "Valle": valley_agents,
        "Promedio": round(avg_agents, 2),
        "Total agente-horas": total_agent_hours,
    }
    analysis = [
        {"Métrica": "Pico", "Valor": peak_agents},
        {"Métrica": "Valle", "Valor": valley_agents},
        {"Métrica": "Promedio", "Valor": round(avg_agents, 2)},
        {"Métrica": "Total agente-horas", "Valor": total_agent_hours},
    ]
    recommendations: List[str] = []
    if peak_agents > avg_agents * 1.2:
        recommendations.append(
            "Considera turnos escalonados para cubrir horas pico."
        )
    if valley_agents < avg_agents * 0.8:
        recommendations.append("Reducir personal en horas valle.")
    if not recommendations:
        recommendations.append("Distribución de agentes balanceada.")

    return {
        "table": table,
        "figure": fig.to_dict(),
        "summary": summary,
        "analysis": analysis,
        "recommendations": recommendations,
    }


__all__ = ["staffing_optimizer"]
