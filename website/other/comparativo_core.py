"""Comparative analysis utilities."""
from __future__ import annotations

from typing import Dict, Any, List

import plotly.express as px

from ..services.erlang import (
    service_level_erlang_c,
    waiting_time_erlang_c,
    sla_x,
    chat_sla,
    chat_asa,
    bl_sla,
    bl_outbound_capacity,
)


def comparative_analysis(
    forecast: float,
    interval_seconds: int,
    aht: float,
    agents: int,
    awt: float,
    lines: int,
    patience: float,
) -> Dict[str, Any]:
    """Return metrics comparing multiple Erlang models."""
    arrival_rate = forecast / interval_seconds

    sl_basic = service_level_erlang_c(arrival_rate, aht, agents, awt)
    asa_basic = waiting_time_erlang_c(arrival_rate, aht, agents)
    occ_basic = arrival_rate * aht / agents if agents else 0.0

    sl_abandon = sla_x(arrival_rate, aht, agents, awt, lines, patience)

    chat_aht = [aht * 0.7, aht * 0.8, aht * 0.9]
    sl_chat = chat_sla(arrival_rate, chat_aht, agents, awt)
    asa_chat = chat_asa(arrival_rate, chat_aht, agents)

    threshold = 3
    sl_blend = bl_sla(arrival_rate, aht, agents, awt, lines, patience, threshold)
    outbound_cap = bl_outbound_capacity(
        arrival_rate, aht, agents, lines, patience, threshold, aht
    )

    table: List[Dict[str, Any]] = [
        {
            "Modelo": "Erlang C",
            "Service Level": f"{sl_basic:.1%}",
            "ASA": f"{asa_basic:.1f}",
            "Ocupacion": f"{occ_basic:.1%}",
            "Características": "Modelo básico, sin abandonment",
        },
        {
            "Modelo": "Erlang X",
            "Service Level": f"{sl_abandon:.1%}",
            "ASA": f"{asa_basic:.1f}",
            "Ocupacion": f"{occ_basic:.1%}",
            "Características": "Con abandonment",
        },
        {
            "Modelo": "Chat Multi-canal",
            "Service Level": f"{sl_chat:.1%}",
            "ASA": f"{asa_chat:.1f}",
            "Ocupacion": f"{occ_basic:.1%}",
            "Características": f"Multi-chat ({len(chat_aht)} simultáneos)",
        },
        {
            "Modelo": "Blending",
            "Service Level": f"{sl_blend:.1%}",
            "ASA": f"{asa_basic:.1f}",
            "Ocupacion": f"{occ_basic:.1%}",
            "Características": f"Outbound: {outbound_cap:.0f} llamadas/h",
        },
    ]

    fig = px.bar(
        x=[row["Modelo"] for row in table],
        y=[sl_basic, sl_abandon, sl_chat, sl_blend],
        labels={"x": "Modelo", "y": "Service Level"},
        title="Comparación de Service Level por Modelo",
    )
    fig.add_hline(y=0.8, line_dash="dash", line_color="red")

    return {"table": table, "figure": fig.to_dict()}


__all__ = ["comparative_analysis"]
