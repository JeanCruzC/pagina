from __future__ import annotations

from typing import Dict, Any, List

import plotly.graph_objects as go

from .erlang_core import service_level_erlang_c, occupancy_erlang_c


def calculate_blending_metrics(
    forecast: float,
    aht: float,
    agents: int,
    awt: float,
    lines: int,
    patience: float,
    threshold: int,
) -> Dict[str, Any]:
    """Compute blending metrics and sensitivity figure.

    Parameters
    ----------
    forecast: float
        Forecasted inbound calls for an hour.
    aht: float
        Average handle time in seconds.
    agents: int
        Total available agents.
    awt: float
        Service level threshold (seconds).
    lines: int
        Unused parameter for compatibility.
    patience: float
        Unused parameter for compatibility.
    threshold: int
        Agents reserved for outbound work.
    """
    interval = 3600.0
    arrival_rate = forecast / interval if interval else 0.0

    available_agents = max(0, agents - threshold)
    sla = service_level_erlang_c(arrival_rate, aht, available_agents, awt)
    occupancy = occupancy_erlang_c(arrival_rate, aht, available_agents)

    inbound_traffic = arrival_rate * aht
    outbound_agents = max(0.0, agents - threshold - inbound_traffic)
    outbound_capacity = outbound_agents * interval / aht if aht else 0.0

    optimal_threshold = 0
    best_capacity = -1.0
    for t in range(0, agents + 1):
        avail = max(0, agents - t)
        sl = service_level_erlang_c(arrival_rate, aht, avail, awt)
        if sl >= 0.8:  # maintain 80% SLA
            out_agents = max(0.0, agents - t - inbound_traffic)
            cap = out_agents * interval / aht if aht else 0.0
            if cap > best_capacity:
                best_capacity = cap
                optimal_threshold = t

    threshold_range: List[int] = list(range(0, min(agents, int(agents * 0.4)) + 1))
    sl_data: List[float] = []
    out_data: List[float] = []
    for t in threshold_range:
        avail = max(0, agents - t)
        sl_val = service_level_erlang_c(arrival_rate, aht, avail, awt)
        out_agents = max(0.0, agents - t - inbound_traffic)
        out_cap = out_agents * interval / aht if aht else 0.0
        sl_data.append(sl_val)
        out_data.append(out_cap)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=threshold_range,
            y=sl_data,
            mode="lines+markers",
            name="Service Level Inbound",
            yaxis="y",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=threshold_range,
            y=out_data,
            mode="lines+markers",
            name="Capacidad Outbound",
            yaxis="y2",
            line=dict(color="green"),
        )
    )
    fig.update_layout(
        title="Service Level vs Capacidad Outbound por Threshold",
        xaxis_title="Threshold (Agentes Reservados)",
        yaxis=dict(title="Service Level Inbound", side="left", range=[0, 1]),
        yaxis2=dict(
            title="Capacidad Outbound (llamadas/hora)",
            side="right",
            overlaying="y",
        ),
        hovermode="x unified",
    )
    if threshold in threshold_range:
        fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Actual")
    if optimal_threshold in threshold_range:
        fig.add_vline(
            x=optimal_threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text="Óptimo",
        )

    return {
        "service_level": sla,
        "outbound_capacity": outbound_capacity,
        "occupancy": occupancy,
        "optimal_threshold": optimal_threshold,
        "figure": fig.to_json(),
    }
