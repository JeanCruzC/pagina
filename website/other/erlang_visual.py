"""Visual helpers for Erlang calculators."""
from __future__ import annotations

import math
from typing import Optional

import plotly.graph_objects as go

from .erlang_core import (
    service_level_erlang_c,
    waiting_time_erlang_c,
    occupancy_erlang_c,
)

# Visual constants replicated from the Streamlit version
BUSY_COLOR = "#EF476F"
AVAILABLE_COLOR = "#06D6A0"
QUEUE_SHORT_COLOR = "#06D6A0"
QUEUE_MED_COLOR = "#FFD166"
QUEUE_LONG_COLOR = "#EF476F"

BUSY_ICON = "📞"
AVAILABLE_ICONS = ["👨‍💼", "👩‍💼"]
QUEUE_ICON = "🧑‍🤝‍🧑"
PLACEHOLDER_ICON = "❔"
PLACEHOLDER_COLOR = "#B0BEC5"


def create_agent_visualization(
    forecast: float,
    aht: float,
    agents: int,
    awt: float,
    interval_seconds: int = 3600,
    required_agents: Optional[int] = None,
) -> go.Figure:
    """Return Plotly figure with agent states and queue simulation."""
    arrival_rate = forecast / interval_seconds
    sl = service_level_erlang_c(arrival_rate, aht, agents, awt)
    asa = waiting_time_erlang_c(arrival_rate, aht, agents)
    occupancy = occupancy_erlang_c(arrival_rate, aht, agents)

    agents_per_row = 10
    display_agents = max(agents, required_agents or agents)
    rows = math.ceil(display_agents / agents_per_row)

    base_y = 1
    queue_y = 0
    metrics_y = base_y + rows + 1

    fig = go.Figure()

    busy_agents = int(agents * occupancy)
    available_agents = agents - busy_agents
    placeholder_agents = (
        max(0, (required_agents or agents) - agents)
        if required_agents is not None
        else 0
    )
    agent_states = (
        ["busy"] * busy_agents
        + ["available"] * available_agents
        + ["missing"] * placeholder_agents
    )

    fig.add_annotation(
        x=agents_per_row * 1.2 / 2,
        y=base_y + rows + 0.6,
        text=f"{busy_agents}/{agents} agentes ocupados",
        showarrow=False,
        font=dict(size=12),
    )

    agent_x = []
    agent_y = []
    agent_icons = []
    agent_colors = []
    avail_index = 0

    for i, state in enumerate(agent_states):
        row = i // agents_per_row
        col = i % agents_per_row
        x = col * 1.2
        y = base_y + rows - row - 1
        agent_x.append(x)
        agent_y.append(y)
        if state == "busy":
            agent_icons.append(BUSY_ICON)
            agent_colors.append(BUSY_COLOR)
        elif state == "available":
            agent_icons.append(AVAILABLE_ICONS[avail_index % len(AVAILABLE_ICONS)])
            avail_index += 1
            agent_colors.append(AVAILABLE_COLOR)
        else:
            agent_icons.append(PLACEHOLDER_ICON)
            agent_colors.append(PLACEHOLDER_COLOR)

    fig.add_trace(
        go.Scatter(
            x=agent_x,
            y=agent_y,
            mode="text",
            text=agent_icons,
            textfont=dict(size=20, color=agent_colors),
            showlegend=False,
        )
    )

    queue_length = max(0, int((1 - sl) * forecast))
    queue_colors = (
        QUEUE_SHORT_COLOR if queue_length < 3 else QUEUE_MED_COLOR if queue_length < 6 else QUEUE_LONG_COLOR
    )
    queue_icons = [QUEUE_ICON] * queue_length
    fig.add_trace(
        go.Scatter(
            x=list(range(queue_length)),
            y=[queue_y] * queue_length,
            mode="text",
            text=queue_icons,
            textfont=dict(size=16, color=queue_colors),
            showlegend=False,
        )
    )

    fig.add_annotation(
        x=agents_per_row * 1.2 / 2,
        y=metrics_y,
        text=f"SL {sl:.1%} | ASA {asa:.1f}s | OCC {occupancy:.1%}",
        showarrow=False,
    )

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=200 + rows * 30,
        margin=dict(l=10, r=10, t=10, b=10),
    )

    return fig


__all__ = [
    "create_agent_visualization",
    "BUSY_COLOR",
    "AVAILABLE_COLOR",
    "QUEUE_SHORT_COLOR",
    "QUEUE_MED_COLOR",
    "QUEUE_LONG_COLOR",
    "BUSY_ICON",
    "AVAILABLE_ICONS",
    "QUEUE_ICON",
    "PLACEHOLDER_ICON",
    "PLACEHOLDER_COLOR",
]
