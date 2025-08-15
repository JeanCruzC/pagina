"""Reusable Erlang calculation utilities.

This module contains helper functions originally written for a Streamlit
prototype.  They are copied here so that Flask views or other backend
components can reuse the pure calculation logic without depending on the
Streamlit runtime.  Expensive helpers are cached using
:func:`functools.lru_cache` to mimic ``st.cache_data`` behaviour.
"""

from __future__ import annotations

from functools import lru_cache
import math

# Color and icon constants kept for backwards compatibility
BUSY_COLOR = "#EF476F"
AVAILABLE_COLOR = "#06D6A0"
QUEUE_SHORT_COLOR = "#06D6A0"
QUEUE_MED_COLOR = "#FFD166"
QUEUE_LONG_COLOR = "#EF476F"

GOOD_COLOR = "#06D6A0"
WARN_COLOR = "#FFD166"
BAD_COLOR = "#EF476F"

BUSY_ICON = "📞"
AVAILABLE_ICONS = ["👨‍💼", "👩‍💼"]
QUEUE_ICON = "🧑‍🤝‍🧑"
PLACEHOLDER_ICON = "❔"
PLACEHOLDER_COLOR = "#B0BEC5"


@lru_cache(maxsize=None)
def factorial_approx(n: int) -> float:
    """Approximate factorial using Stirling's formula for large ``n``."""
    if n < 170:
        return math.factorial(int(n))
    return math.sqrt(2 * math.pi * n) * (n / math.e) ** n


@lru_cache(maxsize=None)
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


@lru_cache(maxsize=None)
def erlang_c(traffic: float, agents: int) -> float:
    """Erlang C waiting probability."""
    agents = int(agents)
    if agents <= traffic:
        return 1.0
    eb = erlang_b(traffic, agents)
    rho = traffic / agents
    if rho >= 1:
        return 1.0
    return eb / (1 - rho + rho * eb)


@lru_cache(maxsize=None)
def service_level_erlang_c(forecast: float, aht: float, agents: int, awt: float) -> float:
    """Calculate service level given arrival rate, AHT, agents and AWT."""
    traffic = forecast * aht
    agents = int(agents)
    if agents <= traffic:
        return 0.0
    pc = erlang_c(traffic, agents)
    if pc == 0:
        return 1.0
    exp_factor = math.exp(-(agents - traffic) * awt / aht)
    return 1 - pc * exp_factor


@lru_cache(maxsize=None)
def waiting_time_erlang_c(forecast: float, aht: float, agents: int) -> float:
    """Average waiting time (ASA) under Erlang C."""
    traffic = forecast * aht
    agents = int(agents)
    if agents <= traffic:
        return float("inf")
    pc = erlang_c(traffic, agents)
    return (pc * aht) / (agents - traffic)


@lru_cache(maxsize=None)
def occupancy_erlang_c(forecast: float, aht: float, agents: int) -> float:
    """Agent occupancy under Erlang C."""
    traffic = forecast * aht
    agents = int(agents)
    return min(traffic / agents, 1.0)


@lru_cache(maxsize=None)
def erlang_x_abandonment(forecast: float, aht: float, agents: int, lines: int, patience: float) -> float:
    """Abandonment probability for Erlang X."""
    traffic = forecast * aht
    agents = int(agents)
    if patience == 0:
        return erlang_b(traffic, lines)
    if agents >= traffic:
        pc = erlang_c(traffic, agents)
        avg_wait = waiting_time_erlang_c(forecast, aht, agents)
        return pc * (1 - math.exp(-avg_wait / patience))
    return min(1.0, traffic / lines)

