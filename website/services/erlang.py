"""High level Erlang calculation services.

This module exposes small wrappers around the core Erlang utilities so that
Flask views or other backend components can reuse the pure calculation logic
initially implemented in the Streamlit scripts.  Expensive helpers are cached
with :func:`functools.lru_cache` to speed up repeated calls during a single
process lifetime.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from scipy import optimize

from ..logic.erlang import (
    service_level_erlang_c,
    waiting_time_erlang_c,
    erlang_b,
    erlang_x_abandonment,
)


# ---------------------------------------------------------------------------
# Core helpers replicated from X, CHAT and BL modules in the Streamlit app.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _sla_x_cached(
    forecast: float,
    aht: float,
    agents: int,
    awt: float,
    lines: int | None,
    patience: float | None,
) -> float:
    """Cached implementation of ``X.SLA.calculate``.

    The function supports optional trunk ``lines`` and caller ``patience``
    parameters.  When both are ``None`` the classic Erlang C service level is
    returned.  When ``lines`` is provided the arrival rate is adjusted by the
    Erlang B blocking probability.  If ``patience`` is also given an Erlang X
    abandonment rate is applied to the base service level.
    """

    if lines is None and patience is None:
        return service_level_erlang_c(forecast, aht, agents, awt)

    traffic = forecast * aht
    if lines is not None and patience is None:
        blocking = erlang_b(traffic, lines)
        effective_forecast = forecast * (1 - blocking)
        return service_level_erlang_c(effective_forecast, aht, agents, awt)

    # Both ``lines`` and ``patience`` provided => Erlang X model
    if agents <= traffic:
        return 0.0
    base_sl = service_level_erlang_c(forecast, aht, agents, awt)
    abandon_rate = erlang_x_abandonment(
        forecast, aht, agents, lines or 999, patience or 999
    )
    return base_sl * (1 - abandon_rate * 0.5)


def sla_x(
    forecast: float,
    aht: float,
    agents: int,
    awt: float,
    lines: int | None = None,
    patience: float | None = None,
) -> float:
    """Public wrapper for the cached ``_sla_x_cached`` helper."""

    return _sla_x_cached(forecast, aht, agents, awt, lines, patience)


@lru_cache(maxsize=None)
def agents_for_sla(
    sl_target: float,
    forecast: float,
    aht: float,
    awt: float,
    lines: int | None = None,
    patience: float | None = None,
) -> int:
    """Return number of agents required to achieve ``sl_target``.

    This mirrors ``X.AGENTS.for_sla`` from the original Streamlit script.
    """

    traffic = forecast * aht

    def objective(x: float) -> float:
        agents = int(round(x))
        if agents <= 0:
            return float("inf")
        sl = sla_x(forecast, aht, agents, awt, lines, patience)
        return abs(sl - sl_target)

    result = optimize.minimize_scalar(
        objective, bounds=(traffic * 0.5, traffic * 3), method="bounded"
    )
    return max(1, round(result.x, 1))


def chat_sla(
    forecast: float,
    aht_list: Sequence[float],
    agents: int,
    awt: float,
    lines: int | None = None,
    patience: float | None = None,
) -> float:
    """Service level for multi-channel chat interactions (``CHAT.sla``).

    ``aht_list`` contains the AHT for each simultaneous chat handled by an
    agent.  ``lines`` and ``patience`` are accepted for API compatibility but
    ignored in this simplified model.
    """

    return _chat_sla_cached(forecast, tuple(aht_list), agents, awt)


@lru_cache(maxsize=None)
def _chat_sla_cached(
    forecast: float,
    aht_tuple: Sequence[float],
    agents: int,
    awt: float,
) -> float:
    parallel_capacity = len(aht_tuple)
    avg_aht = sum(aht_tuple) / parallel_capacity
    effectiveness = 0.7 + (0.3 / parallel_capacity)
    effective_agents = agents * parallel_capacity * effectiveness
    return service_level_erlang_c(forecast, avg_aht, effective_agents, awt)


def chat_agents_for_sla(
    sl_target: float,
    forecast: float,
    aht_list: Sequence[float],
    awt: float,
    lines: int | None = None,
    patience: float | None = None,
) -> int:
    """Agents required to hit ``sl_target`` in chat scenarios."""

    return _chat_agents_for_sla_cached(sl_target, forecast, tuple(aht_list), awt)


@lru_cache(maxsize=None)
def _chat_agents_for_sla_cached(
    sl_target: float,
    forecast: float,
    aht_tuple: Sequence[float],
    awt: float,
) -> int:
    parallel_capacity = len(aht_tuple)
    avg_aht = sum(aht_tuple) / parallel_capacity
    effectiveness = 0.7 + (0.3 / parallel_capacity)

    def objective(x: float) -> float:
        if x <= 0:
            return float("inf")
        effective_agents = x * parallel_capacity * effectiveness
        sl = service_level_erlang_c(forecast, avg_aht, effective_agents, awt)
        return abs(sl - sl_target)

    traffic = forecast * avg_aht
    result = optimize.minimize_scalar(
        objective, bounds=(0.1, traffic), method="bounded"
    )
    return max(1, round(result.x, 1))


def bl_sla(
    forecast: float,
    aht: float,
    agents: int,
    awt: float,
    lines: int | None,
    patience: float | None,
    threshold: float,
) -> float:
    """Service level for a blending scenario (``BL.sla``)."""

    return _bl_sla_cached(forecast, aht, agents, awt, lines, patience, threshold)


@lru_cache(maxsize=None)
def _bl_sla_cached(
    forecast: float,
    aht: float,
    agents: int,
    awt: float,
    lines: int | None,
    patience: float | None,
    threshold: float,
) -> float:
    available_agents = max(0, agents - threshold)
    if available_agents <= 0:
        return 0.0
    return service_level_erlang_c(forecast, aht, available_agents, awt)


def bl_optimal_threshold(
    forecast: float,
    aht: float,
    agents: int,
    awt: float,
    lines: int | None,
    patience: float | None,
    sl_target: float,
) -> float:
    """Optimal threshold for blending (``BL.optimal_threshold``)."""

    return _bl_optimal_threshold_cached(
        forecast, aht, agents, awt, lines, patience, sl_target
    )


@lru_cache(maxsize=None)
def _bl_optimal_threshold_cached(
    forecast: float,
    aht: float,
    agents: int,
    awt: float,
    lines: int | None,
    patience: float | None,
    sl_target: float,
) -> float:
    def objective(threshold: float) -> float:
        if threshold < 0 or threshold > agents:
            return float("inf")
        sl = _bl_sla_cached(forecast, aht, agents, awt, lines, patience, threshold)
        return abs(sl - sl_target)

    result = optimize.minimize_scalar(
        objective, bounds=(0, agents), method="bounded"
    )
    return max(0, round(result.x, 1))


__all__ = [
    "service_level_erlang_c",
    "waiting_time_erlang_c",
    "sla_x",
    "agents_for_sla",
    "chat_sla",
    "chat_agents_for_sla",
    "bl_sla",
    "bl_optimal_threshold",
]
