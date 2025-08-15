"""Utilities for Erlang O (outbound) calculations."""
from __future__ import annotations

import math


def productivity(agents: int, hours_per_day: float, calls_per_hour: float, success_rate: float = 0.3) -> dict:
    """Return productivity metrics for outbound campaigns."""
    total_calls = agents * hours_per_day * calls_per_hour
    successful_calls = total_calls * success_rate
    return {
        "total_calls": total_calls,
        "successful_calls": successful_calls,
        "success_rate": success_rate,
        "calls_per_agent_day": hours_per_day * calls_per_hour,
        "successful_per_agent_day": hours_per_day * calls_per_hour * success_rate,
    }


def agents_for_target(
    target_calls_day: float,
    hours_per_day: float,
    calls_per_hour: float,
    success_rate: float = 0.3,
) -> int:
    """Return agents required to hit a target of successful calls per day."""
    calls_per_agent_day = hours_per_day * calls_per_hour * success_rate
    if calls_per_agent_day <= 0:
        return 0
    return math.ceil(target_calls_day / calls_per_agent_day)


def dialer_ratio(
    answer_rate: float = 0.25,
    agent_talk_time: float = 5.0,
    wait_between_calls: float = 2.0,
) -> float:
    """Return recommended predictive dialer ratio."""
    cycle_time = agent_talk_time + wait_between_calls
    ratio = cycle_time / (agent_talk_time * answer_rate)
    return max(1.0, ratio)


__all__ = ["productivity", "agents_for_target", "dialer_ratio"]
