"""Utilities for Erlang O (outbound) calculations."""
from __future__ import annotations

import math


def total_calls(
    agents: int,
    hours_per_day: float,
    talk_time: float,
    wait_between: float,
    answer_rate: float,
) -> dict:
    """Return call volume metrics for an outbound campaign."""
    cycle = talk_time + wait_between
    if cycle <= 0:
        calls_per_agent_day = 0.0
    else:
        calls_per_agent_day = hours_per_day * 60 / cycle
    successful_calls = agents * calls_per_agent_day
    dialed_calls = successful_calls / answer_rate if answer_rate > 0 else 0.0
    return {
        "calls_per_agent_day": calls_per_agent_day,
        "successful_calls": successful_calls,
        "dialed_calls": dialed_calls,
    }


def agents_needed(
    target_calls_day: float,
    hours_per_day: float,
    talk_time: float,
    wait_between: float,
) -> int:
    """Return agents required to hit a target of successful calls per day."""
    cycle = talk_time + wait_between
    if cycle <= 0:
        return 0
    calls_per_agent_day = hours_per_day * 60 / cycle
    if calls_per_agent_day <= 0:
        return 0
    return math.ceil(target_calls_day / calls_per_agent_day)


def roi(
    agents: int,
    hours_per_day: float,
    talk_time: float,
    wait_between: float,
    answer_rate: float,
    cost_per_agent_day: float,
    revenue_per_call: float,
) -> dict:
    """Return revenue and profitability metrics for a given staffing level."""
    totals = total_calls(agents, hours_per_day, talk_time, wait_between, answer_rate)
    revenue = totals["successful_calls"] * revenue_per_call
    cost = agents * cost_per_agent_day
    profit = revenue - cost
    roi_value = profit / cost if cost else 0.0
    return {
        **totals,
        "revenue": revenue,
        "cost": cost,
        "profit": profit,
        "roi": roi_value,
    }


def dialer_ratio(
    answer_rate: float = 0.25,
    agent_talk_time: float = 5.0,
    wait_between_calls: float = 2.0,
) -> float:
    """Return recommended predictive dialer ratio."""
    cycle_time = agent_talk_time + wait_between_calls
    ratio = cycle_time / (agent_talk_time * answer_rate)
    return max(1.0, ratio)


__all__ = ["total_calls", "agents_needed", "roi", "dialer_ratio"]

