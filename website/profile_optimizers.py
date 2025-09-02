"""Deprecated profile optimizers.

This module remains for backward compatibility. The application now uses
``scheduler.optimize_jean_search`` as the canonical implementation. The
function below simply forwards to that implementation.
"""

from __future__ import annotations


def optimize_jean_search(
    shifts_coverage,
    demand_matrix,
    *,
    cfg=None,
    target_coverage: float = 98.0,
    max_iterations: int = 7,
    verbose: bool = False,
    agent_limit_factor: int = 30,
    excess_penalty: float = 5.0,
    peak_bonus: float = 2.0,
    critical_bonus: float = 2.5,
    time_limit_seconds=None,
    iteration_time_limit: int = 45,
    job_id=None,
):
    """Wrapper for :func:`website.scheduler.optimize_jean_search`.

    The ``cfg`` parameter is accepted for legacy compatibility but ignored.
    All other parameters mirror the scheduler version.
    """

    from .scheduler import optimize_jean_search as _sched_opt

    return _sched_opt(
        shifts_coverage,
        demand_matrix,
        target_coverage=target_coverage,
        max_iterations=max_iterations,
        verbose=verbose,
        agent_limit_factor=agent_limit_factor,
        excess_penalty=excess_penalty,
        peak_bonus=peak_bonus,
        critical_bonus=critical_bonus,
        time_limit_seconds=time_limit_seconds,
        iteration_time_limit=iteration_time_limit,
        job_id=job_id,
    )


# Mapping retained for compatibility with older imports
PROFILE_OPTIMIZERS = {"JEAN": optimize_jean_search}

