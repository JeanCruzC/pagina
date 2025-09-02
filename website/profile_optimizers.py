"""Deprecated profile optimizers.

This module remains for backward compatibility. The application now uses
``scheduler.optimize_jean_search`` as the canonical implementation. The
wrapper below forwards to that implementation, mapping legacy ``cfg``
values to the new explicit parameters and emitting a deprecation
warning when invoked.
"""

from __future__ import annotations

import inspect
import warnings
from functools import wraps

from website.scheduler import optimize_jean_search as _sched_opt


_sched_sig = inspect.signature(_sched_opt)
_params = list(_sched_sig.parameters.values())
cfg_param = inspect.Parameter("cfg", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)
_wrapper_sig = _sched_sig.replace(parameters=_params[:2] + [cfg_param] + _params[2:])


@wraps(_sched_opt)
def optimize_jean_search(*args, **kwargs):
    """Compatibility wrapper for :func:`website.scheduler.optimize_jean_search`.

    Accepts the legacy ``cfg`` object, mapping known keys to the modern
    keyword arguments before delegating to the scheduler implementation.
    """

    bound = _wrapper_sig.bind_partial(*args, **kwargs)
    cfg = bound.arguments.pop("cfg", None)
    warnings.warn(
        "profile_optimizers.optimize_jean_search is deprecated; use "
        "scheduler.optimize_jean_search instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if cfg:
        mapping = {
            "agent_limit_factor": "agent_limit_factor",
            "excess_penalty": "excess_penalty",
            "peak_bonus": "peak_bonus",
            "critical_bonus": "critical_bonus",
            "TARGET_COVERAGE": "target_coverage",
            "max_iterations": "max_iterations",
            "time_limit_seconds": "time_limit_seconds",
            "iteration_time_limit": "iteration_time_limit",
        }
        for old, new in mapping.items():
            if old in cfg and new not in bound.arguments:
                bound.arguments[new] = cfg[old]
        if isinstance(cfg, dict) and "job_id" in cfg and "job_id" not in bound.arguments:
            bound.arguments["job_id"] = cfg["job_id"]

    return _sched_opt(*bound.args, **bound.kwargs)


optimize_jean_search.__signature__ = _wrapper_sig


# Mapping retained for compatibility with older imports
PROFILE_OPTIMIZERS = {"JEAN": optimize_jean_search}

