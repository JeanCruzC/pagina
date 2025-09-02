import numpy as np

import importlib
import importlib.util
import pathlib
import sys


def _scheduler_module():
    """Load the scheduler module ensuring the real implementation is used."""
    if "website.scheduler" in sys.modules and hasattr(sys.modules["website.scheduler"], "optimize_jean_search"):
        return sys.modules["website.scheduler"]

    path = pathlib.Path(__file__).resolve().parents[1] / "website" / "scheduler.py"
    spec = importlib.util.spec_from_file_location("website.scheduler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["website.scheduler"] = module
    return module


scheduler = _scheduler_module()


def _profile_module():
    """Import profile_optimizers after ensuring scheduler module is present."""
    _scheduler_module()
    return importlib.import_module("website.profile_optimizers")


def _sample_data():
    """Create a tiny dataset for testing jean search optimizers."""
    hours = 2
    demand = np.zeros((7, hours), dtype=int)
    demand[0, 0] = 1
    demand[1, 1] = 1

    coverage = np.zeros(7 * hours, dtype=int)
    coverage[0] = 1  # day 0, hour 0
    coverage[3] = 1  # day 1, hour 1

    shifts = {"s1": coverage}
    return shifts, demand


def test_jean_search_equivalence_results():
    shifts, demand = _sample_data()
    params = {
        "max_iterations": 1,
        "iteration_time_limit": 1,
        "agent_limit_factor": 5,
    }

    profile_optimizers = _profile_module()
    sched_res = scheduler.optimize_jean_search(shifts, demand, **params)
    profile_res = profile_optimizers.optimize_jean_search(shifts, demand, **params)
    assert sched_res == profile_res


def test_jean_search_default_params_match():
    import inspect

    profile_optimizers = _profile_module()
    sched_params = inspect.signature(scheduler.optimize_jean_search).parameters
    profile_params = inspect.signature(profile_optimizers.optimize_jean_search).parameters

    for name, param in sched_params.items():
        assert name in profile_params
        assert profile_params[name].default == param.default
