import numpy as np

from website import profile_optimizers, scheduler


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

    sched_res = scheduler.optimize_jean_search(shifts, demand, **params)
    profile_res = profile_optimizers.optimize_jean_search(shifts, demand, **params)
    assert sched_res == profile_res


def test_jean_search_cfg_forwarding():
    shifts, demand = _sample_data()
    cfg = {
        "TARGET_COVERAGE": 97.0,
        "max_iterations": 1,
        "iteration_time_limit": 1,
        "agent_limit_factor": 4,
        "excess_penalty": 3.0,
    }

    sched_res = scheduler.optimize_jean_search(
        shifts,
        demand,
        target_coverage=cfg["TARGET_COVERAGE"],
        max_iterations=cfg["max_iterations"],
        iteration_time_limit=cfg["iteration_time_limit"],
        agent_limit_factor=cfg["agent_limit_factor"],
        excess_penalty=cfg["excess_penalty"],
    )
    profile_res = profile_optimizers.optimize_jean_search(
        shifts,
        demand,
        cfg=cfg,
    )

    assert profile_res == sched_res


def test_jean_search_default_params_match():
    import inspect

    sched_params = inspect.signature(scheduler.optimize_jean_search).parameters
    profile_params = inspect.signature(profile_optimizers.optimize_jean_search).parameters

    for name, param in sched_params.items():
        assert name in profile_params
        assert profile_params[name].default == param.default
