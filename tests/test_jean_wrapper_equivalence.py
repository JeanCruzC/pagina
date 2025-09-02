import os
import sys
import types
import warnings
import numpy as np
import importlib

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)


def test_optimize_jean_search_wrapper_equivalence():
    sys.modules.pop("website.scheduler", None)
    sys.modules.pop("website.profile_optimizers", None)
    sys.modules.setdefault("pandas", types.SimpleNamespace(DataFrame=object, Series=object))

    import website
    scheduler = importlib.import_module("website.scheduler")
    website.scheduler = scheduler
    profile_optimizers = importlib.import_module("website.profile_optimizers")

    shifts_coverage = {"S1": np.zeros((7, 1), dtype=int)}
    shifts_coverage["S1"][0, 0] = 1
    demand_matrix = np.zeros((7, 1), dtype=int)
    demand_matrix[0, 0] = 1

    cfg = {
        "agent_limit_factor": 10,
        "excess_penalty": 3.0,
        "peak_bonus": 1.0,
        "critical_bonus": 2.0,
        "TARGET_COVERAGE": 97.0,
        "max_iterations": 3,
        "time_limit_seconds": 20,
    }

    expected = scheduler.optimize_jean_search(
        shifts_coverage,
        demand_matrix,
        agent_limit_factor=cfg["agent_limit_factor"],
        excess_penalty=cfg["excess_penalty"],
        peak_bonus=cfg["peak_bonus"],
        critical_bonus=cfg["critical_bonus"],
        target_coverage=cfg["TARGET_COVERAGE"],
        max_iterations=cfg["max_iterations"],
        iteration_time_limit=cfg["time_limit_seconds"],
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        result = profile_optimizers.optimize_jean_search(
            shifts_coverage, demand_matrix, cfg=cfg
        )

    assert result == expected
