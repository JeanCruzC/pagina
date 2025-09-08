import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from website.scheduler import (
    load_demand_matrix_from_df,
    generate_shifts_coverage_optimized,
    get_profile_optimizer,
)
from website.profiles import normalize_profile

LEGACY_DIR = "./legacy"
REQ_PATH = f"{LEGACY_DIR}/Requerido.xlsx"


def make_demand_matrix_from_legacy():
    df = pd.read_excel(REQ_PATH)
    demand_matrix = load_demand_matrix_from_df(df)
    shifts_coverage = {}
    for batch in generate_shifts_coverage_optimized(demand_matrix, quality_threshold=24):
        shifts_coverage.update(batch)
    return demand_matrix, shifts_coverage


def run_for_profile(profile, demand_matrix, shifts_coverage):
    prof = normalize_profile(profile)
    optimize_fn = get_profile_optimizer(prof)
    t0 = time.time()
    assigns, tag = optimize_fn(shifts_coverage, demand_matrix, cfg={"optimization_profile": prof})
    dt = time.time() - t0
    D = demand_matrix.shape[0]
    cov = np.zeros_like(demand_matrix)
    for s, c in assigns.items():
        slots = len(shifts_coverage[s]) // D
        cov += np.array(shifts_coverage[s]).reshape(D, slots) * int(c)
    deficit = np.maximum(0, demand_matrix - cov).sum()
    excess = np.maximum(0, cov - demand_matrix).sum()
    agents = sum(assigns.values())
    return {
        "profile": profile,
        "time_sec": round(dt, 3),
        "total_deficit": int(deficit),
        "total_excess": int(excess),
        "num_agents": int(agents),
        "tag": tag,
    }


if __name__ == "__main__":
    dm, sc = make_demand_matrix_from_legacy()
    results = []
    for p in ["100% Exacto", "100% Cobertura Eficiente", "100% Cobertura Total", "JEAN"]:
        results.append(run_for_profile(p, dm, sc))
    import csv
    with open(f"{LEGACY_DIR}/profile_regression.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(results)
