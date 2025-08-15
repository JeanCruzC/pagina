"""Core utilities for KPI calculations and visualisations."""
from __future__ import annotations

from typing import Dict, Iterable, Any

import numpy as np
import plotly.graph_objects as go


def read_assignments(data: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Convert an iterable of assignment records into a dictionary.

    Each record is expected to have ``shift`` and ``count`` keys. The function
    returns a simple ``{shift: count}`` mapping.
    """
    assignments: Dict[str, int] = {}
    for row in data:
        try:
            assignments[str(row["shift"])] = int(row["count"])
        except (KeyError, TypeError, ValueError):
            continue
    return assignments


def analyze_results(assignments, shifts_coverage, demand_matrix, coverage_matrix=None):
    """Compute coverage metrics from solved assignments.

    Parameters use basic Python types and the resulting dictionary only contains
    serialisable objects (lists, ints, floats).
    """
    if not assignments:
        return None

    compute_coverage = coverage_matrix is None
    if compute_coverage:
        slots_per_day = (
            (len(next(iter(shifts_coverage.values()))) // 7) * 8
            if shifts_coverage
            else 24
        )
        coverage_matrix = np.zeros((7, slots_per_day), dtype=np.int16)
    else:
        slots_per_day = coverage_matrix.shape[1]

    total_agents = 0
    ft_agents = 0
    pt_agents = 0
    for shift_name, count in assignments.items():
        total_agents += count
        if shift_name.startswith("FT"):
            ft_agents += count
        else:
            pt_agents += count
        if compute_coverage:
            weekly_pattern = shifts_coverage[shift_name]
            pattern_matrix = np.unpackbits(
                weekly_pattern.reshape(7, -1), axis=1
            )[:, :slots_per_day]
            coverage_matrix += pattern_matrix * count

    demand_arr = np.array(demand_matrix)
    total_demand = demand_arr.sum()
    total_covered = np.minimum(coverage_matrix, demand_arr).sum()
    coverage_percentage = (total_covered / total_demand) * 100 if total_demand > 0 else 0.0
    diff_matrix = coverage_matrix - demand_arr
    overstaffing = int(np.sum(diff_matrix[diff_matrix > 0]))
    understaffing = int(np.sum(np.abs(diff_matrix[diff_matrix < 0])))
    return {
        "total_coverage": coverage_matrix.tolist(),
        "total_agents": int(total_agents),
        "ft_agents": int(ft_agents),
        "pt_agents": int(pt_agents),
        "coverage_percentage": float(coverage_percentage),
        "overstaffing": overstaffing,
        "understaffing": understaffing,
        "diff_matrix": diff_matrix.tolist(),
    }


def plot_agent_distribution(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Return a bar chart showing distribution of full/part time agents."""
    ft = metrics.get("ft_agents", 0)
    pt = metrics.get("pt_agents", 0)
    fig = go.Figure(data=[go.Bar(x=["FT", "PT"], y=[ft, pt])])
    fig.update_layout(title="Distribución de Agentes", xaxis_title="Tipo", yaxis_title="Cantidad")
    return fig.to_dict()
