"""Utilities for handling time series learning history and visualisations."""
from __future__ import annotations

from typing import Dict, Any, Iterable
import json
import hashlib
import time
import os

import numpy as np
import plotly.graph_objects as go


def create_demand_signature(demand_matrix: Iterable[Iterable[float]]) -> str:
    """Return a short hash representing the demand pattern."""
    arr = np.array(list(map(list, demand_matrix)), dtype=float)
    normalized = arr / (arr.max() + 1e-8)
    return hashlib.md5(normalized.tobytes()).hexdigest()[:16]


def load_learning_history(path: str = "learning_history.json") -> Dict[str, Any]:
    """Load adaptive learning history from disk if available."""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}


def save_learning_history(history: Dict[str, Any], path: str = "learning_history.json") -> None:
    """Persist adaptive learning history to disk."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)
    except Exception:
        pass


def get_adaptive_parameters(demand_signature: str, learning_history: Dict[str, Any]) -> Dict[str, Any]:
    """Return learned parameters for ``demand_signature`` or defaults."""
    if demand_signature in learning_history:
        learned = learning_history[demand_signature]
        best = min(learned.get("runs", []), key=lambda x: x.get("score", 0))
        return {
            "agent_limit_factor": best["params"]["agent_limit_factor"],
            "excess_penalty": best["params"]["excess_penalty"],
            "peak_bonus": best["params"]["peak_bonus"],
            "critical_bonus": best["params"]["critical_bonus"],
        }
    return {
        "agent_limit_factor": 22,
        "excess_penalty": 0.5,
        "peak_bonus": 1.5,
        "critical_bonus": 2.0,
    }


def update_learning_history(
    demand_signature: str,
    params: Dict[str, Any],
    results: Dict[str, Any],
    history: Dict[str, Any],
    max_runs: int = 10,
) -> Dict[str, Any]:
    """Update ``history`` with new execution ``results`` for ``demand_signature``."""
    if demand_signature not in history:
        history[demand_signature] = {"runs": []}

    score = results["understaffing"] + results["overstaffing"] * 0.3
    history[demand_signature]["runs"].append(
        {
            "params": params,
            "score": score,
            "total_agents": results.get("total_agents"),
            "coverage": results.get("coverage_percentage"),
            "timestamp": time.time(),
        }
    )
    history[demand_signature]["runs"] = history[demand_signature]["runs"][-max_runs:]
    return history


def plot_learning_history(history: Dict[str, Any], demand_signature: str) -> Dict[str, Any]:
    """Create a Plotly line chart of scores over time for a given signature."""
    runs = history.get(demand_signature, {}).get("runs", [])
    if not runs:
        fig = go.Figure()
    else:
        scores = [r.get("score", 0) for r in runs]
        timestamps = [r.get("timestamp", 0) for r in runs]
        fig = go.Figure(data=[go.Scatter(x=timestamps, y=scores, mode="lines+markers")])
    fig.update_layout(title="Evolución del Score", xaxis_title="Timestamp", yaxis_title="Score")
    return fig.to_dict()
