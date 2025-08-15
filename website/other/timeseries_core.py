"""Utilities for handling time series learning history and visualisations."""
from __future__ import annotations

from typing import Dict, Any, Iterable
import json
import hashlib
import time
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import timeseries_full_core


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


def run(params: Dict[str, Any], file_storage=None) -> Dict[str, Any]:
    """Processing pipeline for the time series demo.

    When ``file_storage`` is provided the file is parsed as CSV or Excel and
    delegated to :func:`timeseries_full_core.process_timeseries`.  Otherwise a
    very small fallback implementation is used that expects a comma separated
    string under the ``values`` key.
    """

    if file_storage is not None:
        filename = (file_storage.filename or "").lower()
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(file_storage)
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_storage)
            else:
                return {}
        except Exception:
            return {}

        return timeseries_full_core.process_timeseries(
            df,
            weight_last=float(params.get("weight_last", 0.7)),
            weight_prev=float(params.get("weight_prev", 0.3)),
            scope=params.get("scope", "Total"),
            view=params.get("view", "Día"),
        )

    # --- Fallback demo behaviour: parse a list of numbers ---
    values = params.get("values", [])
    if isinstance(values, str):
        try:
            values = [float(v) for v in values.split(",") if v.strip()]
        except Exception:
            values = []
    else:
        try:
            values = [float(v) for v in values]
        except Exception:
            values = []

    arr = np.array(values, dtype=float)

    metrics: Dict[str, Any] = {}
    if arr.size:
        metrics = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    table = [{"index": int(i), "value": float(v)} for i, v in enumerate(arr.tolist())]

    fig = go.Figure(data=[go.Scatter(y=arr.tolist(), mode="lines+markers")])
    fig.update_layout(title="Serie de tiempo", xaxis_title="Índice", yaxis_title="Valor")

    return {"metrics": metrics, "table": table, "figure": fig}
