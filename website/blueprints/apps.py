"""Blueprint with lightweight demonstration apps.

This module defines small isolated routes used mostly for demos or simple
interactive utilities.  Each view delegates heavy computations to functions in
``website.other`` so that the routes themselves remain thin controllers.
"""

from flask import Blueprint, redirect, render_template, request, url_for, session
from typing import Any, Dict
import json
import plotly.graph_objects as go

from ..other import timeseries_core, erlang_core

bp = Blueprint("apps", __name__, url_prefix="/apps")


@bp.before_request
def require_login():  # pragma: no cover - simple auth gate
    """Redirect users to the login page when not authenticated."""
    if "user" not in session:
        return redirect(url_for("core.login"))


@bp.route("/")
def index():
    """Redirect to the default app or show a menu."""
    return redirect(url_for("apps.erlang"))


@bp.route("/erlang", methods=["GET", "POST"])
def erlang():
    """Render a minimal Erlang calculator."""

    metrics = {}

    if request.method == "POST":
        calls = request.form.get("calls", type=float, default=0) or 0
        agents = request.form.get("agents", type=float, default=0) or 0

        # Create a simple demand matrix placing the calls in the first slot.
        demand = [[calls] + [0.0] * 23] + [[0.0] * 24 for _ in range(6)]

        result = erlang_core.analyze_demand_matrix(demand)
        if isinstance(result, dict):
            metrics = result
            metrics["agents"] = agents

    return render_template("apps/erlang.html", metrics=metrics)


@bp.route("/predictivo")
def predictivo():
    """Placeholder for the Predictive app."""
    return "Predictive app coming soon"


@bp.route("/timeseries", methods=["GET", "POST"])
def timeseries():
    """Render the time series exploration interface.

    For ``POST`` requests the form parameters are collected into a dictionary
    and handed over to :mod:`website.other.timeseries_core` for processing.  Any
    :class:`plotly.graph_objects.Figure` returned by the core module is
    serialised using ``fig.to_json()`` so that the frontend can mount it with
    ``Plotly.react``.
    """

    metrics: Dict[str, Any] = {}
    recommendation = ""
    weekly_table = []
    heatmap_json = None
    interactive_json = None

    if request.method == "POST":
        file_storage = request.files.get("file")
        params = {}
        for key, value in request.form.items():
            if key == "csrf_token":
                continue
            params[key] = value

        result = timeseries_core.run(params, file_storage=file_storage)

        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        recommendation = result.get("recommendation", "") if isinstance(result, dict) else ""
        weekly_table = result.get("weekly_table", []) if isinstance(result, dict) else []

        heatmap = result.get("heatmap") if isinstance(result, dict) else None
        interactive = result.get("interactive") if isinstance(result, dict) else None
        if isinstance(heatmap, go.Figure):
            heatmap_json = heatmap.to_json()
        elif heatmap is not None:
            heatmap_json = json.dumps(heatmap)
        if isinstance(interactive, go.Figure):
            interactive_json = interactive.to_json()
        elif interactive is not None:
            interactive_json = json.dumps(interactive)

    return render_template(
        "apps/timeseries.html",
        metrics=metrics,
        recommendation=recommendation,
        weekly_table=weekly_table,
        heatmap_json=heatmap_json,
        interactive_json=interactive_json,
    )
