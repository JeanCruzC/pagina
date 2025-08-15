"""Blueprint with lightweight demonstration apps.

This module defines small isolated routes used mostly for demos or simple
interactive utilities.  Each view delegates heavy computations to functions in
``website.other`` so that the routes themselves remain thin controllers.
"""

from flask import Blueprint, redirect, render_template, request, url_for
import json
import plotly.graph_objects as go

from .core import login_required
from ..other import erlang_core, timeseries_core

bp = Blueprint("apps", __name__, url_prefix="/apps")


@bp.route("/")
@login_required
def index():
    """Redirect to the default app or show a menu."""
    return redirect(url_for("apps.erlang"))


@bp.route("/erlang", methods=["GET", "POST"])
@login_required
def erlang():
    """Render the Erlang app and process demand matrices.

    The view accepts a POST parameter called ``matrix`` which should contain
    a JSON encoded 7x24 demand matrix.  The heavy lifting is delegated to
    :mod:`website.other.erlang_core` which returns basic metrics and optional
    heatmaps.  All returned objects are JSON serialisable so that the frontend
    can consume them easily.
    """

    metrics = {}
    heatmaps = {}

    if request.method == "POST":
        matrix = []
        matrix_json = request.form.get("matrix")
        if matrix_json:
            try:
                matrix = json.loads(matrix_json)
            except Exception:
                matrix = []

        if matrix:
            metrics = erlang_core.analyze_demand_matrix(matrix)
            heatmaps = erlang_core.generate_all_heatmaps(matrix)

    return render_template("apps/erlang.html", metrics=metrics, heatmaps=heatmaps)


@bp.route("/timeseries", methods=["GET", "POST"])
@login_required
def timeseries():
    """Render the time series exploration interface.

    For ``POST`` requests the form parameters are collected into a dictionary
    and handed over to :mod:`website.other.timeseries_core` for processing.  Any
    :class:`plotly.graph_objects.Figure` returned by the core module is
    serialised using ``fig.to_json()`` so that the frontend can mount it with
    ``Plotly.react``.
    """

    metrics = {}
    table = []
    figure_json = None

    if request.method == "POST":
        params = {}
        for key, value in request.form.items():
            if key == "csrf_token":
                continue
            try:
                params[key] = json.loads(value)
            except Exception:
                params[key] = value

        core_fn = getattr(timeseries_core, "run", None)
        if core_fn is None:
            core_fn = getattr(timeseries_core, "process", None)
        if core_fn is None:
            core_fn = getattr(timeseries_core, "analyze", None)
        result = core_fn(params) if callable(core_fn) else {}

        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        table = result.get("table", []) if isinstance(result, dict) else []
        fig = result.get("figure") if isinstance(result, dict) else None
        if isinstance(fig, go.Figure):
            figure_json = fig.to_json()
        elif fig is not None:
            figure_json = json.dumps(fig)

    return render_template(
        "apps/timeseries.html",
        metrics=metrics,
        table=table,
        figure_json=figure_json,
    )


@bp.route("/kpis", methods=["GET", "POST"])
@login_required
def kpis():
    """Placeholder route for KPIs demo app."""
    return render_template("apps/kpis.html")
