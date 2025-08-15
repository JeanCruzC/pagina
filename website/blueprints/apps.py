"""Blueprint with lightweight demonstration apps.

This module defines small isolated routes used mostly for demos or simple
interactive utilities.  Each view delegates heavy computations to functions in
``website.other`` so that the routes themselves remain thin controllers.
"""

from flask import Blueprint, redirect, render_template, request, url_for
import json
import plotly.graph_objects as go

from .core import login_required
from ..other import predictivo_core, timeseries_core

bp = Blueprint("apps", __name__, url_prefix="/apps")


@bp.route("/")
def index():
    """Redirect to the default app or show a menu."""
    return redirect(url_for("apps.erlang"))


@bp.route("/erlang")
def erlang():
    """Placeholder for the Erlang app."""
    return "Erlang app coming soon"


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


@bp.route("/predictivo", methods=["GET", "POST"])
@login_required
def predictivo():
    """Upload a time series file and display multi-model forecasts."""

    table_html = None
    figure_json = None

    if request.method == "POST":
        file = request.files.get("file")
        steps = int(request.form.get("steps", 6))
        if file:
            result = predictivo_core.forecast_from_file(file, steps)
            table_html = result["forecast"].to_html(
                classes="table table-sm", border=0
            )
            fig = result.get("figure")
            if isinstance(fig, go.Figure):
                figure_json = fig.to_json()
            elif fig is not None:
                figure_json = json.dumps(fig)

    return render_template(
        "apps/predictivo.html",
        table_html=table_html,
        figure_json=figure_json,
    )
