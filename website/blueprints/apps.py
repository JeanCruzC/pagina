"""Blueprint with lightweight demonstration apps.

This module defines small isolated routes used mostly for demos or simple
interactive utilities.  Each view delegates heavy computations to functions in
``website.other`` so that the routes themselves remain thin controllers.
"""

from flask import Blueprint, redirect, render_template, request, url_for, session
import json
import plotly.graph_objects as go

from ..other import timeseries_core, erlang_core, modelo_predictivo_core

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
        calls = request.form.get("calls", type=float, default=0) or 0.0
        aht = request.form.get("aht", type=float, default=0) or 0.0
        sl = request.form.get("sl", type=float, default=0) or 0.0
        awl = request.form.get("awl", type=float, default=0) or 0.0
        agents = request.form.get("agents", type=int, default=0) or 0
        max_agents = request.form.get("max_agents", type=int, default=agents) or agents
        calc_type = request.form.get("calc_type", default="service")

        sl_target = sl / 100 if sl > 1 else sl

        metrics = erlang_core.calculate_erlang_metrics(
            calls=calls,
            aht=aht,
            sl_target=sl_target,
            awt=awl,
            agents=agents,
            max_agents=max_agents,
            calc_type=calc_type,
        )

    return render_template("apps/erlang.html", metrics=metrics)


@bp.route("/predictivo", methods=["GET", "POST"])
def predictivo():
    """Execute the predictive model workflow."""

    metrics = {}
    table = []
    download = None

    if request.method == "POST":
        file = request.files.get("file")
        steps = request.form.get("steps_horizon", type=int, default=1)
        if file:
            result = modelo_predictivo_core.run(file, steps)
            metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
            table = result.get("table", []) if isinstance(result, dict) else []
            file_bytes = result.get("file_bytes") if isinstance(result, dict) else None
            if file_bytes:
                import base64

                download = base64.b64encode(file_bytes).decode("utf-8")

    return render_template(
        "apps/predictivo.html", metrics=metrics, table=table, download=download
    )


@bp.route("/timeseries", methods=["GET", "POST"])
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
