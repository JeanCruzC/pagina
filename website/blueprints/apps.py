"""Blueprint with lightweight demonstration apps.

This module defines small isolated routes used mostly for demos or simple
interactive utilities.  Each view delegates heavy computations to functions in
``website.other`` so that the routes themselves remain thin controllers.
"""

from flask import Blueprint, redirect, render_template, request, url_for, session
from typing import Any, Dict
import json
import plotly.graph_objects as go

from ..other import timeseries_core, erlang_core, modelo_predictivo_core
from ..logic import erlang as erlang_logic

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


@bp.route("/erlang_cx", methods=["GET", "POST"])
def erlang_cx():
    """Advanced Erlang calculator with optional abandonment."""

    metrics: Dict[str, Any] = {}
    table: Dict[str, Any] = {}
    chart_json = None
    target_sl = 0.80

    if request.method == "POST":
        forecast = request.form.get("forecast", type=float, default=0.0) or 0.0
        interval = request.form.get("interval", type=float, default=3600.0) or 3600.0
        aht = request.form.get("aht", type=float, default=0.0) or 0.0
        agents = request.form.get("agents", type=int, default=0) or 0
        awt = request.form.get("awt", type=float, default=0.0) or 0.0
        target_sl = request.form.get("target_sl", type=float, default=0.80) or 0.80
        use_advanced = request.form.get("use_advanced") == "on"
        lines = request.form.get("lines", type=int, default=0) or 0
        patience = request.form.get("patience", type=float, default=0.0) or 0.0

        arrival_rate = forecast / interval if interval else 0.0

        sl = erlang_logic.service_level_erlang_c(arrival_rate, aht, agents, awt)
        asa = erlang_logic.waiting_time_erlang_c(arrival_rate, aht, agents)
        occupancy = erlang_logic.occupancy_erlang_c(arrival_rate, aht, agents)

        metrics = {
            "service_level": sl,
            "asa": asa,
            "occupancy": occupancy,
        }

        if use_advanced:
            abandonment = erlang_logic.erlang_x_abandonment(
                arrival_rate, aht, agents, lines or agents, patience or 1
            )
            metrics["abandonment"] = abandonment

        recommended = erlang_core.required_agents_for_service_level(
            arrival_rate, aht, awt, target_sl, max_agents=max(agents * 2, agents + 1)
        )
        table = {
            "recommended": recommended,
            "current": agents,
            "difference": recommended - agents,
            "calls_per_agent": forecast / agents if agents else 0.0,
        }

        max_agents_plot = max(recommended, agents) + 10
        agent_range = list(range(1, max_agents_plot + 1))
        sl_values = [
            erlang_logic.service_level_erlang_c(arrival_rate, aht, a, awt)
            for a in agent_range
        ]
        asa_values = [
            erlang_logic.waiting_time_erlang_c(arrival_rate, aht, a)
            for a in agent_range
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=agent_range, y=sl_values, name="SL"))
        fig.add_trace(
            go.Scatter(x=agent_range, y=asa_values, name="ASA", yaxis="y2")
        )
        fig.update_layout(
            xaxis_title="Agentes",
            yaxis=dict(title="SL", range=[0, 1]),
            yaxis2=dict(title="ASA (s)", overlaying="y", side="right"),
            shapes=[
                dict(
                    type="line",
                    x0=agents,
                    x1=agents,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="red", dash="dot"),
                ),
                dict(
                    type="line",
                    x0=recommended,
                    x1=recommended,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="green", dash="dot"),
                ),
            ],
        )
        chart_json = fig.to_json()

    return render_template(
        "apps/erlang_cx.html",
        metrics=metrics,
        table=table,
        chart_json=chart_json,
        target_sl=target_sl,
    )


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


@bp.route("/series", methods=["GET", "POST"])
def series():
    """Alias route that reuses :func:`timeseries`."""
    return timeseries()


@bp.route("/erlang/<path:submodule>", methods=["GET", "POST"])
def erlang_submodule(submodule: str):
    """Dispatch any Erlang submodule to the main view."""
    return erlang()


@bp.route("/kpis", methods=["GET", "POST"])
def kpis():
    """Simple placeholder view for KPI experiments."""
    message = None
    if request.method == "POST":
        message = "placeholder"
    return render_template("apps/kpis.html", message=message)
