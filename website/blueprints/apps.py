"""Blueprint with lightweight demonstration apps.

This module defines small isolated routes used mostly for demos or simple
interactive utilities.  Each view delegates heavy computations to functions in
``website.other`` so that the routes themselves remain thin controllers.
"""

import json
import os
import tempfile
import uuid
from typing import Any, Dict

import pandas as pd

import plotly.graph_objects as go
from flask import (
    Blueprint,
    redirect,
    render_template,
    request,
    url_for,
    session,
    send_file,
    abort,
    after_this_request,
)

from ..other import (
    timeseries_core,
    erlang_core,
    modelo_predictivo_core,
    erlang_visual,
    comparativo_core,
    staffing_core,
    batch_core,
)
from ..services import erlang as erlang_service, erlang_o

# Public blueprint used by the main application factory
apps_bp = Blueprint("apps", __name__, url_prefix="/apps")

# Default temporary directory for generated files
temp_dir = tempfile.gettempdir()


@apps_bp.before_request
def require_login():  # pragma: no cover - simple auth gate
    """Redirect users to the login page when not authenticated."""
    if "user" not in session:
        return redirect(url_for("core.login"))


@apps_bp.route("/")
def index():
    """Redirect to the default app or show a menu."""
    return redirect(url_for("apps.erlang"))


@apps_bp.route("/erlang", methods=["GET", "POST"])
def erlang():
    """Render a minimal Erlang calculator."""

    metrics = {}
    figure_json = None
    target_sl = 0.8
    agents = None

    if request.method == "POST":
        calls = request.form.get("calls", type=float, default=0) or 0.0
        aht = request.form.get("aht", type=float, default=0) or 0.0
        awl = request.form.get("awl", type=float, default=0) or 0.0
        agents = request.form.get("agents", type=int, default=0) or 0
        interval = request.form.get("interval", type=int, default=3600) or 3600
        lines = request.form.get("lines", type=int)
        patience = request.form.get("patience", type=float)
        target_sl = request.form.get("target_sl", type=float, default=0.8) or 0.8

        result = erlang_core.calculate_erlang_metrics(
            calls=calls,
            aht=aht,
            awt=awl,
            agents=agents,
            sl_target=target_sl,
            lines=lines,
            patience=patience,
            interval_seconds=interval,
        )

        if isinstance(result, dict):
            fig = result.pop("figure", None)
            metrics = result
            if isinstance(fig, go.Figure):
                figure_json = fig.to_json()
            elif fig is not None:
                figure_json = json.dumps(fig)

    return render_template(
        "apps/erlang.html",
        metrics=metrics,
        figure_json=figure_json,
        target_sl=target_sl,
        agents=agents,
    )


@apps_bp.route("/predictivo", methods=["GET", "POST"])
def predictivo():
    """Execute the predictive model workflow."""

    metrics = {}
    table = []
    download_url = None

    if request.method == "POST":
        file = request.files.get("file")
        steps = request.form.get("steps_horizon", type=int, default=1)
        if file:
            result = modelo_predictivo_core.run(file, steps)
            metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
            table = result.get("table", []) if isinstance(result, dict) else []
            file_bytes = result.get("file_bytes") if isinstance(result, dict) else None
            if file_bytes:
                job_id = uuid.uuid4().hex
                path = os.path.join(temp_dir, f"{job_id}.csv")
                with open(path, "wb") as f:
                    f.write(file_bytes)
                download_url = url_for("apps.predictivo_download", job_id=job_id)

    return render_template(
        "apps/predictivo.html",
        metrics=metrics,
        table=table,
        download_url=download_url,
    )


@apps_bp.route("/predictivo/download/<job_id>")
def predictivo_download(job_id: str):
    """Serve the generated forecast file and remove it afterwards."""

    path = os.path.join(temp_dir, f"{job_id}.csv")
    if not os.path.exists(path):
        abort(404)

    @after_this_request
    def cleanup(response):
        try:
            os.remove(path)
        except OSError:
            pass
        return response

    return send_file(
        path,
        as_attachment=True,
        download_name="resultados.csv",
        mimetype="text/csv",
    )


@apps_bp.route("/timeseries", methods=["GET", "POST"])
def timeseries():
    """Render the time series exploration interface.

    For ``POST`` requests the form parameters are collected into a dictionary
    and handed over to :mod:`website.other.timeseries_core` for processing.  Any
    :class:`plotly.graph_objects.Figure` returned by the core module is
    serialised using ``fig.to_json()`` so that the frontend can mount it with
    ``Plotly.react``.
    """

    metrics: Dict[str, Any] = {}
    figure_json = None
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


@apps_bp.route("/series", methods=["GET", "POST"])
def series():
    """Alias route that reuses :func:`timeseries`."""
    return timeseries()


@apps_bp.route("/erlang/<path:submodule>", methods=["GET", "POST"])
def erlang_submodule(submodule: str):
    """Dispatch any Erlang submodule to the main view."""
    return erlang()


@apps_bp.route("/erlang_visual", methods=["GET", "POST"])
def erlang_visual_view():
    """Enhanced Erlang view with agent visualisation."""

    metrics: Dict[str, Any] = {}
    matrix = None
    queue = None
    asa_bar = None

    if request.method == "POST":
        forecast = request.form.get("forecast", type=float, default=0.0) or 0.0
        aht = request.form.get("aht", type=float, default=0.0) or 0.0
        agents = request.form.get("agents", type=int, default=0) or 0
        awt = request.form.get("awt", type=float, default=0.0) or 0.0
        interval = request.form.get("interval", default="3600")
        interval_seconds = 1800 if interval == "1800" else 3600
        sl_target = request.form.get("sl_target", type=float, default=0.8)

        action = request.form.get("action")
        if action == "increase_demand":
            forecast *= 1.1
        elif action == "add_agents":
            agents += 5
        elif action == "reduce_aht":
            aht *= 0.9

        arrival_rate = forecast / interval_seconds
        sl = erlang_service.sla_x(arrival_rate, aht, agents, awt)
        asa = erlang_service.waiting_time_erlang_c(arrival_rate, aht, agents)
        occ = erlang_core.occupancy_erlang_c(arrival_rate, aht, agents)
        required = erlang_service.agents_for_sla(sl_target, arrival_rate, aht, awt)
        hourly_forecast = forecast * 3600 / interval_seconds if interval_seconds else 0
        cpa = hourly_forecast / agents if agents else 0

        matrix_data = erlang_visual.generate_agent_matrix(
            forecast, aht, agents, awt, interval_seconds, int(required)
        )
        matrix = matrix_data["rows"]
        queue = erlang_visual.generate_queue(matrix_data["sl"], forecast)
        asa_bar = erlang_visual.generate_asa_bar(matrix_data["asa"], awt)

        metrics = {
            "service_level": f"{sl:.1%}",
            "asa": f"{asa:.1f}",
            "occupancy": f"{occ:.1%}",
            "required_agents": int(required),
            "calls_per_agent": f"{cpa:.1f}",
        }

        if request.headers.get("HX-Request"):
            return render_template(
                "partials/erlang_visual_results.html",
                metrics=metrics,
                matrix=matrix,
                queue=queue,
                asa_bar=asa_bar,
            )

    return render_template(
        "apps/erlang_visual.html",
        metrics=metrics,
        matrix=matrix,
        queue=queue,
        asa_bar=asa_bar,
    )


@apps_bp.route("/chat", methods=["GET", "POST"])
def chat():
    """Chat multi-channel calculator."""

    metrics: Dict[str, Any] = {}
    figure_json = None

    if request.method == "POST":
        forecast = request.form.get("forecast", type=float, default=0.0) or 0.0
        agents = request.form.get("agents", type=int, default=0) or 0
        awt = request.form.get("awt", type=float, default=0.0) or 0.0
        interval = request.form.get("interval", default="3600")
        interval_seconds = 1800 if interval == "1800" else 3600
        sl_target = request.form.get("sl_target", type=float, default=0.8)
        lines = request.form.get("lines", type=int)
        patience = request.form.get("patience", type=float)

        aht_list = [
            float(x) for x in request.form.getlist("ahts") if x.strip()
        ] or [0.0]

        arrival_rate = forecast / interval_seconds
        sl = erlang_service.chat_sla(arrival_rate, aht_list, agents, awt, lines, patience)
        asa = erlang_service.chat_asa(arrival_rate, aht_list, agents, lines, patience)
        required = erlang_service.chat_agents_for_sla(
            sl_target, arrival_rate, aht_list, awt, lines, patience
        )

        metrics: Dict[str, Any] = {
            "service_level": f"{sl:.1%}",
            "asa": f"{asa:.1f}",
            "required_agents": int(required),
        }

        if patience is not None:
            parallel_capacity = len(aht_list)
            avg_aht = sum(aht_list) / parallel_capacity
            effectiveness = 0.7 + (0.3 / parallel_capacity)
            effective_agents = agents * parallel_capacity * effectiveness
            abandon = erlang_service.erlang_x_abandonment(
                arrival_rate, avg_aht, effective_agents, lines or 999, patience
            )
            metrics["abandonment"] = f"{abandon:.1%}"

        if request.headers.get("HX-Request"):
            return render_template("partials/chat_results.html", metrics=metrics)

    return render_template("apps/chat.html", metrics=metrics)


@apps_bp.route("/blending", methods=["GET", "POST"])
def blending():
    """Blending calculator."""

    metrics: Dict[str, Any] = {}
    figure_json = None

    if request.method == "POST":
        forecast = request.form.get("forecast", type=float, default=0.0) or 0.0
        aht = request.form.get("aht", type=float, default=0.0) or 0.0
        agents = request.form.get("agents", type=int, default=0) or 0
        awt = request.form.get("awt", type=float, default=0.0) or 0.0
        threshold = request.form.get("threshold", type=float, default=0.0) or 0.0
        interval = request.form.get("interval", default="3600")
        interval_seconds = 1800 if interval == "1800" else 3600
        sl_target = request.form.get("sl_target", type=float, default=0.8)
        lines = request.form.get("lines", type=int)
        patience = request.form.get("patience", type=float)

        arrival_rate = forecast / interval_seconds
        sl = erlang_service.bl_sla(
            arrival_rate, aht, agents, awt, lines, patience, threshold
        )
        outbound = (
            erlang_service.bl_outbound_capacity(
                arrival_rate, aht, agents, lines, patience, threshold, aht
            )
            * 3600
        )
        optimal = erlang_service.bl_optimal_threshold(
            arrival_rate, aht, agents, awt, lines, patience, sl_target
        )
        metrics = {
            "service_level": f"{sl:.1%}",
            "outbound_capacity": f"{outbound:.1f} llamadas/h",
            "optimal_threshold": f"{optimal:.1f}",
        }

        threshold_range = range(0, int(agents * 0.4))
        sl_data = []
        outbound_data = []
        for t in threshold_range:
            sl_val = erlang_service.bl_sla(
                arrival_rate, aht, agents, awt, lines, patience, t
            )
            out_val = (
                erlang_service.bl_outbound_capacity(
                    arrival_rate, aht, agents, lines, patience, t, aht
                )
                * 3600
            )
            sl_data.append(sl_val)
            outbound_data.append(out_val)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(threshold_range),
                y=sl_data,
                mode="lines+markers",
                name="Service Level Inbound",
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(threshold_range),
                y=outbound_data,
                mode="lines+markers",
                name="Capacidad Outbound",
                yaxis="y2",
            )
        )
        fig.update_layout(
            title="Service Level vs Capacidad Outbound por Threshold",
            xaxis_title="Threshold (Agentes Reservados)",
            yaxis=dict(title="Service Level Inbound", side="left", range=[0, 1]),
            yaxis2=dict(
                title="Capacidad Outbound (llamadas/hora)",
                side="right",
                overlaying="y",
            ),
            hovermode="x unified",
        )
        fig.add_vline(
            x=threshold, line_dash="dash", line_color="red", annotation_text="Actual"
        )
        fig.add_vline(
            x=optimal, line_dash="dash", line_color="orange", annotation_text="Óptimo"
        )
        figure_json = fig.to_json()

        if request.headers.get("HX-Request"):
            return render_template(
                "partials/blending_results.html", metrics=metrics, figure_json=figure_json
            )

    return render_template(
        "apps/blending.html", metrics=metrics, figure_json=figure_json
    )


@apps_bp.route("/erlang_o", methods=["GET", "POST"])
def erlang_o_view():
    """Erlang O outbound calculator."""

    metrics: Dict[str, Any] = {}

    if request.method == "POST":
        agents = request.form.get("agents", type=int, default=0) or 0
        hours_per_day = request.form.get("hours_per_day", type=float, default=0.0) or 0.0
        calls_per_hour = request.form.get("calls_per_hour", type=float, default=0.0) or 0.0
        success_rate = request.form.get("success_rate", type=float, default=0.3) or 0.3
        metrics = erlang_o.productivity(agents, hours_per_day, calls_per_hour, success_rate)

        if request.headers.get("HX-Request"):
            return render_template("partials/erlang_o_results.html", metrics=metrics)

    return render_template("apps/erlang_o.html", metrics=metrics)


@apps_bp.route("/comparativo", methods=["GET", "POST"])
def comparativo():
    """Run comparative analysis across models."""

    table = []
    figure_json = None

    if request.method == "POST":
        forecast = request.form.get("forecast", type=float, default=0.0) or 0.0
        aht = request.form.get("aht", type=float, default=0.0) or 0.0
        agents = request.form.get("agents", type=int, default=0) or 0
        awt = request.form.get("awt", type=float, default=0.0) or 0.0
        lines = request.form.get("lines", type=int, default=agents) or agents
        patience = request.form.get("patience", type=float, default=180.0) or 180.0
        interval = request.form.get("interval", default="3600")
        interval_seconds = 1800 if interval == "1800" else 3600

        result = comparativo_core.comparative_analysis(
            forecast, interval_seconds, aht, agents, awt, lines, patience
        )
        table = result.get("table", [])
        fig = result.get("figure")
        if isinstance(fig, dict):
            figure_json = json.dumps(fig)
        elif isinstance(fig, go.Figure):
            figure_json = fig.to_json()

        if request.headers.get("HX-Request"):
            return render_template(
                "partials/comparativo_results.html", table=table, figure_json=figure_json
            )

    return render_template(
        "apps/comparativo.html", table=table, figure_json=figure_json
    )


@apps_bp.route("/staffing", methods=["GET", "POST"])
def staffing():
    """Staffing optimisation utility."""

    table = []
    figure_json = None
    summary = {}

    if request.method == "POST":
        forecasts_raw = request.form.get("forecasts", "")
        try:
            forecasts = [float(x) for x in forecasts_raw.split(",") if x.strip()]
        except ValueError:
            forecasts = []
        aht = request.form.get("aht", type=float, default=0.0) or 0.0
        interval = request.form.get("interval", default="3600")
        interval_seconds = 1800 if interval == "1800" else 3600
        sl_target = request.form.get("sl_target", type=float, default=0.8)

        result = staffing_core.staffing_optimizer(
            forecasts, aht, interval_seconds, sl_target
        )
        table = result.get("table", [])
        summary = result.get("summary", {})
        fig = result.get("figure")
        if isinstance(fig, dict):
            figure_json = json.dumps(fig)
        elif isinstance(fig, go.Figure):
            figure_json = fig.to_json()

        if request.headers.get("HX-Request"):
            return render_template(
                "partials/staffing_results.html",
                table=table,
                summary=summary,
                figure_json=figure_json,
            )

    return render_template(
        "apps/staffing.html", table=table, summary=summary, figure_json=figure_json
    )


@apps_bp.route("/batch", methods=["GET", "POST"])
def batch():
    """Batch processing of contact centre scenarios."""

    table = []
    download_url = None

    if request.method == "POST":
        file = request.files.get("file")
        sl_target = request.form.get("sl_target", type=float, default=0.8)
        awt = request.form.get("awt", type=float, default=20.0)
        interval_choice = request.form.get("interval", default="3600")
        interval_seconds = 1800 if interval_choice == "1800" else 3600
        default_channel = request.form.get("default_channel", "Llamadas")
        if file:
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            processed_rows = []
            for _, row in df.iterrows():
                metrics = batch_core.process_batch_row(
                    row, sl_target, awt, interval_seconds, default_channel
                )
                processed_rows.append({**row, **metrics})
            result_df = pd.DataFrame(processed_rows)
            table = result_df.to_dict("records")
            csv_bytes = batch_core.export_results(result_df).to_csv(index=False).encode("utf-8")
            job_id = uuid.uuid4().hex
            path = os.path.join(temp_dir, f"{job_id}.csv")
            with open(path, "wb") as f:
                f.write(csv_bytes)
            download_url = url_for("apps.batch_download", job_id=job_id)

        if request.headers.get("HX-Request"):
            return render_template(
                "partials/batch_results.html", table=table, download_url=download_url
            )

    return render_template(
        "apps/batch.html", table=table, download_url=download_url
    )


@apps_bp.route("/batch/download/<job_id>")
def batch_download(job_id: str):
    """Send processed batch results."""
    path = os.path.join(temp_dir, f"{job_id}.csv")
    if not os.path.exists(path):
        abort(404)

    @after_this_request
    def cleanup(response):
        try:
            os.remove(path)
        except OSError:
            pass
        return response

    return send_file(
        path,
        as_attachment=True,
        download_name="batch_result.csv",
        mimetype="text/csv",
    )


@apps_bp.route("/kpis", methods=["GET", "POST"])
def kpis():
    """Simple placeholder view for KPI experiments."""
    message = None
    if request.method == "POST":
        message = "placeholder"
    return render_template("apps/kpis.html", message=message)
