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

from ..other import timeseries_core, erlang_core, modelo_predictivo_core

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


@apps_bp.route("/kpis", methods=["GET", "POST"])
def kpis():
    """Simple placeholder view for KPI experiments."""
    message = None
    if request.method == "POST":
        message = "placeholder"
    return render_template("apps/kpis.html", message=message)
