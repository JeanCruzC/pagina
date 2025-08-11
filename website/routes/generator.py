"""Main application routes including the schedule generator."""

from __future__ import annotations

import io
import json
import os
import tempfile
from datetime import datetime

from flask import (
    Blueprint,
    after_this_request,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from .. import scheduler
from ..auth import login_required


generator_bp = Blueprint("routes", __name__)


def _on(name: str) -> bool:
    v = request.form.get(name)
    return v is not None and str(v).lower() in {"on", "1", "true", "yes"}


@generator_bp.route("/", endpoint="landing")
def landing():
    return render_template(
        "landing.html",
        title="Schedules",
        year=datetime.now().year,
    )


@generator_bp.route("/app", methods=["GET"], endpoint="app_entry")
def app_entry():
    user = session.get("user")
    if user:
        from ..payments import has_active_subscription

        if has_active_subscription(user):
            return redirect(url_for("routes.generador"))
        flash("Suscripción activa requerida")
        return redirect(url_for("payments.subscribe"))
    return redirect(url_for("auth.login"))


@generator_bp.route("/generador", methods=["GET", "POST"], endpoint="generador")
@login_required
def generador():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response

    if request.method == "POST":
        try:
            excel = request.files.get("excel")
            if not excel:
                return {"error": "No file provided"}, 400

            cfg = {
                "TIME_SOLVER": request.form.get("solver_time", type=int),
                "TARGET_COVERAGE": request.form.get("coverage", type=float),
                "use_ft": _on("use_ft"),
                "use_pt": _on("use_pt"),
                "allow_8h": _on("allow_8h"),
                "allow_10h8": _on("allow_10h8"),
                "allow_pt_4h": _on("allow_pt_4h"),
                "allow_pt_6h": _on("allow_pt_6h"),
                "allow_pt_5h": _on("allow_pt_5h"),
                "break_from_start": request.form.get("break_from_start", type=float),
                "break_from_end": request.form.get("break_from_end", type=float),
                "optimization_profile": request.form.get("profile"),
                "agent_limit_factor": request.form.get(
                    "agent_limit_factor", type=int
                ),
                "excess_penalty": request.form.get("excess_penalty", type=float),
                "peak_bonus": request.form.get("peak_bonus", type=float),
                "critical_bonus": request.form.get("critical_bonus", type=float),
                "iterations": request.form.get("iterations", type=int),
            }

            jean_template = request.files.get("jean_file")
            if jean_template and jean_template.filename:
                try:
                    with io.TextIOWrapper(jean_template, encoding="utf-8") as fh:
                        cfg.update(json.load(fh))
                except Exception:
                    flash("Plantilla JEAN inválida")

            try:
                result = scheduler.run_complete_optimization(excel, config=cfg)
            except Exception as e:
                return {"error": f"Error en optimización: {str(e)}"}, 500

            result["download_url"] = (
                url_for("routes.download_excel")
                if session.get("last_excel_file")
                else None
            )

            if session.get("last_result_file"):
                try:
                    os.remove(session["last_result_file"])
                except Exception:
                    pass
            tmp = tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json", encoding="utf-8"
            )
            json.dump(result, tmp)
            tmp.flush()
            tmp.close()
            session["last_result_file"] = tmp.name
            session["effective_config"] = cfg

            return redirect(url_for("routes.resultados"))

        except Exception as e:
            return {"error": f"Server error: {str(e)}"}, 500

    return render_template("generador.html")


@generator_bp.route("/download_excel", endpoint="download_excel")
@login_required
def download_excel():
    file_path = session.get("last_excel_file")
    if not file_path or not os.path.exists(file_path):
        flash("No hay archivo para descargar.")
        session.pop("last_excel_file", None)
        return redirect(url_for("routes.generador"))

    @after_this_request
    def cleanup(response):
        try:
            os.remove(file_path)
        except Exception:
            pass
        session.pop("last_excel_file", None)
        return response

    return send_file(file_path, download_name="horario.xlsx", as_attachment=True)


@generator_bp.route("/resultados", endpoint="resultados")
@login_required
def resultados():
    result_file = session.get("last_result_file")
    cfg = session.get("effective_config")
    excel_file = session.get("last_excel_file")
    if not result_file or not os.path.exists(result_file) or cfg is None:
        flash("No hay resultados disponibles. Genera un nuevo horario.")
        return redirect(url_for("routes.generador"))
    with open(result_file) as f:
        result = json.load(f)
    result["download_url"] = (
        url_for("routes.download_excel") if excel_file else None
    )
    result["effective_config"] = cfg
    return render_template("resultados.html", resultado=result)


@generator_bp.route("/configuracion", endpoint="configuracion")
@login_required
def configuracion():
    return render_template("configuracion.html")


@generator_bp.route("/perfil", endpoint="perfil")
@login_required
def perfil():
    return redirect(url_for("routes.configuracion"))


__all__ = ["generator_bp"]

