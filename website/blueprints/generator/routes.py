from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
    make_response,
    current_app,
)
from flask import after_this_request
import io
import json
import os
import tempfile

from ... import scheduler
from ...app import login_required
from ...utils.generator import form_on, serve_excel


generator_bp = Blueprint("generator", __name__)


@generator_bp.route("/generador", methods=["GET", "POST"])
@login_required
def generador():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response
    current_app.logger.debug("Request method: %s", request.method)
    current_app.logger.debug("Content-Type: %s", request.content_type)
    current_app.logger.debug("Files: %s", list(request.files.keys()))
    current_app.logger.debug("Form: %s", list(request.form.keys()))
    current_app.logger.debug("User session: %s", session.get("user", "NO_USER"))

    if request.method == "POST":
        try:
            current_app.logger.debug("Entering POST logic")
            current_app.logger.debug("Starting POST processing")

            excel = request.files.get("excel")
            if not excel:
                current_app.logger.error("No file received")
                return {"error": "No file provided"}, 400

            current_app.logger.debug("Received file: %s", excel.filename)

            current_app.logger.debug("Building configuration...")

            cfg = {
                "TIME_SOLVER": request.form.get("solver_time", type=int),
                "TARGET_COVERAGE": request.form.get("coverage", type=float),
                "use_ft": form_on("use_ft"),
                "use_pt": form_on("use_pt"),
                "allow_8h": form_on("allow_8h"),
                "allow_10h8": form_on("allow_10h8"),
                "allow_pt_4h": form_on("allow_pt_4h"),
                "allow_pt_6h": form_on("allow_pt_6h"),
                "allow_pt_5h": form_on("allow_pt_5h"),
                "break_from_start": request.form.get("break_from_start", type=float),
                "break_from_end": request.form.get("break_from_end", type=float),
                "optimization_profile": request.form.get("profile"),
                "agent_limit_factor": request.form.get("agent_limit_factor", type=int),
                "excess_penalty": request.form.get("excess_penalty", type=float),
                "peak_bonus": request.form.get("peak_bonus", type=float),
                "critical_bonus": request.form.get("critical_bonus", type=float),
                "iterations": request.form.get("iterations", type=int),
            }

            current_app.logger.debug("Configuration created: %s", cfg)
            current_app.logger.debug("Calling scheduler.run_complete_optimization...")

            jean_template = request.files.get("jean_file")
            if jean_template and jean_template.filename:
                try:
                    with io.TextIOWrapper(jean_template, encoding="utf-8") as fh:
                        cfg.update(json.load(fh))
                except Exception:
                    flash("Plantilla JEAN inválida")

            try:
                result = scheduler.run_complete_optimization(excel, config=cfg)
                current_app.logger.info("Scheduler completed successfully")
                current_app.logger.debug("Result type: %s", type(result))
                current_app.logger.debug(
                    "Result keys: %s",
                    list(result.keys()) if isinstance(result, dict) else "No es dict",
                )
                current_app.logger.debug("Verifying libraries...")
                try:
                    import pulp

                    current_app.logger.debug("PuLP available: %s", pulp.__version__)
                except Exception:
                    current_app.logger.warning("PuLP not available")
                try:
                    import numpy as np

                    current_app.logger.debug("NumPy: %s", np.__version__)
                except Exception:
                    current_app.logger.warning("NumPy not available")
            except Exception as e:
                current_app.logger.exception("Scheduler exception: %s", e)
                return {"error": f"Error en optimización: {str(e)}"}, 500

            current_app.logger.debug("Adding download_url...")
            result["download_url"] = (
                url_for("generator.download_excel")
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

            return redirect(url_for("generator.resultados"))

        except Exception as e:
            current_app.logger.exception("Exception in POST: %s", e)
            code = 400 if isinstance(e, ValueError) else 500
            return {"error": f"Server error: {str(e)}"}, code

    return render_template("generador.html")


@generator_bp.route("/download_excel")
@login_required
def download_excel():
    file_path = session.get("last_excel_file")
    return serve_excel(file_path)


@generator_bp.route("/resultados")
@login_required
def resultados():
    result_file = session.get("last_result_file")
    cfg = session.get("effective_config")
    excel_file = session.get("last_excel_file")
    if not result_file or not os.path.exists(result_file) or cfg is None:
        flash("No hay resultados disponibles. Genera un nuevo horario.")
        return redirect(url_for("generator.generador"))
    with open(result_file) as f:
        result = json.load(f)
    result["download_url"] = (
        url_for("generator.download_excel") if excel_file else None
    )
    result["effective_config"] = cfg
    return render_template("resultados.html", resultado=result)
