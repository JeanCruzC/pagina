from flask import Blueprint, render_template, request, current_app, jsonify
from flask_wtf.csrf import generate_csrf
from . import scheduler

bp = Blueprint("generator", __name__)

def _cfg_from_request(form):
    return {
        "profile": form.get("profile", "JEAN"),
        "use_ft": form.get("use_ft", "true") == "true",
        "use_pt": form.get("use_pt", "true") == "true",
        "allow_8h": True,
        "allow_10h8": True,
        "allow_pt_4h": True,
        "allow_pt_5h": True,
        "allow_pt_6h": True,
        "break_from_start": float(form.get("break_from_start", 2)),
        "break_from_end": float(form.get("break_from_end", 2)),
        "solver_time": int(form.get("solver_time", current_app.config.get("TIME_SOLVER", 120))),
        "solver_msg": True,
        "coverage": float(form.get("coverage", 98)),
        "agent_limit_factor": int(form.get("agent_limit_factor", 30)),
        "excess_penalty": float(form.get("excess_penalty", 5)),
        "peak_bonus": float(form.get("peak_bonus", 2)),
        "critical_bonus": float(form.get("critical_bonus", 2.5)),
        "iterations": int(form.get("iterations", 6)),
        "use_pulp": True,
        "use_greedy": True,
        "export_files": False,
        "ft_first_pt_last": True,
        "optimization_profile": form.get("optimization_profile", "JEAN"),
    }

@bp.route("/generador", methods=["GET", "POST"])
def generador():
    if request.method == "GET":
        # Muestra formulario simple de carga (sin JS, sin polling)
        # Genera token CSRF y pásalo al template
        return render_template(
            "generador.html", mode="sync", csrf_token=generate_csrf()
        )

    # POST: ejecutar TODO en la MISMA request (modo Streamlit)
    xls = request.files.get("file") or request.files.get("excel")
    if not xls:
        return jsonify({"error": "Falta el archivo Excel"}), 400

    cfg = _cfg_from_request(request.form)
    payload = scheduler.run_complete_optimization(
        xls,
        config=cfg,
        generate_charts=True,
        job_id=None,
        return_payload=True  # <- clave: devuelve figuras/base64 y métricas
    )
    return render_template("resultados.html", payload=payload, mode="sync")
