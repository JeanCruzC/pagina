from flask import Blueprint, render_template, request, current_app, jsonify
from .scheduler import run_complete_optimization as run_opt

bp = Blueprint("generator", __name__)


def _b(v):
    return str(v).lower() in ("true", "1", "on", "yes", "si", "sí")


def _cfg_from_request(form):
    return {
        "optimization_profile": form.get("optimization_profile", "JEAN"),
        "use_ft": _b(form.get("use_ft", "on")),
        "use_pt": _b(form.get("use_pt", "on")),
        "break_from_start": float(form.get("break_from_start", 2)),
        "break_from_end": float(form.get("break_from_end", 2)),
        "solver_time": int(form.get("solver_time", current_app.config.get("TIME_SOLVER", 120))),
        "coverage": float(form.get("coverage", 98)),
        "iterations": int(form.get("iterations", 3)),
        # constantes JEAN por defecto (idénticas a tu perfil)
        "agent_limit_factor": int(form.get("agent_limit_factor", 30)),
        "excess_penalty": float(form.get("excess_penalty", 5)),
        "peak_bonus": float(form.get("peak_bonus", 2)),
        "critical_bonus": float(form.get("critical_bonus", 2.5)),
        "use_pulp": True,
        "use_greedy": True,
        "export_files": False,
        "ft_first_pt_last": True,
    }


@bp.route("/generador", methods=["GET", "POST"])
def generador():
    if request.method == "GET":
        return render_template("generador.html", mode="sync")

    xls = request.files.get("file") or request.files.get("excel")
    if not xls:
        return jsonify({"error": "Falta el archivo Excel"}), 400

    cfg = _cfg_from_request(request.form)
    payload = run_opt(
        xls,
        config=cfg,
        generate_charts=True,
        job_id=None,
        return_payload=True,
    )
    return render_template("resultados.html", payload=payload, mode="sync")

