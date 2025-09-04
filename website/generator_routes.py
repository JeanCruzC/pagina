from flask import Blueprint, render_template, request, current_app, jsonify
from .scheduler import run_complete_optimization as run_opt

# (Opcional pero recomendado) Si usas CSRFProtect, exime este blueprint:
try:
    # En __init__.py habrá un objeto csrf registrado en app.extensions["csrf"]
    from flask import current_app as _current_app
    _csrf = _current_app.extensions.get("csrf")  # puede ser None aquí si aún no hay app context
except Exception:
    _csrf = None

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
        # MÁS RÁPIDO por defecto que tu job async: 15s como en tu Streamlit
        "solver_time": int(form.get("solver_time", current_app.config.get("TIME_SOLVER", 15))),
        "solver_msg": True,
        "coverage": float(form.get("coverage", 98)),
        "agent_limit_factor": int(form.get("agent_limit_factor", 30)),
        "excess_penalty": float(form.get("excess_penalty", 5)),
        "peak_bonus": float(form.get("peak_bonus", 2)),
        "critical_bonus": float(form.get("critical_bonus", 2.5)),
        # Para respuesta instantánea al probar: 2 iteraciones como base
        "iterations": int(form.get("iterations", 2)),
        "use_pulp": True,
        "use_greedy": True,
        "export_files": False,
        "ft_first_pt_last": True,
        "optimization_profile": form.get("optimization_profile", "JEAN"),
    }


@bp.route("/generador", methods=["GET", "POST"])
def generador():
    # Eximir CSRF en runtime si existe (evita 400 “The CSRF token is missing”)
    try:
        app = current_app._get_current_object()
        csrf_ext = app.extensions.get("csrf")
        if csrf_ext:
            csrf_ext.exempt(generador)
    except Exception:
        pass

    if request.method == "GET":
        return render_template("generador.html", mode="sync")

    # POST: ejecutar TODO en la MISMA request (modo Streamlit)
    xls = request.files.get("file") or request.files.get("excel")
    if not xls:
        return jsonify({"error": "Falta el archivo Excel"}), 400

    cfg = _cfg_from_request(request.form)
    payload = run_opt(
        xls,
        config=cfg,
        generate_charts=True,
        job_id=None,
        return_payload=True,  # <- devuelve figuras/base64 y métricas
    )
    return render_template("resultados.html", payload=payload, mode="sync")
