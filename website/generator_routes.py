from flask import Blueprint, render_template, request, jsonify
from .scheduler import run_complete_optimization

bp = Blueprint("generator", __name__)

def _cfg_from_request(form):
    # Mapea 1:1 con tus controles de Streamlit (nombres y defaults)
    return {
        "MAX_ITER": int(form.get("max_iter", 30)),
        "TIME_SOLVER": (lambda v: None if int(v) == 0 else float(v))(form.get("time_solver", 0)),
        "SOLVER_MSG": int(form.get("solver_msg", 1)),
        "SOLVER_THREADS": int(form.get("solver_threads", 1)),
        "TARGET_COVERAGE": float(form.get("coverage", 98)),
        "VERBOSE": form.get("verbose", "0") == "1",

        "use_ft": form.get("use_ft", "1") == "1",
        "use_pt": form.get("use_pt", "1") == "1",

        "allow_8h": form.get("allow_8h", "1") == "1",
        "allow_10h8": form.get("allow_10h8", "0") == "1",

        "allow_pt_4h": form.get("allow_pt_4h", "1") == "1",
        "allow_pt_5h": form.get("allow_pt_5h", "1") == "1",
        "allow_pt_6h": form.get("allow_pt_6h", "0") == "1",

        "break_from_start": float(form.get("break_from_start", 2)),
        "break_from_end": float(form.get("break_from_end", 2)),

        # Perfil JEAN por defecto (como tu app)
        "optimization_profile": form.get("profile", "JEAN"),
        "agent_limit_factor": int(form.get("agent_limit_factor", 30)),
        "excess_penalty": float(form.get("excess_penalty", 5)),
        "peak_bonus": float(form.get("peak_bonus", 2)),
        "critical_bonus": float(form.get("critical_bonus", 2.5)),
    }

@bp.route("/generador", methods=["GET", "POST"])
def generador():
    if request.method == "GET":
        return render_template("generador.html")

    xls = request.files.get("file")
    if not xls:
        return jsonify({"error": "Falta el archivo Excel"}), 400

    cfg = _cfg_from_request(request.form)
    payload = run_complete_optimization(
        xls, config=cfg, generate_charts=True, job_id=None, return_payload=True
    )
    return render_template("resultados.html", payload=payload)
