from flask import Blueprint, render_template, request, current_app
from .scheduler import run_complete_optimization as run_opt

bp = Blueprint("generator", __name__)

def _cfg_from_request():
    form = request.form
    return {
        "iterations": int(form.get("iterations", 30)),
        "solver_time": int(form.get("solver_time", 240)),
        "solver_msg": form.get("solver_msg", "1") == "1",
        "coverage": float(form.get("coverage", 98)),
        "use_ft": form.get("use_ft", "true") == "true",
        "use_pt": form.get("use_pt", "true") == "true",
        "allow_8h": form.get("allow_8h", "true") == "true",
        "allow_10h8": form.get("allow_10h8", "true") == "true",
        "allow_pt_4h": form.get("allow_pt_4h", "true") == "true",
        "allow_pt_5h": form.get("allow_pt_5h", "false") == "true",
        "allow_pt_6h": form.get("allow_pt_6h", "true") == "true",
        "break_from_start": float(form.get("break_from_start", 2.5)),
        "break_from_end": float(form.get("break_from_end", 2.5)),
        "optimization_profile": form.get("optimization_profile", "JEAN"),
        "profile": form.get("optimization_profile", "JEAN"),
        "agent_limit_factor": 30,
        "excess_penalty": 5.0,
        "peak_bonus": 2.0,
        "critical_bonus": 2.5,
        "use_pulp": True,
        "use_greedy": True,
        "ft_first_pt_last": True,
        "export_files": False,
        "verbose": form.get("verbose") is not None,
    }

@bp.route("/generador", methods=["GET", "POST"])
def generador():
    if request.method == "GET":
        return render_template("generador.html", payload=None)

    xls = request.files.get("file")
    if not xls:
        return render_template("generador.html", payload=None)

    cfg = _cfg_from_request()
    payload = run_opt(
        xls, config=cfg, generate_charts=True, job_id=None, return_payload=True
    )
    return render_template("generador.html", payload=payload)
