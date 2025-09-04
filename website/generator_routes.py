from flask import Blueprint, render_template, request, current_app
from .scheduler import run_complete_optimization as run_opt

bp = Blueprint("generator", __name__)

def _bool(name):  # checkbox helper
    return request.form.get(name) is not None

def _cfg_from_request():
    return {
        "iterations":           int(request.form.get("iterations", 30)),
        "solver_time":          int(request.form.get("solver_time", current_app.config.get("TIME_SOLVER", 240))),
        "solver_msg":           request.form.get("solver_msg", "1") == "1",
        "coverage":             float(request.form.get("coverage", 98)),
        "use_ft":               _bool("use_ft"),
        "use_pt":               _bool("use_pt"),
        "allow_8h":             _bool("allow_8h"),
        "allow_10h8":           _bool("allow_10h8"),
        "allow_pt_4h":          _bool("allow_pt_4h"),
        "allow_pt_5h":          _bool("allow_pt_5h"),
        "allow_pt_6h":          _bool("allow_pt_6h"),
        "break_from_start":     float(request.form.get("break_from_start", 2.0)),
        "break_from_end":       float(request.form.get("break_from_end", 2.0)),
        "optimization_profile": request.form.get("optimization_profile", "JEAN"),
        "profile":              request.form.get("optimization_profile", "JEAN"),  # alias para legacy
        "agent_limit_factor":   30,
        "excess_penalty":       5.0,
        "peak_bonus":           2.0,
        "critical_bonus":       2.5,
        "use_pulp":             True,
        "use_greedy":           True,
        "ft_first_pt_last":     True,
        "export_files":         False,
        "verbose":              _bool("verbose"),
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
