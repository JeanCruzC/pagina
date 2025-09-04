from flask import Blueprint, render_template, request
from .scheduler import run_complete_optimization as run_opt

bp = Blueprint("generator", __name__)


# helpers
def _bool(name: str) -> bool:
    v = request.form.get(name)
    return v in ("1", "true", "on", "yes")


def _cfg_from_form():
    return {
        # === idéntico a Streamlit ===
        "iterations": int(request.form.get("iterations", 3)),  # JEAN (sidebar)
        "solver_time": int(request.form.get("solver_time", 120)),  # Tiempo solver (s)
        "coverage": float(request.form.get("coverage", 98)),  # Cobertura objetivo (%)
        "break_from_start": float(request.form.get("break_from_start", 2)),
        "break_from_end": float(request.form.get("break_from_end", 2)),
        "use_ft": _bool("use_ft"),  # Full Time (48h)
        "use_pt": _bool("use_pt"),  # Part Time (24h)

        # Tipos PT habilitados (idéntico a app1.py)
        "allow_pt_4h": True,
        "allow_pt_5h": True,
        "allow_pt_6h": True,

        # FT permitidos (idéntico a app1.py)
        "allow_8h": True,
        "allow_10h8": True,

        # Solver avanzado
        "solver_msg": True,  # mostrar progreso
        "agent_limit_factor": int(request.form.get("agent_limit_factor", 30)),
        "excess_penalty": float(request.form.get("excess_penalty", 5)),
        "peak_bonus": float(request.form.get("peak_bonus", 2)),
        "critical_bonus": float(request.form.get("critical_bonus", 2.5)),

        # Perfil
        "optimization_profile": request.form.get("profile", "JEAN"),
        "profile": request.form.get("profile", "JEAN"),

        # Forzamos uso de PuLP + Greedy (como el streamlit)
        "use_pulp": True,
        "use_greedy": True,

        # Ejecutamos todo en el mismo request
        "ft_first_pt_last": True,
        "export_files": False,
    }


@bp.route("/generador", methods=["GET", "POST"])
def generador():
    if request.method == "GET":
        # UI estilo Streamlit (idéntica estructura y textos)
        return render_template("generador.html")

    # POST síncrono (modo Streamlit)
    xls = request.files.get("file")
    if not xls:
        return render_template("generador.html", error="Falta el archivo Excel")

    cfg = _cfg_from_form()
    payload = run_opt(
        xls,
        config=cfg,
        generate_charts=True,
        job_id=None,
        return_payload=True,
    )
    # Mostramos resultados con la misma estética/densidad visual de tu app
    return render_template("resultados.html", payload=payload, cfg=cfg)

