from flask import Blueprint, render_template, request
from .scheduler import run_complete_optimization as run_opt

bp = Blueprint("generator", __name__)


def _is_on(form, name):
    v = form.get(name)
    return str(v).lower() in ("on", "true", "1", "yes")


@bp.route("/generador", methods=["GET", "POST"])
def generador():
    if request.method == "POST":
        form = request.form

        cfg = {
            # núcleo
            "iterations": int(form.get("max_iters", 30)),
            "solver_time": int(form.get("solver_time", 240)),
            "solver_msg": _is_on(form, "verbose"),
            "target_coverage": int(form.get("target_coverage", 98)),

            # breaks
            "break_start_from": float(form.get("break_from", 2.0)),
            "break_before_end": float(form.get("break_to", 2.0)),

            # contrato (habilitan familias)
            "allow_ft": _is_on(form, "allow_ft") or _is_on(form, "full_time"),
            "allow_pt": _is_on(form, "allow_pt") or _is_on(form, "part_time"),

            # FT permitidos
            "ft_8h": _is_on(form, "ft_8h"),
            "ft_10h5d": _is_on(form, "ft_10h5d"),

            # PT permitidos
            "pt_4h6d": _is_on(form, "pt_4h6d"),
            "pt_6h4d": _is_on(form, "pt_6h4d"),
            "pt_5h5d": _is_on(form, "pt_5h5d"),

            # perfil/solver
            "optimization_profile": form.get("profile", "JEAN"),
            "random_seed": 42,
            "solver_threads": int(form.get("threads", 1)),
        }

        # --- Defaults de seguridad (igual que en tu Streamlit) ---
        if not any(
            [
                cfg["ft_8h"],
                cfg["ft_10h5d"],
                cfg["pt_4h6d"],
                cfg["pt_6h4d"],
                cfg["pt_5h5d"],
            ]
        ):
            cfg["ft_8h"] = True
            cfg["pt_4h6d"] = True
            cfg["pt_6h4d"] = True

        if not (cfg["allow_ft"] or cfg["allow_pt"]):
            cfg["allow_ft"] = True
            cfg["allow_pt"] = True

        xls = request.files.get("file")
        if not xls:
            return render_template("generador.html", payload=None)

        payload = run_opt(
            xls, config=cfg, generate_charts=True, job_id=None, return_payload=True
        )
        return render_template("generador.html", payload=payload)

    return render_template("generador.html", payload=None)

