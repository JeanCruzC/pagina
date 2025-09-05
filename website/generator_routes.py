from flask import Blueprint, render_template, request
from .scheduler import run_complete_optimization as run_opt

bp = Blueprint("generator", __name__)


def _is_on(form, *keys):
    for k in keys:
        v = form.get(k)
        if isinstance(v, str) and v.lower() in ("on", "true", "1", "yes"):
            return True
        if v is True:
            return True
    return False


def _get_profile_from_form(form):
    # Acepta cualquiera de estos nombres tal como vienen del front
    for k in ("optimization_profile", "profile", "perfil"):
        v = form.get(k)
        if v and str(v).strip():
            return str(v).strip()
    # Si no llega nada, usa un perfil neutral (no fuerces JEAN)
    return "Equilibrado (Recomendado)"


def _normalize_cfg_for_scheduler(form):
    # Toggles de tipo de contrato
    use_ft = _is_on(form, "use_ft", "allow_ft", "full_time", "ft")
    use_pt = _is_on(form, "use_pt", "allow_pt", "part_time", "pt")

    # Familias FT esperadas por el scheduler
    allow_8h = use_ft and _is_on(form, "allow_8h", "ft_8h", "ft_8h6d", "ft8")
    allow_10h8 = use_ft and _is_on(
        form, "allow_10h8", "ft_10h5d", "ft_10h_5d", "ft_10h"
    )

    # Familias PT esperadas por el scheduler
    allow_pt_4h = use_pt and _is_on(form, "allow_pt_4h", "pt_4h6d", "pt_4h")
    allow_pt_5h = use_pt and _is_on(form, "allow_pt_5h", "pt_5h5d", "pt_5h")
    allow_pt_6h = use_pt and _is_on(form, "allow_pt_6h", "pt_6h5d", "pt_6h4d", "pt_6h")

    # Seguridad: si el usuario no activa ningún tipo, mantenemos el comportamiento antiguo
    if not (use_ft or use_pt):
        use_ft = True
        use_pt = True

    # Seguridad: si no activó ninguna familia, caemos al mínimo seguro (8h FT + 4h PT)
    if not (allow_8h or allow_10h8 or allow_pt_4h or allow_pt_5h or allow_pt_6h):
        allow_8h = use_ft
        allow_pt_4h = use_pt

    return {
        "use_ft": use_ft,
        "use_pt": use_pt,
        "allow_8h": allow_8h,
        "allow_10h8": allow_10h8,
        "allow_pt_4h": allow_pt_4h,
        "allow_pt_5h": allow_pt_5h,
        "allow_pt_6h": allow_pt_6h,
    }


@bp.route("/generador", methods=["GET", "POST"])
def generador():
    if request.method == "POST":
        form = request.form

        cfg = {k: (str(v).lower() in ("on", "true", "1", "yes")) for k, v in form.items()}

        cfg.update(
            {
                # núcleo
                "iterations": int(form.get("max_iters", 30)),
                "solver_time": int(form.get("solver_time", 240)),
                "solver_msg": _is_on(form, "verbose"),
                "target_coverage": int(form.get("target_coverage", 98)),

                # breaks
                "break_start_from": float(form.get("break_from", 2.0)),
                "break_before_end": float(form.get("break_to", 2.0)),

                # perfil/solver
                "random_seed": 42,
                "solver_threads": int(form.get("threads", 1)),
            }
        )

        profile_name = _get_profile_from_form(form)
        cfg["profile"] = profile_name
        cfg["optimization_profile"] = profile_name

        cfg.update(_normalize_cfg_for_scheduler(form))

        for k in [
            "full_time",
            "part_time",
            "ft_8h",
            "ft_10h5d",
            "pt_4h6d",
            "pt_6h4d",
            "pt_6h5d",
            "pt_5h5d",
        ]:
            cfg.pop(k, None)

        xls = request.files.get("file")
        if not xls:
            return render_template("generador.html", payload=None)

        payload = run_opt(
            xls, config=cfg, generate_charts=True, job_id=None, return_payload=True
        )
        return render_template("generador.html", payload=payload)

    return render_template("generador.html", payload=None)

