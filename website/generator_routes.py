from flask import Blueprint, render_template, request
from .scheduler import run_complete_optimization as run_opt

bp = Blueprint("generator", __name__)


def _is_on(form, *names):
    """Devuelve True si CUALQUIERA de los nombres llega encendido en el form."""
    for n in names:
        v = form.get(n)
        if v is None:
            continue
        if str(v).lower() in ("on", "true", "1", "yes"):
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

            # perfil/solver
            "random_seed": 42,
            "solver_threads": int(form.get("threads", 1)),
        }

        profile_name = _get_profile_from_form(form)
        cfg["profile"] = profile_name
        cfg["optimization_profile"] = profile_name

        # === NORMALIZACIÓN DE FAMILIAS ===
        cfg.update(
            {
                # Habilitar FT/PT (contratos)
                "use_ft": _is_on(form, "allow_ft", "full_time", "use_ft"),
                "use_pt": _is_on(form, "allow_pt", "part_time", "use_pt"),

                # Familias FT
                "ft_8h": _is_on(form, "ft_8h", "allow_8h", "ft_8h6d"),
                "ft_10h5d": _is_on(form, "ft_10h5d", "allow_10h8", "ft_10h"),

                # Familias PT
                "pt_4h6d": _is_on(form, "pt_4h6d", "allow_pt_4h"),
                "pt_5h5d": _is_on(form, "pt_5h5d", "allow_pt_5h"),
                "pt_6h4d": _is_on(form, "pt_6h4d", "allow_pt_6h"),
            }
        )

        # Solo si el usuario NO seleccionó ninguna familia, activamos un set seguro
        familias = ["ft_8h", "ft_10h5d", "pt_4h6d", "pt_5h5d", "pt_6h4d"]
        if not any(cfg.get(k) for k in familias):
            cfg["ft_8h"] = True
            cfg["pt_4h6d"] = True
            cfg["pt_6h4d"] = True
            print("[GEN] fallback aplicado: ft_8h + pt_4h6d + pt_6h4d")

        # (Opcional) si tu otra variante del scheduler espera las claves "allow_*", duplicamos:
        cfg["allow_8h"] = cfg["ft_8h"]
        cfg["allow_10h8"] = cfg["ft_10h5d"]
        cfg["allow_pt_4h"] = cfg["pt_4h6d"]
        cfg["allow_pt_5h"] = cfg["pt_5h5d"]
        cfg["allow_pt_6h"] = cfg["pt_6h4d"]

        # DEBUG: imprime qué familias quedaron realmente activas
        print("[GEN] familias activas:", [f for f in familias if cfg.get(f)])

        xls = request.files.get("file")
        if not xls:
            return render_template("generador.html", payload=None)

        payload = run_opt(
            xls, config=cfg, generate_charts=True, job_id=None, return_payload=True
        )
        return render_template("generador.html", payload=payload)

    return render_template("generador.html", payload=None)

