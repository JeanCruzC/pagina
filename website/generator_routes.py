from flask import Blueprint, render_template, request, current_app, jsonify
import math

# intenta leer bandera de PuLP desde tu core; si no existe, detecta localmente
try:
    from .scheduler import run_complete_optimization, PULP_AVAILABLE  # ya lo usas
except Exception:
    from .scheduler import run_complete_optimization
    try:
        import pulp  # type: ignore
        PULP_AVAILABLE = True
    except Exception:
        PULP_AVAILABLE = False

bp = Blueprint("generator", __name__)

# Perfiles EXACTOS como en Streamlit (mismo orden y textos)
# ref: st.sidebar.selectbox(... opciones ...)  :contentReference[oaicite:4]{index=4}
PROFILE_OPTIONS = [
    "Equilibrado (Recomendado)", "Conservador", "Agresivo",
    "Máxima Cobertura", "Mínimo Costo",
    "100% Cobertura Eficiente", "100% Cobertura Total",
    "Cobertura Perfecta", "100% Exacto",
    "JEAN", "JEAN Personalizado", "Personalizado", "Aprendizaje Adaptativo"
]


def _get_bool(name, default=False):
    v = request.form.get(name)
    if v is None:
        return default
    return v in ("1", "true", "True", "on", "yes")


def _cfg_from_request(form):
    # Controles y nombres igual que tu Streamlit sidebar
    # ref: “Iteraciones máximas”, “Tiempo solver (s)”, “Cobertura objetivo (%)”, etc. :contentReference[oaicite:5]{index=5}
    cfg = {
        "iterations": int(form.get("max_iter", 30)),
        "solver_time": (None if form.get("time_solver", "0") in ("0", "", None) else float(form.get("time_solver"))),
        "solver_msg": int(form.get("solver_msg", 1)),
        "solver_threads": int(form.get("solver_threads", 1)),
        "coverage": float(form.get("target_coverage", 98)),
        "verbose": _get_bool("verbose", False),

        # Contratos y turnos (mismos textos)
        "use_ft": _get_bool("use_ft", True),     # “Full Time (48h)” :contentReference[oaicite:6]{index=6}
        "use_pt": _get_bool("use_pt", True),     # “Part Time (24h)”
        "allow_8h": _get_bool("allow_8h", True),           # 8 horas (6 días) :contentReference[oaicite:7]{index=7}
        "allow_10h8": _get_bool("allow_10h8", False),      # 10h + día de 8h (5 días)
        "allow_pt_4h": _get_bool("allow_pt_4h", True),     # 4 horas (6 días) :contentReference[oaicite:8]{index=8}
        "allow_pt_6h": _get_bool("allow_pt_6h", True),     # 6 horas (4 días)
        "allow_pt_5h": _get_bool("allow_pt_5h", False),    # 5 horas (5 días)

        # Breaks
        "break_from_start": float(form.get("break_from_start", 2.5)),  # “Break desde inicio (horas)” :contentReference[oaicite:9]{index=9}
        "break_from_end": float(form.get("break_from_end", 2.5)),      # “Break antes del fin (horas)” :contentReference[oaicite:10]{index=10}

        # Perfil de optimización (selector)
        "optimization_profile": form.get("optimization_profile", "Equilibrado (Recomendado)"),  # :contentReference[oaicite:11]{index=11}
    }

    # Flags útiles para tu core
    cfg["use_pulp"] = True
    cfg["use_greedy"] = True

    return cfg


@bp.route("/generador", methods=["GET", "POST"])
def generador():
    if request.method == "GET":
        return render_template(
            "generador.html",
            pulp_available=PULP_AVAILABLE,
            profile_options=PROFILE_OPTIONS
        )

    # POST síncrono (modo Streamlit)
    xls = request.files.get("file") or request.files.get("excel")
    if not xls:
        return jsonify({"error": "Falta el archivo Excel"}), 400

    cfg = _cfg_from_request(request.form)

    payload = run_complete_optimization(
        xls,
        config=cfg,
        generate_charts=True,
        job_id=None,
        return_payload=True,
    )

    # Fallbacks por si tu core no devuelve todo
    payload.setdefault("analysis", {})
    payload.setdefault("status", "ok")
    payload.setdefault("elapsed", None)

    return render_template(
        "resultados.html",
        payload=payload,
        cfg=cfg,
        pulp_available=PULP_AVAILABLE,
        profile_options=PROFILE_OPTIONS
    )

