import json
from flask import Blueprint, render_template, request, jsonify
from .scheduler import run_complete_optimization
from .profiles import apply_profile

bp = Blueprint("generator", __name__)

# -------------------------
# Helpers de parsing seguro
# -------------------------
TRUE_SET = {"true", "1", "on", "yes", "y", "si", "sí"}

def _is_on(val):
    if val is None:
        return False
    return str(val).strip().lower() in TRUE_SET

def _any_on(form, keys):
    # Retorna True si CUALQUIERA de las llaves viene marcada en el POST
    for k in keys:
        v = form.get(k)
        if v is None:
            # checkboxes pueden venir como list si usas <input name="...[]">
            lst = form.getlist(k)
            if lst and any(_is_on(x) for x in lst):
                return True
            continue
        if _is_on(v):
            return True
    return False

def _cfg_from_request(form):
    # PERFIL EXACTO (sin caer siempre en JEAN)
    # Acepta tanto "optimization_profile" (nuevo) como "profile" (legacy)
    optimization_profile = (
        form.get("optimization_profile")
        or form.get("profile")
        or "Equilibrado (Recomendado)"
    )

    # FAMILIAS: SOLO lo que el usuario marque.
    allow_8h    = _any_on(form, ["ft_8h", "allow_8h", "FT_8H"])
    allow_10h8  = _any_on(form, ["ft_10h5d", "allow_10h8", "FT_10H_5D", "ft_10h", "ft_10h_5d"])
    allow_pt_4h = _any_on(form, ["pt_4h6d", "allow_pt_4h", "PT_4H_6D", "pt_4h"])
    allow_pt_5h = _any_on(form, ["pt_5h5d", "allow_pt_5h", "PT_5H_5D", "pt_5h"])
    allow_pt_6h = _any_on(form, ["pt_6h4d", "allow_pt_6h", "PT_6H_4D", "pt_6h"])

    # Grupos FT/PT
    use_ft = allow_8h or allow_10h8
    use_pt = allow_pt_4h or allow_pt_5h or allow_pt_6h

    # Si NO marcó ninguna familia, devolvemos 400 (en el original ejecutas solo lo seleccionado)
    if not (use_ft or use_pt):
        raise ValueError("Selecciona al menos una familia de turnos (FT y/o PT).")

    # Otros parámetros (sin inventar opciones que no están en tu UI)
    solver_time = int(form.get("solver_time", 240))
    iterations  = int(form.get("iterations", 6))

    # Penalizaciones/bonos por si vienen del form (si no, usa defaults sanos)
    coverage      = float(form.get("coverage", 98))
    agent_factor  = int(form.get("agent_limit_factor", 30))
    excess_pen    = float(form.get("excess_penalty", 5))
    peak_bonus    = float(form.get("peak_bonus", 2))
    critical_bonus= float(form.get("critical_bonus", 2.5))

    return {
        "optimization_profile": optimization_profile,
        # Grupos
        "use_ft": use_ft,
        "use_pt": use_pt,
        # Familias exactas
        "allow_8h": allow_8h,
        "allow_10h8": allow_10h8,
        "allow_pt_4h": allow_pt_4h,
        "allow_pt_5h": allow_pt_5h,
        "allow_pt_6h": allow_pt_6h,
        # Parámetros del solver/estrategia
        "solver_time": solver_time,
        "iterations": iterations,
        "coverage": coverage,
        "agent_limit_factor": agent_factor,
        "excess_penalty": excess_pen,
        "peak_bonus": peak_bonus,
        "critical_bonus": critical_bonus,
        # Constantes del pipeline (no exponemos opciones que no están en la UI)
        "solver_msg": False,
        "use_pulp": True,
        "use_greedy": True,
        "export_files": True,
        "ft_first_pt_last": True,
    }

@bp.route("/generador", methods=["GET", "POST"])
def generador():
    if request.method == "GET":
        return render_template("generador.html", mode="sync")

    # POST: flujo síncrono
    xls = request.files.get("file") or request.files.get("excel")
    if not xls:
        return jsonify({"error": "Falta el archivo Excel"}), 400

    try:
        cfg = _cfg_from_request(request.form)
    except ValueError as e:
        # Respuesta limpia si no marcó ninguna familia
        return render_template(
            "generador.html",
            payload={"error": str(e), "config": {"optimization_profile": "—"}},
            mode="sync",
        )

    # 2) Aplica perfil (sobrescribe defaults)
    cfg = apply_profile(cfg)

    # 3) JEAN (Personalizado): JSON del usuario domina
    profile = (cfg.get("optimization_profile") or "").lower()
    if profile.startswith("jean"):
        raw = None
        if "jean_json" in request.files and request.files["jean_json"].filename:
            raw = request.files["jean_json"].read().decode("utf-8", "ignore")
        elif "jean_json" in request.form:
            raw = request.form.get("jean_json", "").strip()
        if raw:
            cfg["custom_shifts_json"] = raw
            try:
                user_overrides = json.loads(raw)
                cfg.update(user_overrides)
            except Exception:
                pass

    # 4) Ejecuta optimización con cfg final
    payload = run_complete_optimization(
        xls,
        config=cfg,
        generate_charts=True,
        job_id=None,
        return_payload=True  # <- devolvemos figuras/base64 + métricas + export
    )
    return render_template("generador.html", payload=payload, mode="sync")
