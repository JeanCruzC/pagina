# -*- coding: utf-8 -*-
"""
Perfiles de optimización EXACTOS del Streamlit original
"""

# Perfiles EXACTOS del Streamlit original
PROFILES = {
    "Equilibrado (Recomendado)": {
        "agent_limit_factor": 25,
        "excess_penalty": 2.5,
        "peak_hours": [11, 12, 13, 14, 15, 16],
        "peak_bonus": 1.5,
        "critical_days": [2, 4],
        "critical_bonus": 2.0,
        "hybrid": False
    },
    "100% Exacto": {
        "agent_limit_factor": 0,
        "excess_penalty": 2.5,
        "peak_hours": [],
        "peak_bonus": 0.0,
        "critical_days": [],
        "critical_bonus": 0.0,
        "hybrid": False
    },
    "JEAN": {
        "agent_limit_factor": 25,
        "excess_penalty": 5.0,
        "peak_hours": [11, 12, 13, 14, 15, 16],
        "peak_bonus": 2.0,
        "critical_days": [2, 4],
        "critical_bonus": 2.5,
        "hybrid": True
    },
}

def apply_profile(cfg: dict) -> dict:
    """Normaliza el nombre del perfil, aplica los pesos y deja logs claros."""
    name = (cfg.get("optimization_profile") or "JEAN").strip()
    name = PROFILE_ALIASES.get(name, name)
    profile = PROFILES.get(name)
    if not profile:
        print(f"[PROFILE] ADVERTENCIA: Perfil '{name}' no encontrado. Usando 'JEAN'.")
        name = "JEAN"
        profile = PROFILES[name]
    merged = {**cfg, **profile, "optimization_profile": name}
    print(f"[PROFILE] Aplicado: {name}")
    return merged

<<<<<<< HEAD
PROFILE_ALIASES = {
    "Jean": "JEAN",
    "jean": "JEAN",
    "JEAN BCR=100% + Greedy": "JEAN",
    "100% exacto": "100% Exacto",
    "Exacto": "100% Exacto",
    "Cobertura eficiente": "Equilibrado (Recomendado)",
}

def normalize_profile(name: str) -> str:
    return PROFILE_ALIASES.get(name, name)
=======
    # 1) Rellenar con defaults
    cfg = merge_config(cfg)

    # 2) Aplicar perfil seleccionado
    requested = cfg.get("optimization_profile") or "Equilibrado (Recomendado)"
    profile_name = resolve_profile_name(requested) or requested
    if profile_name != requested:
        print(f"[PROFILE] Usando alias '{requested}' -> '{profile_name}'")
    print(f"[PROFILE] Aplicando perfil: {profile_name}")
    params = PROFILES.get(profile_name, {})

    # 3) El perfil sobrescribe siempre
    if params:
        for key, val in params.items():
            cfg[key] = val

        # Normalizar nombre de cobertura si el perfil trae 'target_coverage'
        if "target_coverage" in params and "TARGET_COVERAGE" not in params:
            cfg["TARGET_COVERAGE"] = params["target_coverage"]

        # Configuraciones específicas por perfil
        if profile_name == "JEAN":
            cfg["optimization_profile"] = "JEAN"
            cfg["use_jean_search"] = True
            print(
                f"[PROFILE] JEAN: factor={cfg['agent_limit_factor']}, penalty={cfg['excess_penalty']}, target={cfg.get('TARGET_COVERAGE', 98)}%"
            )
        elif profile_name == "JEAN Personalizado":
            cfg["optimization_profile"] = "JEAN Personalizado"
            cfg["use_jean_search"] = True
            print(f"[PROFILE] JEAN Personalizado configurado")

        print(f"[PROFILE] Configuración aplicada - Strategy: {cfg.get('strategy', 'default')}")
    else:
        print(
            f"[PROFILE] ADVERTENCIA: Perfil '{profile_name}' no encontrado, usando configuración por defecto"
        )

    return cfg

# --- ALIASES para nombres de UI ---
ALIASES = {
    "100% Exacto": "Máxima Cobertura",
    "100% Cobertura Eficiente": "Máxima Cobertura",
    "Cobertura Perfecta": "Máxima Cobertura",
}


def resolve_profile_name(name: str) -> str:
    if not name:
        return None
    if name in PROFILES:
        return name
    return ALIASES.get(name)


def normalize_profile(name: str) -> str:
    return resolve_profile_name(name) or name
>>>>>>> 7579ccf68f3995aad4d6bc86549dc11d0a49be61
