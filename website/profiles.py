# -*- coding: utf-8 -*-
"""
Perfiles de optimización EXACTOS del Streamlit original
"""

# Perfiles EXACTOS del legacy Streamlit
PROFILES = {
    "Equilibrado (Recomendado)": {
        "agent_limit_factor": 12,
        "excess_penalty": 2.0,
        "peak_bonus": 1.5,
        "critical_bonus": 2.0,
        "description": "Balance óptimo entre cobertura y costo",
        "strategy": "balanced",
        "solver_time": 300,
        "precision_mode": False,
    },
    "Conservador": {
        "agent_limit_factor": 30,
        "excess_penalty": 0.5,
        "peak_bonus": 1.0,
        "critical_bonus": 1.2,
        "description": "Minimiza riesgos, permite más agentes",
        "strategy": "conservative",
        "solver_time": 240,
        "precision_mode": False,
    },
    "Agresivo": {
        "agent_limit_factor": 15,
        "excess_penalty": 0.05,
        "peak_bonus": 1.5,
        "critical_bonus": 2.0,
        "description": "Maximiza eficiencia, tolera déficit menor",
        "strategy": "aggressive",
        "solver_time": 180,
        "precision_mode": True,
    },
    "Máxima Cobertura": {
        "agent_limit_factor": 7,
        "excess_penalty": 0.005,
        "peak_bonus": 3.0,
        "critical_bonus": 4.0,
        "description": "Prioriza cobertura completa sobre costo",
        "strategy": "max_coverage",
        "solver_time": 400,
        "precision_mode": True,
        "target_coverage": 99.5,
    },
    "Mínimo Costo": {
        "agent_limit_factor": 35,
        "excess_penalty": 0.8,
        "peak_bonus": 0.8,
        "critical_bonus": 1.0,
        "description": "Minimiza número de agentes",
        "strategy": "min_cost",
        "solver_time": 200,
        "precision_mode": False,
        "allow_deficit": True,
    },
    "JEAN": {
        "agent_limit_factor": 30,
        "excess_penalty": 5.0,
        "peak_bonus": 2.0,
        "critical_bonus": 2.5,
        "TARGET_COVERAGE": 98.0,
        "description": "Búsqueda iterativa para minimizar exceso+déficit",
        "strategy": "jean_search",
        "solver_time": 240,
        "precision_mode": True,
        "iterative_search": True,
        "search_iterations": 5,
        "use_jean_search": True,
    },
    "JEAN Personalizado": {
        "agent_limit_factor": 30,
        "excess_penalty": 5.0,
        "peak_bonus": 2.0,
        "critical_bonus": 2.5,
        "TARGET_COVERAGE": 98.0,
        "description": "JEAN con configuración personalizada de turnos",
        "strategy": "jean_custom",
        "solver_time": 240,
        "precision_mode": True,
        "custom_shifts": True,
        "ft_pt_strategy": True,
        "use_jean_search": True,
        "slot_duration_minutes": 30,
    },
    "Personalizado": {
        "agent_limit_factor": 25,
        "excess_penalty": 0.5,
        "peak_bonus": 1.5,
        "critical_bonus": 2.0,
        "description": "Configuración manual de parámetros",
        "strategy": "custom",
        "solver_time": 300,
        "precision_mode": False,
        "user_defined": True,
    },
}

def apply_profile(cfg=None):
    """Aplicar perfil de optimización sobre la configuración."""
    from .scheduler_core import merge_config

    # 1) Rellenar con defaults
    cfg = merge_config(cfg)

    # 2) Aplicar perfil seleccionado
    profile_name = cfg.get("optimization_profile") or "Equilibrado (Recomendado)"
    profile_name = normalize_profile(profile_name)
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
    "100% Exacto": "Cobertura Exacta",
    "100% Cobertura Eficiente": "Cobertura Eficiente 100",
    "100% Cobertura Total": "Cobertura Total 100",
    "Aprendizaje Adaptativo": "Adaptativo-Recomendado",
}


def normalize_profile(name: str) -> str:
    return ALIASES.get(name, name)
