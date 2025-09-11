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
        "agent_limit_factor": 15,
        "excess_penalty": 5.0,
        "peak_hours": [11, 12, 13, 14, 15, 16],
        "peak_bonus": 2.0,
        "critical_days": [2, 4],
        "critical_bonus": 2.5,
        "hybrid": True,
        "TARGET_COVERAGE": 100.0
    },
    "Cobertura Máxima (Completo)": {
        "agent_limit_factor": 15,
        "excess_penalty": 1.0,
        "peak_hours": [11, 12, 13, 14, 15, 16],
        "peak_bonus": 3.0,
        "critical_days": [2, 4],
        "critical_bonus": 4.0,
        "hybrid": False,
        "TARGET_COVERAGE": 100.0,
        "deficit_penalty": 1000.0
    },
    "Cobertura Real_Jean": {
        "agent_limit_factor": 15,
        "excess_penalty": 10.0,
        "peak_hours": [11, 12, 13, 14, 15, 16],
        "peak_bonus": 2.0,
        "critical_days": [2, 4],
        "critical_bonus": 2.0,
        "TARGET_COVERAGE": 100.0,
        "max_excess_ratio": 0.02,
        "allow_deficit": False,
        "solver_time": 300,
        "hybrid": False
    },
    "Grid Search Automático": {
        "agent_limit_factor": 15,
        "excess_penalty": 5.0,
        "peak_hours": [11, 12, 13, 14, 15, 16],
        "peak_bonus": 2.0,
        "critical_days": [2, 4],
        "critical_bonus": 2.0,
        "TARGET_COVERAGE": 100.0,
        "max_excess_ratio": 0.02,
        "allow_deficit": False,
        "solver_time": 60,
        "hybrid": False
    },
}

PROFILE_ALIASES = {
    "Jean": "JEAN",
    "jean": "JEAN",
    "JEAN BCR=100% + Greedy": "JEAN",
    "JEAN Personalizado": "JEAN",
    "100% exacto": "100% Exacto",
    "Exacto": "100% Exacto",
    "Cobertura eficiente": "Equilibrado (Recomendado)",
    "MAX_COBERTURA": "Cobertura Máxima (Completo)",
    "Máxima Cobertura": "Cobertura Máxima (Completo)",
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

def normalize_profile(name: str) -> str:
    return PROFILE_ALIASES.get(name, name)

def resolve_profile_name(name: str) -> str:
    if not name:
        return None
    if name in PROFILES:
        return name
    return PROFILE_ALIASES.get(name)