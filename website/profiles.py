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
