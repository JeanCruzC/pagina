# -*- coding: utf-8 -*-
"""
Configuración de cálculo de cobertura (métrica única)

Este módulo documenta la única métrica de cobertura permitida en el sistema:
Eficiencia de staffing (simétrica).
"""

# Métodos de cálculo de cobertura disponibles
ONLY_COVERAGE_METHOD = {
    "key": "efficiency",
    "name": "Eficiencia de Staffing (Simétrica)",
    "description": "Única métrica activa. Penaliza excesos y déficits simétricamente.",
    "formula": "min(asignado/requerido, requerido/asignado) * 100"
}

# Configuración por defecto
DEFAULT_COVERAGE_CONFIG = {
    "coverage_method": "efficiency",
}

# Ejemplos de cálculo para documentación
COVERAGE_EXAMPLES = [
    {
        "scenario": "Asignación perfecta",
        "required": 4,
        "assigned": 4,
        "efficiency": 100.0
    },
    {
        "scenario": "Exceso de 1 agente",
        "required": 4,
        "assigned": 5,
        "efficiency": 80.0
    },
    {
        "scenario": "Déficit de 1 agente",
        "required": 4,
        "assigned": 3,
        "efficiency": 75.0
    }
]

def get_coverage_method_info(method_name):
    """Compatibilidad: siempre retorna la métrica única si se solicita 'efficiency'."""
    if method_name == "efficiency":
        return ONLY_COVERAGE_METHOD
    return {}

def validate_coverage_config(config):
    """Valida configuración: exige 'efficiency' como única opción."""
    method = config.get("coverage_method", "efficiency")
    if method != "efficiency":
        raise ValueError(f"Método de cobertura inválido: {method}. Solo 'efficiency' está permitido.")
    return True
