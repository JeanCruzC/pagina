# -*- coding: utf-8 -*-
"""
Funciones core del scheduler - versión core (rápida)
"""

def merge_config(cfg):
    """Mezcla configuración base con defaults"""
    base_config = {
        "agent_limit_factor": 15,
        "excess_penalty": 5.0,
        "peak_bonus": 2.0,
        "critical_bonus": 2.0,
        "TARGET_COVERAGE": 100.0,
        "allow_deficit": False,
        "allow_excess": True,
        "solver_time": 180,
        "hpo_trials": 12
    }
    
    if cfg:
        base_config.update(cfg)
    
    return base_config

# Re-export analyze_results from scheduler
def analyze_results(assignments, shifts_coverage, demand_matrix):
    """Wrapper para analyze_results del scheduler - versión core (rápida)"""
    from .scheduler import analyze_results as _analyze_results
    return _analyze_results(assignments, shifts_coverage, demand_matrix)