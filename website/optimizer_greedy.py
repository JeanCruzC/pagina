# -*- coding: utf-8 -*-
"""
Optimizadores Greedy según especificaciones exactas
"""

import numpy as np
import time

def optimize_greedy_fast(shifts_coverage, demand_matrix, cfg=None, job_id=None):
    """Optimizador greedy rápido para HPO (máx 60s y heurística ligera)"""
    cfg = cfg or {}
    max_time = cfg.get("greedy_fast_time", 60)  # 60s máximo
    
    start_time = time.time()
    
    # Seleccionar patrones por eficiencia
    scored = []
    for name, pat in shifts_coverage.items():
        pat_arr = np.array(pat).reshape(demand_matrix.shape)
        coverage = np.minimum(pat_arr, demand_matrix).sum()
        hours = pat_arr.sum()
        efficiency = coverage / max(hours, 1)
        scored.append((efficiency, name, pat))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    assignments = {}
    coverage = np.zeros_like(demand_matrix)
    
    # Asignación greedy con límite de tiempo
    for efficiency, name, pat in scored:
        if time.time() - start_time > max_time:
            break
            
        pat_2d = np.array(pat).reshape(demand_matrix.shape)
        remaining = np.maximum(0, demand_matrix - coverage)
        
        if remaining.sum() == 0:
            break
        
        # Calcular beneficio marginal
        benefit = np.minimum(pat_2d, remaining).sum()
        if benefit > 0:
            assignments[name] = 1
            coverage += pat_2d
    
    return assignments, "GREEDY_FAST"

def optimize_with_greedy(shifts_coverage, demand_matrix, cfg=None, job_id=None):
    """Optimizador greedy completo (exacto del original, con timeout 120s)"""
    cfg = cfg or {}
    max_time = cfg.get("solver_time", 120)
    
    start_time = time.time()
    
    # Algoritmo greedy más completo
    assignments = {}
    coverage = np.zeros_like(demand_matrix)
    
    while time.time() - start_time < max_time:
        best_benefit = 0
        best_pattern = None
        
        # Encontrar el mejor patrón para agregar
        for name, pat in shifts_coverage.items():
            pat_2d = np.array(pat).reshape(demand_matrix.shape)
            remaining = np.maximum(0, demand_matrix - coverage)
            benefit = np.minimum(pat_2d, remaining).sum()
            
            if benefit > best_benefit:
                best_benefit = benefit
                best_pattern = (name, pat_2d)
        
        if best_benefit == 0 or best_pattern is None:
            break
        
        # Agregar el mejor patrón
        name, pat_2d = best_pattern
        assignments[name] = assignments.get(name, 0) + 1
        coverage += pat_2d
        
        # Verificar si se completó la cobertura
        if np.all(coverage >= demand_matrix):
            break
    
    return assignments, "GREEDY_COMPLETE"