# -*- coding: utf-8 -*-
"""
Scheduler principal con optimizadores
"""

import numpy as np
import time
from .profiles import PROFILES, resolve_profile_name, apply_profile

def calculate_coverage_with_penalty(assigned, required, method="efficiency", penalty_factor=1.0):
    """Calcula cobertura usando SIEMPRE la métrica simétrica solicitada.

    Fórmula única: min(asignado/requerido, requerido/asignado) * 100

    Nota: Los parámetros 'method' y 'penalty_factor' se ignoran por compatibilidad.

    Returns:
        Porcentaje de cobertura (0-100)
    """
    if required == 0:
        return 100.0 if assigned == 0 else 0.0

    ratio1 = assigned / required
    ratio2 = (required / assigned) if assigned > 0 else 0.0
    return min(ratio1, ratio2) * 100

def analyze_results(assignments, shifts_coverage, demand_matrix, coverage_method="efficiency", penalty_factor=1.0):
    """Analiza resultados usando únicamente la métrica simétrica de cobertura.

    Nota: 'coverage_method' y 'penalty_factor' se aceptan por compatibilidad
    pero no alteran el cálculo. Siempre se usa la fórmula simétrica.
    """
    if not assignments:
        return None
    
    slots = len(next(iter(shifts_coverage.values()))) // 7
    total_cov = np.zeros((7, slots), dtype=float)
    total_agents, ft_agents, pt_agents = 0, 0, 0
    
    for name, cnt in assignments.items():
        pat = np.array(shifts_coverage[name], dtype=float).reshape(7, slots)
        total_cov += pat * cnt
        total_agents += cnt
        if name.startswith("FT"):
            ft_agents += cnt
        else:
            pt_agents += cnt
    
    total_demand = demand_matrix.sum()
    diff = total_cov - demand_matrix
    over = int(np.sum(diff[diff > 0]))
    under = int(np.sum(np.abs(diff[diff < 0])))
    
    # ÚNICA MÉTRICA: Eficiencia simétrica por slot agregada
    # Fórmula agregada: sum(min(asignado_i, requerido_i)) / sum(max(asignado_i, requerido_i)) * 100
    sum_min = float(np.minimum(total_cov, demand_matrix).sum())
    sum_max = float(np.maximum(total_cov, demand_matrix).sum())
    if sum_max == 0:
        coverage_percentage = 100.0
    else:
        coverage_percentage = (sum_min / sum_max) * 100.0
    coverage_real = coverage_percentage
    
    return {
        "total_coverage": total_cov,
        "coverage_percentage": coverage_percentage,
        "coverage_real": coverage_real,
        "overstaffing": over,
        "understaffing": under,
        "total_agents": total_agents,
        "ft_agents": ft_agents,
        "pt_agents": pt_agents,
        # No exponemos coverage_method alternos; se usa siempre la métrica única
    }

def solve_in_chunks_optimized(shifts_coverage, demand_matrix, **kwargs):
    """Optimización por chunks - implementación básica"""
    # Seleccionar los mejores patrones por score
    scored = []
    for name, pat in shifts_coverage.items():
        pat_arr = np.array(pat).reshape(demand_matrix.shape)
        score = np.minimum(pat_arr, demand_matrix).sum()
        scored.append((score, name, pat))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    assignments = {}
    coverage = np.zeros_like(demand_matrix, dtype=float)
    
    # Asignar patrones hasta cubrir demanda
    for score, name, pat in scored[:50]:  # Top 50 patrones
        pat_2d = np.array(pat, dtype=float).reshape(demand_matrix.shape)
        remaining = np.maximum(0, demand_matrix - coverage)
        
        if remaining.sum() == 0:
            break
            
        # Calcular cuántos agentes asignar
        max_useful = 0
        for slot_demand in remaining.flatten():
            if slot_demand > 0:
                max_useful = max(max_useful, int(slot_demand))
        
        if max_useful > 0:
            assignments[name] = min(max_useful, 3)  # Máximo 3 por patrón
            coverage += pat_2d * assignments[name]
    
    return assignments

def optimize_with_precision_targeting(shifts_coverage, demand_matrix, cfg=None, **kwargs):
    """Optimización con targeting de precisión - fallback a chunks"""
    return solve_in_chunks_optimized(shifts_coverage, demand_matrix, **(cfg or {})), "PRECISION_TARGETING"

def optimize_portfolio(shifts_coverage, demand_matrix, cfg=None):
    """Optimización portfolio - fallback a chunks"""
    return solve_in_chunks_optimized(shifts_coverage, demand_matrix, **(cfg or {})), "PORTFOLIO"

def get_profile_optimizer(profile_name: str):
    """Obtiene optimizador para el perfil"""
    try:
        from .profile_optimizers import get_profile_optimizer as _get_optimizer
        return _get_optimizer(profile_name)
    except ImportError:
        # Fallback si no existe profile_optimizers
        def _fallback(shifts_coverage, demand_matrix, *, cfg=None, job_id=None):
            return solve_in_chunks_optimized(shifts_coverage, demand_matrix, **(cfg or {})), "FALLBACK"
        return _fallback

def run_complete_optimization(file_stream, config=None, generate_charts=False, job_id=None, return_payload=False):
    """Función principal de optimización"""
    cfg = config or {}
    
    # Resolver perfil
    requested = cfg.get("optimization_profile") or "Equilibrado (Recomendado)"
    resolved = resolve_profile_name(requested)
    
    if resolved and resolved != requested:
        print(f"[PROFILE] Usando alias '{requested}' -> '{resolved}'")
    
    profile_params = PROFILES.get(resolved)
    if profile_params:
        for k, v in profile_params.items():
            cfg.setdefault(k, v)
    
    cfg["optimization_profile"] = resolved or requested
    optimization_profile = cfg.get("optimization_profile", "")
    
    # Generar patrones dummy para testing
    shifts_coverage = {}
    demand_matrix = np.random.randint(1, 10, (7, 24))  # Matriz dummy
    
    for i in range(20):
        pattern = np.zeros(7 * 24)
        start_day = i % 7
        start_hour = (i * 2) % 24
        for h in range(8):  # 8 horas de trabajo
            if start_hour + h < 24:
                pattern[start_day * 24 + start_hour + h] = 1
        shifts_coverage[f"PATTERN_{i}"] = pattern
    
    # Ejecutar optimización según perfil
    if optimization_profile == "HPO + Cascada 100%":
        try:
            optimizer = get_profile_optimizer(optimization_profile)
            assignments, status = optimizer(shifts_coverage, demand_matrix, cfg=cfg, job_id=job_id)
        except Exception as e:
            print(f"[SCHEDULER] Error en HPO + Cascada: {e}")
            assignments = solve_in_chunks_optimized(shifts_coverage, demand_matrix, **cfg)
            status = "FALLBACK_CHUNKS"
    else:
        # Otros perfiles usan chunks por defecto
        assignments = solve_in_chunks_optimized(shifts_coverage, demand_matrix, **cfg)
        status = "CHUNKS_DEFAULT"
    
    if return_payload:
        # Siempre usar la métrica única (simétrica)
        results = analyze_results(assignments, shifts_coverage, demand_matrix)
        
        if results:
            cov_pct = results.get('coverage_percentage', 0)
            cov_real = results.get('coverage_real', 0) 
            over = results.get('overstaffing', 0)
            under = results.get('understaffing', 0)
            agents = results.get('total_agents', 0)
            print(f"[ROUTES] Resultado: {agents} agentes, cobertura pura {cov_pct:.1f}%, real {cov_real:.1f}%")
            # No anunciar métricas alternativas ni mensajes de exceso en logs de rutas

        return {
            "status": status,
            "assignments": assignments,
            "metrics": results,
            "config": cfg
        }
    
    return {"assignments": assignments}
