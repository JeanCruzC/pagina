# -*- coding: utf-8 -*-
"""
Optimizador PuLP/ILP según especificaciones exactas
"""

import numpy as np

def optimize_with_pulp(shifts_coverage, demand_matrix, cfg=None, job_id=None):
    """Optimizador PuLP (ILP con time limit) - usa tu lógica interna"""
    cfg = cfg or {}
    
    try:
        import pulp
        
        # Crear problema de optimización
        prob = pulp.LpProblem("ShiftOptimization", pulp.LpMinimize)
        
        # Variables de decisión
        shift_vars = {}
        shifts_list = list(shifts_coverage.keys())
        
        for i, shift in enumerate(shifts_list):
            shift_vars[shift] = pulp.LpVariable(f"x_{i}", 0, None, pulp.LpInteger)
        
        # Variables de déficit y exceso
        D, H = demand_matrix.shape
        deficit_vars = {}
        excess_vars = {}
        
        for d in range(D):
            for h in range(H):
                deficit_vars[(d, h)] = pulp.LpVariable(f"def_{d}_{h}", 0, None)
                excess_vars[(d, h)] = pulp.LpVariable(f"exc_{d}_{h}", 0, None)
        
        # Función objetivo
        total_deficit = pulp.lpSum(deficit_vars.values())
        total_excess = pulp.lpSum(excess_vars.values())
        total_agents = pulp.lpSum(shift_vars.values())
        
        excess_penalty = cfg.get("excess_penalty", 1.0)
        prob += total_deficit * 1000 + total_excess * excess_penalty + total_agents * 0.01
        
        # Restricciones de cobertura
        for d in range(D):
            for h in range(H):
                coverage = pulp.lpSum([
                    shift_vars[s] * np.array(shifts_coverage[s]).reshape(D, H)[d, h]
                    for s in shifts_list
                ])
                prob += coverage + deficit_vars[(d, h)] - excess_vars[(d, h)] == int(demand_matrix[d, h])
        
        # Resolver con HiGHS→CBC si puede; mantener ese orden ayuda
        solver_time = cfg.get("solver_time", 180)
        
        try:
            # Intentar HiGHS primero
            solver = pulp.HiGHS_CMD(msg=False, timeLimit=solver_time)
            prob.solve(solver)
        except:
            # Fallback a CBC
            solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=solver_time)
            prob.solve(solver)
        
        # Extraer solución
        assignments = {}
        if prob.status == pulp.LpStatusOptimal:
            for shift in shifts_list:
                val = int(shift_vars[shift].varValue or 0)
                if val > 0:
                    assignments[shift] = val
        
        return assignments, "PULP_OPTIMAL"
        
    except ImportError:
        # Fallback si no hay PuLP
        from .scheduler import optimize_schedule_greedy_enhanced
        assignments = optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix, cfg=cfg)
        return assignments, "PULP_FALLBACK_GREEDY"
    except Exception as e:
        print(f"[PULP] Error: {e}")
        from .scheduler import optimize_schedule_greedy_enhanced
        assignments = optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix, cfg=cfg)
        return assignments, "PULP_ERROR_FALLBACK"