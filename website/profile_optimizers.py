# -*- coding: utf-8 -*-
"""
Optimizadores específicos para cada perfil de optimización.
Implementa todas las lógicas y características específicas de cada perfil.

Nota:
    La implementación canónica de ``optimize_jean_search`` vive en
    ``website.scheduler``.  Este módulo mantiene un *wrapper* deprecado
    solo para compatibilidad con configuraciones legadas.
"""

import numpy as np
import time
import json
import os
import warnings
import inspect
try:
    import pulp as pl
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

from . import scheduler

try:
    from .scheduler import (
        merge_config,
        single_model,
        analyze_results,
        optimize_schedule_greedy_enhanced,
        optimize_with_precision_targeting,
        get_adaptive_params,
        save_execution_result,
        load_shift_patterns,
        optimize_ft_then_pt_strategy,
        _write_partial_result,
        optimize_jean_search,
    )
except Exception:  # pragma: no cover
    from scheduler import (
        merge_config,
        single_model,
        analyze_results,
        optimize_schedule_greedy_enhanced,
        optimize_with_precision_targeting,
        get_adaptive_params,
        save_execution_result,
        load_shift_patterns,
        optimize_ft_then_pt_strategy,
        _write_partial_result,
        optimize_jean_search,
    )

_sched_sig = inspect.signature(scheduler.optimize_jean_search)
_TARGET_COVERAGE_DEFAULT = _sched_sig.parameters["target_coverage"].default
_MAX_ITERATIONS_DEFAULT = _sched_sig.parameters["max_iterations"].default


@single_model
def optimize_minimum_cost(shifts_coverage, demand_matrix, *, cfg=None):
    """Optimización para mínimo costo (menos agentes)."""
    cfg = merge_config(cfg)
    
    print(f"[MIN_COST] Iniciando optimización de mínimo costo")
    
    if not PULP_AVAILABLE:
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix, cfg=cfg)
    
    try:
        shifts_list = list(shifts_coverage.keys())
        if not shifts_list:
            return {}, "NO_SHIFTS"
        
        prob = pl.LpProblem("Minimum_Cost", pl.LpMinimize)
        
        # Variables con límites restrictivos
        total_demand = demand_matrix.sum()
        max_per_shift = max(8, int(total_demand / cfg["agent_limit_factor"]))
        
        shift_vars = {}
        for shift in shifts_list:
            shift_vars[shift] = pl.LpVariable(f"shift_{shift}", 0, max_per_shift, pl.LpInteger)
        
        # Variables de déficit (permitir déficit controlado)
        deficit_vars = {}
        hours = demand_matrix.shape[1]
        patterns_unpacked = {
            s: p.reshape(7, hours) if len(p) == 7 * hours else p.reshape(7, -1)[:, :hours]
            for s, p in shifts_coverage.items()
        }
        
        for day in range(7):
            for hour in range(hours):
                deficit_vars[(day, hour)] = pl.LpVariable(f"deficit_{day}_{hour}", 0, None)
        
        # Función objetivo: priorizar minimizar agentes
        total_deficit = pl.lpSum([deficit_vars[(day, hour)] for day in range(7) for hour in range(hours)])
        total_agents = pl.lpSum([shift_vars[shift] for shift in shifts_list])
        
        # Prioridad en minimizar agentes, permitir déficit controlado
        prob += total_agents * 1000 + total_deficit * cfg["excess_penalty"]
        
        # Restricciones de cobertura (solo déficit)
        for day in range(7):
            for hour in range(hours):
                coverage = pl.lpSum([
                    shift_vars[shift] * patterns_unpacked[shift][day, hour]
                    for shift in shifts_list
                ])
                demand = demand_matrix[day, hour]
                
                prob += coverage + deficit_vars[(day, hour)] >= demand
        
        # Permitir hasta 15% de déficit si está configurado
        if cfg.get("allow_deficit", False):
            prob += total_deficit <= total_demand * 0.15
        
        # Límite muy restrictivo de agentes
        prob += total_agents <= int(total_demand / cfg["agent_limit_factor"])
        
        # Resolver
        solver = pl.PULP_CBC_CMD(
            msg=cfg.get("solver_msg", 0),
            timeLimit=cfg.get("solver_time", 200),
        )
        
        prob.solve(solver)
        
        assignments = {}
        if prob.status == pl.LpStatusOptimal:
            for shift in shifts_list:
                val = int(shift_vars[shift].varValue or 0)
                if val > 0:
                    assignments[shift] = val
            return assignments, "MINIMUM_COST"
        else:
            print(f"[MIN_COST] Status no óptimo: {prob.status}")
            return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix, cfg=cfg)
    
    except Exception as e:
        print(f"[MIN_COST] Error: {e}")
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix, cfg=cfg)


@single_model
def optimize_maximum_coverage(shifts_coverage, demand_matrix, *, cfg=None):
    """Optimización para máxima cobertura."""
    cfg = merge_config(cfg)
    
    print(f"[MAX_COV] Iniciando optimización de máxima cobertura")
    
    if not PULP_AVAILABLE:
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix, cfg=cfg)
    
    try:
        shifts_list = list(shifts_coverage.keys())
        if not shifts_list:
            return {}, "NO_SHIFTS"
        
        prob = pl.LpProblem("Maximum_Coverage", pl.LpMinimize)
        
        # Variables con límites generosos
        total_demand = demand_matrix.sum()
        max_per_shift = max(25, int(total_demand / cfg["agent_limit_factor"]))
        
        shift_vars = {}
        for shift in shifts_list:
            shift_vars[shift] = pl.LpVariable(f"shift_{shift}", 0, max_per_shift, pl.LpInteger)
        
        # Variables de déficit y exceso
        deficit_vars = {}
        excess_vars = {}
        hours = demand_matrix.shape[1]
        patterns_unpacked = {
            s: p.reshape(7, hours) if len(p) == 7 * hours else p.reshape(7, -1)[:, :hours]
            for s, p in shifts_coverage.items()
        }
        
        for day in range(7):
            for hour in range(hours):
                deficit_vars[(day, hour)] = pl.LpVariable(f"deficit_{day}_{hour}", 0, None)
                excess_vars[(day, hour)] = pl.LpVariable(f"excess_{day}_{hour}", 0, None)
        
        # Función objetivo: eliminar déficit completamente
        total_deficit = pl.lpSum([deficit_vars[(day, hour)] for day in range(7) for hour in range(hours)])
        total_excess = pl.lpSum([excess_vars[(day, hour)] for day in range(7) for hour in range(hours)])
        total_agents = pl.lpSum([shift_vars[shift] for shift in shifts_list])
        
        # Prioridad máxima en eliminar déficit
        prob += total_deficit * 1000000 + total_excess * cfg["excess_penalty"] + total_agents * 0.01
        
        # Restricciones de cobertura
        for day in range(7):
            for hour in range(hours):
                coverage = pl.lpSum([
                    shift_vars[shift] * patterns_unpacked[shift][day, hour]
                    for shift in shifts_list
                ])
                demand = demand_matrix[day, hour]
                
                prob += coverage + deficit_vars[(day, hour)] >= demand
                prob += coverage - excess_vars[(day, hour)] <= demand
        
        # Penalizaciones adicionales por días críticos y horas pico
        daily_totals = demand_matrix.sum(axis=1)
        hourly_totals = demand_matrix.sum(axis=0)
        critical_days = np.argsort(daily_totals)[-2:] if len(daily_totals) > 1 else [np.argmax(daily_totals)]
        peak_hours = np.where(hourly_totals >= np.percentile(hourly_totals[hourly_totals > 0], 75))[0]

        critical_penalty = 0
        for cd in critical_days:
            if cd < 7:
                for h in range(hours):
                    if demand_matrix[cd, h] > 0:
                        critical_penalty += deficit_vars[(cd, h)] * cfg["critical_bonus"] * 1000

        peak_penalty = 0
        for h in peak_hours:
            if h < hours:
                for d in range(7):
                    if demand_matrix[d, h] > 0:
                        peak_penalty += deficit_vars[(d, h)] * cfg["peak_bonus"] * 1000

        prob += critical_penalty + peak_penalty
        
        # Límite generoso de agentes
        prob += total_agents <= int(total_demand / max(1, cfg["agent_limit_factor"] - 3))
        
        # Resolver con tiempo extendido
        solver = pl.PULP_CBC_CMD(
            msg=cfg.get("solver_msg", 0),
            timeLimit=cfg.get("solver_time", 400),
        )
        
        prob.solve(solver)
        
        assignments = {}
        if prob.status == pl.LpStatusOptimal:
            for shift in shifts_list:
                val = int(shift_vars[shift].varValue or 0)
                if val > 0:
                    assignments[shift] = val
            return assignments, "MAXIMUM_COVERAGE"
        else:
            print(f"[MAX_COV] Status no óptimo: {prob.status}")
            return optimize_with_precision_targeting(
                shifts_coverage,
                demand_matrix,
                agent_limit_factor=cfg["agent_limit_factor"],
                excess_penalty=cfg["excess_penalty"],
                peak_bonus=cfg["peak_bonus"],
                critical_bonus=cfg["critical_bonus"],
                TIME_SOLVER=cfg.get("TIME_SOLVER", cfg.get("solver_time", 240)),
            )

    except Exception as e:
        print(f"[MAX_COV] Error: {e}")
        return optimize_with_precision_targeting(
            shifts_coverage,
            demand_matrix,
            agent_limit_factor=cfg["agent_limit_factor"],
            excess_penalty=cfg["excess_penalty"],
            peak_bonus=cfg["peak_bonus"],
            critical_bonus=cfg["critical_bonus"],
            TIME_SOLVER=cfg.get("TIME_SOLVER", cfg.get("solver_time", 240)),
        )


@single_model
def optimize_conservative(shifts_coverage, demand_matrix, *, cfg=None):
    """Optimización conservadora con margen de seguridad."""
    cfg = merge_config(cfg)
    
    print(f"[CONSERVATIVE] Iniciando optimización conservadora")
    
    if not PULP_AVAILABLE:
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix, cfg=cfg)
    
    try:
        shifts_list = list(shifts_coverage.keys())
        if not shifts_list:
            return {}, "NO_SHIFTS"
        
        prob = pl.LpProblem("Conservative_Scheduling", pl.LpMinimize)
        
        # Variables con límites generosos (enfoque conservador)
        total_demand = demand_matrix.sum()
        max_per_shift = max(20, int(total_demand / (cfg["agent_limit_factor"] - 5)))  # Más generoso
        
        shift_vars = {}
        for shift in shifts_list:
            shift_vars[shift] = pl.LpVariable(f"shift_{shift}", 0, max_per_shift, pl.LpInteger)
        
        # Variables de déficit y exceso
        deficit_vars = {}
        excess_vars = {}
        hours = demand_matrix.shape[1]
        patterns_unpacked = {
            s: p.reshape(7, hours) if len(p) == 7 * hours else p.reshape(7, -1)[:, :hours]
            for s, p in shifts_coverage.items()
        }
        
        for day in range(7):
            for hour in range(hours):
                deficit_vars[(day, hour)] = pl.LpVariable(f"deficit_{day}_{hour}", 0, None)
                excess_vars[(day, hour)] = pl.LpVariable(f"excess_{day}_{hour}", 0, None)
        
        # Función objetivo conservadora: evitar déficit, tolerar exceso
        total_deficit = pl.lpSum([deficit_vars[(day, hour)] for day in range(7) for hour in range(hours)])
        total_excess = pl.lpSum([excess_vars[(day, hour)] for day in range(7) for hour in range(hours)])
        total_agents = pl.lpSum([shift_vars[shift] for shift in shifts_list])
        
        # Penalizar fuertemente el déficit, ser suave con el exceso
        prob += total_deficit * 10000 + total_excess * cfg["excess_penalty"] + total_agents * 0.1
        
        # Restricciones de cobertura
        for day in range(7):
            for hour in range(hours):
                coverage = pl.lpSum([
                    shift_vars[shift] * patterns_unpacked[shift][day, hour]
                    for shift in shifts_list
                ])
                demand = demand_matrix[day, hour]
                
                prob += coverage + deficit_vars[(day, hour)] >= demand
                prob += coverage - excess_vars[(day, hour)] <= demand
        
        # Margen de seguridad: permitir hasta 20% de exceso
        prob += total_excess <= total_demand * 0.20
        
        # Límite generoso de agentes
        prob += total_agents <= int(total_demand / max(1, cfg["agent_limit_factor"] - 8))
        
        # Resolver
        solver = pl.PULP_CBC_CMD(
            msg=cfg.get("solver_msg", 0),
            timeLimit=cfg.get("solver_time", 240),
        )
        
        prob.solve(solver)
        
        assignments = {}
        if prob.status == pl.LpStatusOptimal:
            for shift in shifts_list:
                val = int(shift_vars[shift].varValue or 0)
                if val > 0:
                    assignments[shift] = val
            return assignments, "CONSERVATIVE"
        else:
            print(f"[CONSERVATIVE] Status no óptimo: {prob.status}")
            return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix, cfg=cfg)
    
    except Exception as e:
        print(f"[CONSERVATIVE] Error: {e}")
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix, cfg=cfg)


@single_model
def optimize_aggressive(shifts_coverage, demand_matrix, *, cfg=None):
    """Optimización agresiva para máxima eficiencia."""
    cfg = merge_config(cfg)
    
    print(f"[AGGRESSIVE] Iniciando optimización agresiva")
    
    if not PULP_AVAILABLE:
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix, cfg=cfg)
    
    try:
        shifts_list = list(shifts_coverage.keys())
        if not shifts_list:
            return {}, "NO_SHIFTS"
        
        prob = pl.LpProblem("Aggressive_Scheduling", pl.LpMinimize)
        
        # Variables con límites restrictivos
        total_demand = demand_matrix.sum()
        max_per_shift = max(12, int(total_demand / cfg["agent_limit_factor"]))
        
        shift_vars = {}
        for shift in shifts_list:
            shift_vars[shift] = pl.LpVariable(f"shift_{shift}", 0, max_per_shift, pl.LpInteger)
        
        # Variables de déficit y exceso
        deficit_vars = {}
        excess_vars = {}
        hours = demand_matrix.shape[1]
        patterns_unpacked = {
            s: p.reshape(7, hours) if len(p) == 7 * hours else p.reshape(7, -1)[:, :hours]
            for s, p in shifts_coverage.items()
        }
        
        for day in range(7):
            for hour in range(hours):
                deficit_vars[(day, hour)] = pl.LpVariable(f"deficit_{day}_{hour}", 0, None)
                excess_vars[(day, hour)] = pl.LpVariable(f"excess_{day}_{hour}", 0, None)
        
        # Función objetivo agresiva: minimizar agentes, tolerar déficit menor
        total_deficit = pl.lpSum([deficit_vars[(day, hour)] for day in range(7) for hour in range(hours)])
        total_excess = pl.lpSum([excess_vars[(day, hour)] for day in range(7) for hour in range(hours)])
        total_agents = pl.lpSum([shift_vars[shift] for shift in shifts_list])
        
        # Balance agresivo: priorizar eficiencia
        prob += total_agents * 100 + total_deficit * 50 + total_excess * cfg["excess_penalty"] * 1000
        
        # Restricciones de cobertura
        for day in range(7):
            for hour in range(hours):
                coverage = pl.lpSum([
                    shift_vars[shift] * patterns_unpacked[shift][day, hour]
                    for shift in shifts_list
                ])
                demand = demand_matrix[day, hour]
                
                prob += coverage + deficit_vars[(day, hour)] >= demand
                prob += coverage - excess_vars[(day, hour)] <= demand
        
        # Permitir déficit controlado (hasta 5%)
        prob += total_deficit <= total_demand * 0.05
        
        # Límite estricto de exceso
        prob += total_excess <= total_demand * 0.03
        
        # Límite agresivo de agentes
        prob += total_agents <= int(total_demand / cfg["agent_limit_factor"])
        
        # Resolver rápido
        solver = pl.PULP_CBC_CMD(
            msg=cfg.get("solver_msg", 0),
            timeLimit=cfg.get("solver_time", 180),
            gapRel=0.05  # Gap más amplio para velocidad
        )
        
        prob.solve(solver)
        
        assignments = {}
        if prob.status == pl.LpStatusOptimal:
            for shift in shifts_list:
                val = int(shift_vars[shift].varValue or 0)
                if val > 0:
                    assignments[shift] = val
            return assignments, "AGGRESSIVE"
        else:
            print(f"[AGGRESSIVE] Status no óptimo: {prob.status}")
            return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix, cfg=cfg)
    
    except Exception as e:
        print(f"[AGGRESSIVE] Error: {e}")
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix, cfg=cfg)


@single_model
def optimize_perfect_coverage(shifts_coverage, demand_matrix, *, cfg=None):
    """Optimización para cobertura perfecta (100%)."""
    cfg = merge_config(cfg)
    profile = cfg.get("optimization_profile", "")
    
    print(f"[PERFECT] Iniciando optimización de cobertura perfecta: {profile}")
    
    if not PULP_AVAILABLE:
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix, cfg=cfg)
    
    try:
        shifts_list = list(shifts_coverage.keys())
        if not shifts_list:
            return {}, "NO_SHIFTS"
        
        prob = pl.LpProblem("Perfect_Coverage", pl.LpMinimize)
        
        # Variables con límites según el perfil
        total_demand = demand_matrix.sum()
        if profile in ("100% Cobertura Total", "Cobertura Total 100"):
            max_per_shift = max(20, int(total_demand / 3))  # Más generoso
        else:
            max_per_shift = max(15, int(total_demand / cfg["agent_limit_factor"]))
        
        shift_vars = {}
        for shift in shifts_list:
            shift_vars[shift] = pl.LpVariable(f"shift_{shift}", 0, max_per_shift, pl.LpInteger)
        
        # Variables de déficit y exceso
        deficit_vars = {}
        excess_vars = {}
        hours = demand_matrix.shape[1]
        patterns_unpacked = {
            s: p.reshape(7, hours) if len(p) == 7 * hours else p.reshape(7, -1)[:, :hours]
            for s, p in shifts_coverage.items()
        }
        
        for day in range(7):
            for hour in range(hours):
                deficit_vars[(day, hour)] = pl.LpVariable(f"deficit_{day}_{hour}", 0, None)
                excess_vars[(day, hour)] = pl.LpVariable(f"excess_{day}_{hour}", 0, None)
        
        # Función objetivo según el perfil
        total_deficit = pl.lpSum([deficit_vars[(day, hour)] for day in range(7) for hour in range(hours)])
        total_excess = pl.lpSum([excess_vars[(day, hour)] for day in range(7) for hour in range(hours)])
        total_agents = pl.lpSum([shift_vars[shift] for shift in shifts_list])
        
        if profile in ("100% Exacto", "Cobertura Exacta"):
            # Prohibir cualquier déficit o exceso
            prob += total_deficit * 1000000 + total_excess * 1000000 + total_agents * 1
            # Restricciones estrictas
            prob += total_deficit == 0
            prob += total_excess == 0
        elif profile in ("100% Cobertura Eficiente", "Cobertura Eficiente 100"):
            # Minimizar exceso pero permitir cobertura completa
            prob += total_deficit * 100000 + total_excess * cfg["excess_penalty"] * 1000 + total_agents * 1
            prob += total_deficit == 0  # Sin déficit
            prob += total_excess <= total_demand * cfg.get("max_excess_ratio", 0.02)
        elif profile in ("100% Cobertura Total", "Cobertura Total 100"):
            # Cobertura completa sin restricciones de exceso
            prob += total_deficit * 100000 + total_excess * cfg["excess_penalty"] + total_agents * 0.1
            prob += total_deficit == 0  # Sin déficit
        else:
            # Cobertura Perfecta - balance
            prob += total_deficit * 50000 + total_excess * cfg["excess_penalty"] * 100 + total_agents * 1
        
        # Restricciones de cobertura
        for day in range(7):
            for hour in range(hours):
                coverage = pl.lpSum([
                    shift_vars[shift] * patterns_unpacked[shift][day, hour]
                    for shift in shifts_list
                ])
                demand = demand_matrix[day, hour]
                
                prob += coverage + deficit_vars[(day, hour)] >= demand
                prob += coverage - excess_vars[(day, hour)] <= demand
        
        # Resolver con tiempo extendido
        solver_time = cfg.get("solver_time", 600)
        solver = pl.PULP_CBC_CMD(
            msg=cfg.get("solver_msg", 0),
            timeLimit=solver_time,
            gapRel=0.001 if profile == "100% Exacto" else 0.01
        )
        
        prob.solve(solver)
        
        assignments = {}
        if prob.status == pl.LpStatusOptimal:
            for shift in shifts_list:
                val = int(shift_vars[shift].varValue or 0)
                if val > 0:
                    assignments[shift] = val
            return assignments, f"PERFECT_{profile.upper().replace(' ', '_')}"
        else:
            print(f"[PERFECT] Status no óptimo: {prob.status}, usando fallback")
            return optimize_with_precision_targeting(
                shifts_coverage,
                demand_matrix,
                agent_limit_factor=cfg["agent_limit_factor"],
                excess_penalty=cfg["excess_penalty"],
                peak_bonus=cfg["peak_bonus"],
                critical_bonus=cfg["critical_bonus"],
                TIME_SOLVER=cfg.get("TIME_SOLVER", cfg.get("solver_time", 240)),
            )

    except Exception as e:
        print(f"[PERFECT] Error: {e}")
        return optimize_with_precision_targeting(
            shifts_coverage,
            demand_matrix,
            agent_limit_factor=cfg["agent_limit_factor"],
            excess_penalty=cfg["excess_penalty"],
            peak_bonus=cfg["peak_bonus"],
            critical_bonus=cfg["critical_bonus"],
            TIME_SOLVER=cfg.get("TIME_SOLVER", cfg.get("solver_time", 240)),
        )


@single_model
def optimize_jean_search(
    shifts_coverage,
    demand_matrix,
    *,
    cfg=None,
    target_coverage=_TARGET_COVERAGE_DEFAULT,
    max_iterations=_MAX_ITERATIONS_DEFAULT,
    job_id=None,
):
    """Deprecated wrapper for ``website.scheduler.optimize_jean_search``.

    Parameters
    ----------
    shifts_coverage, demand_matrix:
        Datos de entrada para la búsqueda JEAN.
    cfg: dict | None
        Configuración legada; los parámetros relevantes se mapean a la
        implementación canónica.
    target_coverage, max_iterations, job_id:
        Parámetros compatibles con la función canónica.

    Returns
    -------
    tuple
        Resultado de :func:`website.scheduler.optimize_jean_search`.

    """

    warnings.warn(
        "profile_optimizers.optimize_jean_search está deprecado; "
        "usa website.scheduler.optimize_jean_search",
        DeprecationWarning,
        stacklevel=2,
    )

    cfg = merge_config(cfg)
    kwargs = {
        "agent_limit_factor": cfg.get("agent_limit_factor"),
        "excess_penalty": cfg.get("excess_penalty"),
        "peak_bonus": cfg.get("peak_bonus"),
        "critical_bonus": cfg.get("critical_bonus"),
        "iteration_time_limit": cfg.get(
            "iteration_time_limit", cfg.get("time_limit_seconds")
        ),
    }

    if (
        target_coverage == _TARGET_COVERAGE_DEFAULT
        and "TARGET_COVERAGE" in cfg
    ):
        target_coverage = cfg["TARGET_COVERAGE"]
    if max_iterations == _MAX_ITERATIONS_DEFAULT:
        if "search_iterations" in cfg:
            max_iterations = cfg["search_iterations"]
        elif "iterations" in cfg:
            max_iterations = cfg["iterations"]

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    return scheduler.optimize_jean_search(
        shifts_coverage,
        demand_matrix,
        target_coverage=target_coverage,
        max_iterations=max_iterations,
        job_id=job_id,
        **kwargs,
    )


@single_model
def optimize_jean_personalizado(shifts_coverage, demand_matrix, *, cfg=None):
    """JEAN Personalizado con configuración de turnos personalizada."""
    cfg = merge_config(cfg)
    
    print(f"[JEAN_CUSTOM] Iniciando JEAN Personalizado")
    
    # Si hay configuración de turnos personalizada, cargar patrones
    if cfg.get("custom_shifts") and cfg.get("shift_config_file"):
        try:
            custom_patterns = load_shift_patterns(
                cfg["shift_config_file"],
                slot_duration_minutes=cfg.get("slot_duration_minutes", 30),
                demand_matrix=demand_matrix,
                max_patterns=cfg.get("max_patterns")
            )
            if custom_patterns:
                shifts_coverage = custom_patterns
                print(f"[JEAN_CUSTOM] Cargados {len(custom_patterns)} patrones personalizados")
        except Exception as e:
            print(f"[JEAN_CUSTOM] Error cargando patrones personalizados: {e}")
    
    # Estrategia 2 fases si se usan FT y PT
    if cfg.get("ft_pt_strategy") and cfg.get("use_ft") and cfg.get("use_pt"):
        print(f"[JEAN_CUSTOM] Usando estrategia 2 fases FT->PT")
        assignments, status = optimize_ft_then_pt_strategy(
            shifts_coverage,
            demand_matrix,
            agent_limit_factor=cfg["agent_limit_factor"],
            excess_penalty=cfg["excess_penalty"],
            TIME_SOLVER=cfg.get("TIME_SOLVER", cfg.get("solver_time", 240)),
        )
        
        # Verificar si necesita refinamiento con búsqueda JEAN
        results = analyze_results(assignments, shifts_coverage, demand_matrix)
        if results:
            coverage = results["coverage_percentage"]
            score = results["overstaffing"] + results["understaffing"]
            target = cfg.get("TARGET_COVERAGE", 98.0)
            
            if coverage < target or score > 0:
                print(f"[JEAN_CUSTOM] Refinando con búsqueda JEAN (cov: {coverage:.1f}%, score: {score:.1f})")
                refined_assignments, refined_status = scheduler.optimize_jean_search(
                    shifts_coverage,
                    demand_matrix,
                    target_coverage=target,
                    max_iterations=cfg.get(
                        "search_iterations", cfg.get("iterations", 30)
                    ),
                    agent_limit_factor=cfg["agent_limit_factor"],
                    excess_penalty=cfg["excess_penalty"],
                    peak_bonus=cfg["peak_bonus"],
                    critical_bonus=cfg["critical_bonus"],
                    job_id=job_id if 'job_id' in locals() or 'job_id' in globals() else None,
                )
                if refined_assignments:
                    return refined_assignments, f"JEAN_CUSTOM_REFINED_{refined_status}"
        
        return assignments, f"JEAN_CUSTOM_{status}"
    else:
        # Usar búsqueda JEAN estándar
        return scheduler.optimize_jean_search(
            shifts_coverage,
            demand_matrix,
            target_coverage=cfg.get("TARGET_COVERAGE", 98.0),
            max_iterations=cfg.get(
                "search_iterations", cfg.get("iterations", 30)
            ),
            agent_limit_factor=cfg["agent_limit_factor"],
            excess_penalty=cfg["excess_penalty"],
            peak_bonus=cfg["peak_bonus"],
            critical_bonus=cfg["critical_bonus"],
            job_id=job_id if 'job_id' in locals() or 'job_id' in globals() else None,
        )


@single_model
def optimize_adaptive_learning(shifts_coverage, demand_matrix, *, cfg=None):
    """Optimización con aprendizaje adaptativo."""
    cfg = merge_config(cfg)
    
    print(f"[ADAPTIVE] Iniciando optimización adaptativa")
    
    start_time = time.time()
    
    # Aplicar parámetros adaptativos
    adaptive_params = get_adaptive_params(demand_matrix, cfg.get("TARGET_COVERAGE", 98.0))
    temp_cfg = cfg.copy()
    temp_cfg.update(adaptive_params)
    
    print(f"[ADAPTIVE] Parámetros evolutivos aplicados: {adaptive_params.get('evolution_step', 'unknown')}")
    
    # Usar optimización de precisión con parámetros adaptativos
    assignments, status = optimize_with_precision_targeting(
        shifts_coverage,
        demand_matrix,
        agent_limit_factor=temp_cfg["agent_limit_factor"],
        excess_penalty=temp_cfg["excess_penalty"],
        peak_bonus=temp_cfg["peak_bonus"],
        critical_bonus=temp_cfg["critical_bonus"],
        TIME_SOLVER=temp_cfg.get("TIME_SOLVER", temp_cfg.get("solver_time", 240)),
    )
    
    # Guardar resultado para aprendizaje
    if assignments:
        results = analyze_results(assignments, shifts_coverage, demand_matrix)
        if results:
            execution_time = time.time() - start_time
            save_execution_result(
                demand_matrix, 
                temp_cfg, 
                results["coverage_percentage"], 
                results["total_agents"], 
                execution_time
            )
            print(f"[ADAPTIVE] Resultado guardado para aprendizaje")
    
    return assignments, f"ADAPTIVE_{status}"


@single_model
def optimize_equilibrado(shifts_coverage, demand_matrix, *, cfg=None):
    """Optimización equilibrada (perfil recomendado)."""
    cfg = merge_config(cfg)
    
    print(f"[EQUILIBRADO] Iniciando optimización equilibrada")
    
    # Usar optimización estándar con parámetros equilibrados
    return optimize_with_precision_targeting(
        shifts_coverage,
        demand_matrix,
        agent_limit_factor=cfg["agent_limit_factor"],
        excess_penalty=cfg["excess_penalty"],
        peak_bonus=cfg["peak_bonus"],
        critical_bonus=cfg["critical_bonus"],
        TIME_SOLVER=cfg.get("TIME_SOLVER", cfg.get("solver_time", 240)),
    )


@single_model
def optimize_personalizado(shifts_coverage, demand_matrix, *, cfg=None):
    """Optimización personalizada con parámetros definidos por el usuario."""
    cfg = merge_config(cfg)
    
    print(f"[PERSONALIZADO] Iniciando optimización personalizada")
    print(f"[PERSONALIZADO] Parámetros: factor={cfg['agent_limit_factor']}, penalty={cfg['excess_penalty']}")
    
    # Usar optimización estándar con parámetros personalizados
    return optimize_with_precision_targeting(
        shifts_coverage,
        demand_matrix,
        agent_limit_factor=cfg["agent_limit_factor"],
        excess_penalty=cfg["excess_penalty"],
        peak_bonus=cfg["peak_bonus"],
        critical_bonus=cfg["critical_bonus"],
        TIME_SOLVER=cfg.get("TIME_SOLVER", cfg.get("solver_time", 240)),
    )


# --- HPO + Cascada 100% Implementation ---
import random

def _score_result_hpo(assignments, shifts_coverage, demand_matrix, target=100.0, coverage_method="original", penalty_factor=1.0):
    """Score unificado: |100 - cobertura_real| + (exceso + déficit) normalizados."""
    res = analyze_results(assignments, shifts_coverage, demand_matrix, coverage_method, penalty_factor)
    if not res:
        return float("inf")
    # Usar coverage_real que ahora puede ser con penalización según el método
    cov = res.get("coverage_real", 0.0)
    over = float(res.get("overstaffing", 0.0))
    under = float(res.get("understaffing", 0.0))
    total_dem = float(demand_matrix.sum() or 1.0)
    penalty_balance = (over + under) / total_dem * 100.0
    return abs(target - cov) + 0.25 * penalty_balance, res

def _sample_space_hpo():
    """Espacio pequeño y estable (rápido). Ajusta rangos si lo necesitas."""
    return {
        "agent_limit_factor": random.randint(8, 28),
        "excess_penalty": 10 ** random.uniform(-2, 0.7),  # ~[0.01, ~5]
        "peak_bonus": random.uniform(1.0, 3.5),
        "critical_bonus": random.uniform(1.0, 3.5),
        # switches suaves
        "allow_deficit": False,
        "allow_excess": True,   # dejamos un poco de exceso en HPO para evitar estancarse
    }

def _try_import_optuna():
    try:
        import optuna
        return optuna
    except Exception:
        return None

def _hpo_unico(shifts_coverage, demand_matrix, base_cfg, n_trials=12, job_id=None):
    """HPO rápido común a todos los solvers (eval barata con greedy_fast / chunks)."""
    from .scheduler_core import merge_config
    cfg = merge_config(base_cfg)
    optuna = _try_import_optuna()

    def eval_cfg(params):
        # Mezcla con cfg base; eval rápida (greedy_fast). Si falla, intenta chunks.
        cand = cfg.copy()
        cand.update(params)
        try:
            from .optimizer_greedy import optimize_greedy_fast
            assign, _ = optimize_greedy_fast(shifts_coverage, demand_matrix, cfg=cand, job_id=job_id)
        except Exception:
            # Import tardío para evitar ciclos en import
            from .scheduler import solve_in_chunks_optimized
            assign = solve_in_chunks_optimized(shifts_coverage, demand_matrix, **cand)
        coverage_method = cand.get("coverage_method", "original")
        penalty_factor = cand.get("penalty_factor", 1.0)
        score, _ = _score_result_hpo(assign, shifts_coverage, demand_matrix, target=cand.get("TARGET_COVERAGE", 100.0), coverage_method=coverage_method, penalty_factor=penalty_factor)
        return score, cand

    # --- Ruta A: Optuna si está disponible (TPE) ---
    if optuna:
        def objective(trial):
            params = {
                "agent_limit_factor": trial.suggest_int("agent_limit_factor", 8, 28),
                "excess_penalty": trial.suggest_float("excess_penalty", 0.01, 5.0, log=True),
                "peak_bonus": trial.suggest_float("peak_bonus", 1.0, 3.5),
                "critical_bonus": trial.suggest_float("critical_bonus", 1.0, 3.5),
                "allow_excess": True,
                "allow_deficit": False,
            }
            score, _ = eval_cfg(params)
            return score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        best.update({"allow_excess": True, "allow_deficit": False})
        return best

    # --- Ruta B: Random Search (sin dependencias) ---
    best_params, best_score = None, float("inf")
    for _ in range(max(3, n_trials)):
        params = _sample_space_hpo()
        score, _ = eval_cfg(params)
        if score < best_score:
            best_params, best_score = params, score
            if best_score < 1e-3:  # prácticamente perfecto
                break
    return best_params or {}

@single_model
def optimize_hpo_then_solve(shifts_coverage, demand_matrix, *, cfg=None, job_id=None):
    """
    1) Busca hiperparámetros una sola vez (rápido).
    2) Ejecuta cascada con esa config: PuLP -> Greedy -> Chunks.
    3) Devuelve la mejor asignación según el score unificado.
    """
    from .scheduler_core import merge_config
    cfg = merge_config(cfg)
    target = cfg.get("TARGET_COVERAGE", 100.0)
    trials = int(cfg.get("hpo_trials", 12))

    # ---------- Fase 1: HPO único ----------
    best_cfg = _hpo_unico(shifts_coverage, demand_matrix, cfg, n_trials=trials, job_id=job_id)
    run_cfg = cfg.copy()
    run_cfg.update(best_cfg)

    # ---------- Fase 2: Cascada ----------
    candidates = []

    # 2a) PuLP (HiGHS si procede) con tiempo acotado (usa tu lógica interna)
    try:
        from .optimizer_pulp import optimize_with_pulp
        a_pulp, s_pulp = optimize_with_pulp(shifts_coverage, demand_matrix, cfg=run_cfg, job_id=job_id)
        if a_pulp:
            coverage_method = run_cfg.get("coverage_method", "original")
            penalty_factor = run_cfg.get("penalty_factor", 1.0)
            sc, m = _score_result_hpo(a_pulp, shifts_coverage, demand_matrix, target=target, coverage_method=coverage_method, penalty_factor=penalty_factor)
            candidates.append(("PULP", sc, a_pulp, m))
    except Exception:
        pass

    # 2b) Greedy completo
    try:
        from .optimizer_greedy import optimize_with_greedy
        a_greedy, s_gr = optimize_with_greedy(shifts_coverage, demand_matrix, cfg=run_cfg, job_id=job_id)
        if a_greedy:
            coverage_method = run_cfg.get("coverage_method", "original")
            penalty_factor = run_cfg.get("penalty_factor", 1.0)
            sc, m = _score_result_hpo(a_greedy, shifts_coverage, demand_matrix, target=target, coverage_method=coverage_method, penalty_factor=penalty_factor)
            candidates.append(("GREEDY", sc, a_greedy, m))
    except Exception:
        pass

    # 2c) Fallback: chunks
    try:
        from .scheduler import solve_in_chunks_optimized  # import tardío evita ciclos
        a_chunks = solve_in_chunks_optimized(shifts_coverage, demand_matrix, **run_cfg)
        if a_chunks:
            coverage_method = run_cfg.get("coverage_method", "original")
            penalty_factor = run_cfg.get("penalty_factor", 1.0)
            sc, m = _score_result_hpo(a_chunks, shifts_coverage, demand_matrix, target=target, coverage_method=coverage_method, penalty_factor=penalty_factor)
            candidates.append(("CHUNKS", sc, a_chunks, m))
    except Exception:
        pass

    if not candidates:
        return {}, "HPO_CASCADE_NO_SOLUTION"

    # Elige la mejor por menor score
    candidates.sort(key=lambda t: t[1])
    best_name, best_score, best_assign, best_metrics = candidates[0]
    print(f"[HPO+CASCADA] Mejor: {best_name} | score={best_score:.3f} | cov={best_metrics.get('coverage_percentage', 0):.1f}%")
    return best_assign, f"HPO_CASCADE_{best_name}"

# Mapeo de perfiles a funciones optimizadoras
PROFILE_OPTIMIZERS = {
    "Equilibrado (Recomendado)": optimize_equilibrado,
    "Conservador": optimize_conservative,
    "Agresivo": optimize_aggressive,
    "Máxima Cobertura": optimize_maximum_coverage,
    "Mínimo Costo": optimize_minimum_cost,
    "100% Cobertura Eficiente": optimize_perfect_coverage,
    "Cobertura Eficiente 100": optimize_perfect_coverage,
    "100% Cobertura Total": optimize_perfect_coverage,
    "Cobertura Total 100": optimize_perfect_coverage,
    "Cobertura Perfecta": optimize_perfect_coverage,
    "100% Exacto": optimize_perfect_coverage,
    "Cobertura Exacta": optimize_perfect_coverage,
    "JEAN": optimize_jean_search,
    "JEAN Personalizado": optimize_jean_personalizado,
    "Personalizado": optimize_personalizado,
    "Aprendizaje Adaptativo": optimize_adaptive_learning,
    "Adaptativo-Recomendado": optimize_adaptive_learning,
    "HPO + Cascada 100%": optimize_hpo_then_solve,
    "HPO y Cascada": optimize_hpo_then_solve,
}

# Actualizar el mapeo con los nuevos perfiles
PROFILE_OPTIMIZERS.update({
    "HPO + Cascada 100%": optimize_hpo_then_solve,
    "HPO y Cascada": optimize_hpo_then_solve,  # alias opcional
})


def get_profile_optimizer(profile_name):
    """Obtiene la función optimizadora para un perfil específico."""
    optimizer = PROFILE_OPTIMIZERS.get(profile_name, optimize_equilibrado)
    
    # Wrapper para pasar job_id a los optimizadores que lo soportan
    def optimizer_wrapper(shifts_coverage, demand_matrix, *, cfg=None, job_id=None, **kwargs):
        import inspect
        sig = inspect.signature(optimizer)
        if 'job_id' in sig.parameters:
            return optimizer(shifts_coverage, demand_matrix, cfg=cfg, job_id=job_id, **kwargs)
        else:
            return optimizer(shifts_coverage, demand_matrix, cfg=cfg, **kwargs)
    
    return optimizer_wrapper


def select_best_result_by_profile(pulp_result, greedy_result, profile, cfg=None):
    """Selecciona el mejor resultado según el perfil de optimización."""
    cfg = merge_config(cfg)
    
    pulp_assignments, pulp_status = pulp_result
    greedy_assignments, greedy_status = greedy_result
    
    # Si solo hay un resultado, usarlo
    if not pulp_assignments and greedy_assignments:
        return greedy_assignments, f"GREEDY_{greedy_status}"
    if pulp_assignments and not greedy_assignments:
        return pulp_assignments, f"PULP_{pulp_status}"
    if not pulp_assignments and not greedy_assignments:
        return {}, "NO_RESULTS"
    
    # Criterios de selección por perfil
    if profile in ["Mínimo Costo", "Agresivo"]:
        # Preferir menor número de agentes
        pulp_agents = sum(pulp_assignments.values())
        greedy_agents = sum(greedy_assignments.values())
        if greedy_agents < pulp_agents:
            return greedy_assignments, f"GREEDY_{greedy_status}_SELECTED"
        else:
            return pulp_assignments, f"PULP_{pulp_status}_SELECTED"
    
    elif profile in ["Máxima Cobertura", "100% Cobertura Total", "100% Cobertura Eficiente"]:
        # Preferir PuLP para mejor cobertura
        return pulp_assignments, f"PULP_{pulp_status}_SELECTED"
    
    elif profile in ["JEAN", "JEAN Personalizado"]:
        # Siempre usar PuLP para JEAN
        return pulp_assignments, f"PULP_{pulp_status}_SELECTED"
    
    else:
        # Balance general - usar el que tenga mejor ratio cobertura/agentes
        try:
            pulp_score = sum(pulp_assignments.values()) * 1.1  # Penalizar ligeramente PuLP
            greedy_score = sum(greedy_assignments.values())
            
            if greedy_score <= pulp_score:
                return greedy_assignments, f"GREEDY_{greedy_status}_SELECTED"
            else:
                return pulp_assignments, f"PULP_{pulp_status}_SELECTED"
        except:
            # Fallback: usar PuLP
            return pulp_assignments, f"PULP_{pulp_status}_SELECTED"


def apply_profile_specific_greedy_config(cfg):
    """Aplica configuración específica del greedy según el perfil."""
    profile = cfg.get("optimization_profile", "Equilibrado (Recomendado)")
    
    # Configuraciones específicas por perfil para el algoritmo greedy
    if profile == "Mínimo Costo":
        cfg["greedy_cost_weight"] = 10.0  # Priorizar costo
        cfg["greedy_coverage_weight"] = 1.0
        cfg["greedy_excess_tolerance"] = 0.1
    elif profile == "Máxima Cobertura":
        cfg["greedy_cost_weight"] = 0.1
        cfg["greedy_coverage_weight"] = 10.0  # Priorizar cobertura
        cfg["greedy_excess_tolerance"] = 0.5
    elif profile == "Agresivo":
        cfg["greedy_cost_weight"] = 5.0
        cfg["greedy_coverage_weight"] = 2.0
        cfg["greedy_excess_tolerance"] = 0.05
    elif profile == "Conservador":
        cfg["greedy_cost_weight"] = 1.0
        cfg["greedy_coverage_weight"] = 3.0
        cfg["greedy_excess_tolerance"] = 0.3
    else:
        # Equilibrado
        cfg["greedy_cost_weight"] = 1.0
        cfg["greedy_coverage_weight"] = 2.0
        cfg["greedy_excess_tolerance"] = 0.2
    
    return cfg