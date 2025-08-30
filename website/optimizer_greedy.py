# -*- coding: utf-8 -*-
"""
Optimizador Greedy - Lógica EXACTA 1:1 del Streamlit original
"""
import time
import numpy as np
from .scheduler_core import merge_config

def optimize_with_greedy(shifts_coverage, demand_matrix, *, cfg=None, job_id=None):
    """Algoritmo greedy mejorado EXACTO del original."""
    print("[GREEDY] Iniciando algoritmo greedy")
    cfg = merge_config(cfg)
    
    agent_limit_factor = cfg["agent_limit_factor"]
    excess_penalty = cfg["excess_penalty"]
    peak_bonus = cfg["peak_bonus"]
    critical_bonus = cfg["critical_bonus"]
    
    shifts_list = list(shifts_coverage.keys())
    assignments = {}
    current_coverage = np.zeros_like(demand_matrix, dtype=float)
    
    # Límite de agentes EXACTO del original
    total_demand = demand_matrix.sum()
    max_agents = max(100, int(total_demand / max(1, agent_limit_factor - 5)))
    
    print(f"[GREEDY] Procesando {len(shifts_list)} turnos, max {max_agents} agentes")
    
    # Análisis de patrones críticos EXACTO del original
    daily_totals = demand_matrix.sum(axis=1)
    hourly_totals = demand_matrix.sum(axis=0)
    if daily_totals.size == 0 or daily_totals.max() == 0:
        critical_days = []
    else:
        critical_days = (
            np.argpartition(daily_totals, -2)[-2:]
            if daily_totals.size > 1
            else [int(np.argmax(daily_totals))]
        )
    peak_threshold = (
        np.percentile(hourly_totals[hourly_totals > 0], 75)
        if np.any(hourly_totals > 0)
        else 0
    )
    peak_hours = np.where(hourly_totals >= peak_threshold)[0]
    
    print(f"[GREEDY] Días críticos: {critical_days}, Horas pico: {peak_hours}")
    
    start_time = time.time()
    max_time = 120  # Máximo 2 minutos
    
    for iteration in range(max_agents):
        # Verificar timeout
        if time.time() - start_time > max_time:
            print(f"[GREEDY] Timeout alcanzado ({max_time}s)")
            break
        
        if iteration % 25 == 0:
            print(f"[GREEDY] Iteración {iteration}/{max_agents}")
            # Parar temprano si ya tenemos una buena solución
            if iteration > 50:
                remaining_deficit = np.sum(np.maximum(0, demand_matrix - current_coverage))
                if remaining_deficit < demand_matrix.sum() * 0.05:  # Menos del 5% de déficit
                    print(f"[GREEDY] Parada temprana: déficit < 5%")
                    break
        
        best_shift = None
        best_score = -float("inf")
        best_pattern = None
        
        for shift_name in shifts_list:
            try:
                # Obtener patrón del turno EXACTO del original
                slots_per_day = len(shifts_coverage[shift_name]) // 7
                base_pattern = np.array(shifts_coverage[shift_name]).reshape(7, slots_per_day)
                
                # Ajustar dimensiones si es necesario
                if slots_per_day != demand_matrix.shape[1]:
                    pattern = np.zeros((7, demand_matrix.shape[1]))
                    cols_to_copy = min(slots_per_day, demand_matrix.shape[1])
                    pattern[:, :cols_to_copy] = base_pattern[:, :cols_to_copy]
                else:
                    pattern = base_pattern
                
                new_coverage = current_coverage + pattern
                
                # Cálculo de score EXACTO del original
                current_deficit = np.maximum(0, demand_matrix - current_coverage)
                new_deficit = np.maximum(0, demand_matrix - new_coverage)
                deficit_reduction = np.sum(current_deficit - new_deficit)
                
                # Penalización inteligente de exceso EXACTA del original
                current_excess = np.maximum(0, current_coverage - demand_matrix)
                new_excess = np.maximum(0, new_coverage - demand_matrix)
                
                smart_excess_penalty = 0
                for day in range(7):
                    for hour in range(demand_matrix.shape[1]):
                        if demand_matrix[day, hour] == 0 and new_excess[day, hour] > current_excess[day, hour]:
                            smart_excess_penalty += 1000  # Penalización extrema
                        elif demand_matrix[day, hour] <= 2:
                            smart_excess_penalty += (new_excess[day, hour] - current_excess[day, hour]) * excess_penalty * 10
                        else:
                            smart_excess_penalty += (new_excess[day, hour] - current_excess[day, hour]) * excess_penalty
                
                # Bonificaciones por patrones críticos EXACTAS del original
                critical_bonus_score = 0
                for critical_day in critical_days:
                    if critical_day < 7:
                        day_improvement = np.sum(current_deficit[critical_day] - new_deficit[critical_day])
                        critical_bonus_score += day_improvement * critical_bonus * 2
                
                peak_bonus_score = 0
                for hour in peak_hours:
                    if hour < demand_matrix.shape[1]:
                        hour_improvement = np.sum(current_deficit[:, hour] - new_deficit[:, hour])
                        peak_bonus_score += hour_improvement * peak_bonus * 2
                
                # Score final EXACTO del original
                score = (deficit_reduction * 100 + 
                        critical_bonus_score + 
                        peak_bonus_score - 
                        smart_excess_penalty)
                
                if score > best_score:
                    best_score = score
                    best_shift = shift_name
                    best_pattern = pattern
                    
            except Exception as e:
                print(f"[GREEDY] Error procesando {shift_name}: {e}")
                continue
        
        # Criterio de parada EXACTO del original
        if best_shift is None or best_score <= 0.5:
            print(f"[GREEDY] Parada en iteración {iteration}, mejor score: {best_score}")
            break
        
        # Aplicar mejor turno
        if best_shift not in assignments:
            assignments[best_shift] = 0
        assignments[best_shift] += 1
        current_coverage += best_pattern
        
        # Verificar si se completó la cobertura
        remaining_deficit = np.sum(np.maximum(0, demand_matrix - current_coverage))
        if remaining_deficit == 0:
            print(f"[GREEDY] Cobertura completa en iteración {iteration}")
            break
    
    elapsed = time.time() - start_time
    total_agents = sum(assignments.values())
    print(f"[GREEDY] Completado en {elapsed:.1f}s: {total_agents} agentes en {len(assignments)} turnos")
    
    return assignments, "GREEDY_ENHANCED"

def optimize_greedy_fast(shifts_coverage, demand_matrix, *, cfg=None, job_id=None):
    """Versión rápida del greedy para fallback."""
    print("[GREEDY_FAST] Iniciando greedy rápido")
    cfg = merge_config(cfg)
    
    shifts_list = list(shifts_coverage.keys())
    assignments = {}
    current_coverage = np.zeros_like(demand_matrix, dtype=float)
    
    # Límite reducido para velocidad
    max_agents = min(200, int(demand_matrix.sum() / max(1, cfg["agent_limit_factor"] - 3)))
    
    start_time = time.time()
    max_time = 60  # Máximo 1 minuto
    
    for iteration in range(max_agents):
        # Verificar timeout más frecuentemente
        if iteration % 10 == 0 and time.time() - start_time > max_time:
            print(f"[GREEDY_FAST] Timeout alcanzado ({max_time}s)")
            break
        
        best_shift = None
        best_score = -float("inf")
        best_pattern = None
        
        # Evaluación simplificada para velocidad
        for shift_name in shifts_list:
            try:
                slots_per_day = len(shifts_coverage[shift_name]) // 7
                base_pattern = np.array(shifts_coverage[shift_name]).reshape(7, slots_per_day)
                
                if slots_per_day != demand_matrix.shape[1]:
                    pattern = np.zeros((7, demand_matrix.shape[1]))
                    cols_to_copy = min(slots_per_day, demand_matrix.shape[1])
                    pattern[:, :cols_to_copy] = base_pattern[:, :cols_to_copy]
                else:
                    pattern = base_pattern
                
                new_coverage = current_coverage + pattern
                
                # Score simplificado para velocidad
                current_deficit = np.maximum(0, demand_matrix - current_coverage)
                new_deficit = np.maximum(0, demand_matrix - new_coverage)
                deficit_reduction = np.sum(current_deficit - new_deficit)
                
                # Penalización básica de exceso
                excess_increase = np.sum(np.maximum(0, new_coverage - demand_matrix)) - np.sum(np.maximum(0, current_coverage - demand_matrix))
                
                score = deficit_reduction * 100 - excess_increase * cfg["excess_penalty"]
                
                if score > best_score:
                    best_score = score
                    best_shift = shift_name
                    best_pattern = pattern
                    
            except Exception:
                continue
        
        if best_shift is None or best_score <= 0:
            break
        
        if best_shift not in assignments:
            assignments[best_shift] = 0
        assignments[best_shift] += 1
        current_coverage += best_pattern
        
        # Verificar cobertura completa
        if np.sum(np.maximum(0, demand_matrix - current_coverage)) == 0:
            break
    
    elapsed = time.time() - start_time
    total_agents = sum(assignments.values())
    print(f"[GREEDY_FAST] Completado en {elapsed:.1f}s: {total_agents} agentes")
    
    return assignments, "GREEDY_FAST"