# -*- coding: utf-8 -*-
"""
Optimizador PuLP - Lógica EXACTA 1:1 del Streamlit original
"""
import time
import numpy as np
from threading import RLock
from functools import wraps

try:
    import pulp as pl
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False
    pl = None

from .scheduler_core import merge_config, analyze_results

try:
from .scheduler import _write_partial_result, _compact_patterns_for
except Exception:  # pragma: no cover - fallback if scheduler not available
    def _write_partial_result(*args, **kwargs):
        pass

# Lock para evitar múltiples modelos simultáneos
_MODEL_LOCK = RLock()

def single_model(func):
    """Asegurar que solo un modelo de optimización se ejecute a la vez."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with _MODEL_LOCK:
            return func(*args, **kwargs)
    return wrapper

@single_model
def optimize_with_pulp(shifts_coverage, demand_matrix, *, cfg=None, job_id=None):
    """Optimización con PuLP - EXACTA del Streamlit original."""
    print("[PULP] Iniciando optimización PuLP")
    cfg = merge_config(cfg)
    
    if not PULP_AVAILABLE:
        print("[PULP] PuLP no disponible")
        return {}, "PULP_NOT_AVAILABLE"
    
    try:
        shifts_list = list(shifts_coverage.keys())
        if not shifts_list:
            return {}, "NO_SHIFTS"
        
        print(f"[PULP] Procesando {len(shifts_list)} turnos")
        
        # Crear problema de optimización
        prob = pl.LpProblem("Schedule_Optimization", pl.LpMinimize)
        
        # Variables con límites dinámicos EXACTOS del original
        total_demand = demand_matrix.sum()
        peak_demand = demand_matrix.max()
        max_per_shift = max(20, int(total_demand / max(1, cfg["agent_limit_factor"])))
        
        shift_vars = {}
        for shift in shifts_list:
            shift_vars[shift] = pl.LpVariable(f"shift_{shift}", 0, max_per_shift, pl.LpInteger)
        
        # Variables de déficit y exceso
        deficit_vars = {}
        excess_vars = {}
        hours = demand_matrix.shape[1]
        
        # Desempaquetar patrones
        patterns_unpacked = {}
        for s, p in shifts_coverage.items():
            if len(p) == 7 * hours:
                patterns_unpacked[s] = np.array(p).reshape(7, hours)
            else:
                slots_per_day = len(p) // 7
                pattern_temp = np.array(p).reshape(7, slots_per_day)
                pattern_matrix = np.zeros((7, hours))
                cols_to_copy = min(slots_per_day, hours)
                pattern_matrix[:, :cols_to_copy] = pattern_temp[:, :cols_to_copy]
                patterns_unpacked[s] = pattern_matrix
        
        for day in range(7):
            for hour in range(hours):
                deficit_vars[(day, hour)] = pl.LpVariable(f"deficit_{day}_{hour}", 0, None)
                excess_vars[(day, hour)] = pl.LpVariable(f"excess_{day}_{hour}", 0, None)
        
        # Análisis de patrones críticos EXACTO del original
        daily_totals = demand_matrix.sum(axis=1)
        hourly_totals = demand_matrix.sum(axis=0)
        critical_days = np.argsort(daily_totals)[-2:] if len(daily_totals) > 1 else [np.argmax(daily_totals)]
        peak_threshold = np.percentile(hourly_totals[hourly_totals > 0], 75) if np.any(hourly_totals > 0) else 0
        peak_hours = np.where(hourly_totals >= peak_threshold)[0]
        
        # Función objetivo EXACTA del original
        total_deficit = pl.lpSum([deficit_vars[(day, hour)] for day in range(7) for hour in range(hours)])
        total_excess = pl.lpSum([excess_vars[(day, hour)] for day in range(7) for hour in range(hours)])
        total_agents = pl.lpSum([shift_vars[shift] for shift in shifts_list])
        
        # Bonificaciones por días críticos y horas pico EXACTAS del original
        critical_bonus_value = 0
        peak_bonus_value = 0
        
        # Días críticos
        for critical_day in critical_days:
            if critical_day < 7:
                for hour in range(hours):
                    if demand_matrix[critical_day, hour] > 0:
                        critical_bonus_value -= deficit_vars[(critical_day, hour)] * cfg["critical_bonus"]
        
        # Horas pico
        for hour in peak_hours:
            if hour < hours:
                for day in range(7):
                    if demand_matrix[day, hour] > 0:
                        peak_bonus_value -= deficit_vars[(day, hour)] * cfg["peak_bonus"]
        
        # Función objetivo EXACTA del original
        prob += (total_deficit * 1000 + 
                 total_excess * cfg["excess_penalty"] + 
                 total_agents * 0.1 + 
                 critical_bonus_value + 
                 peak_bonus_value)
        
        # Restricciones de cobertura EXACTAS del original
        for day in range(7):
            for hour in range(hours):
                coverage = pl.lpSum([
                    shift_vars[shift] * patterns_unpacked[shift][day, hour]
                    for shift in shifts_list
                ])
                demand = demand_matrix[day, hour]
                
                prob += coverage + deficit_vars[(day, hour)] >= demand
                prob += coverage - excess_vars[(day, hour)] <= demand
        
        # Límite dinámico de agentes EXACTO del original
        dynamic_agent_limit = max(
            int(total_demand / max(1, cfg["agent_limit_factor"])),
            int(peak_demand * 1.1),
        )
        prob += total_agents <= dynamic_agent_limit
        
        # Resolver con configuración EXACTA del legacy
        solver_time = cfg.get("solver_time", 240)  # EXACTO del legacy - sin límite artificial
        
        print(f"[PULP] Resolviendo con timeout {solver_time}s")
        start_time = time.time()
        
        # Progreso real basado en tiempo y nodos
        def update_progress():
            elapsed = time.time() - start_time
            max_time = min(solver_time, 120)  # Máximo 2 minutos
            
            # Progreso basado en tiempo (mínimo)
            time_progress = (elapsed / max_time) * 100
            
            # Mostrar progreso real
            if elapsed < 30:
                progress = min(time_progress, 25)  # Primeros 30s: hasta 25%
                status_msg = "Inicializando modelo..."
            elif elapsed < 60:
                progress = min(time_progress, 50)  # Siguiente 30s: hasta 50%
                status_msg = "Resolviendo..."
            elif elapsed < 90:
                progress = min(time_progress, 75)  # Siguiente 30s: hasta 75%
                status_msg = "Optimizando solución..."
            else:
                progress = min(time_progress, 95)  # Últimos 30s: hasta 95%
                status_msg = "Finalizando..."
            
            try:
                from .extensions import scheduler as store
                store.update_progress(job_id, {
                    "pulp_progress": f"{progress:.0f}%",
                    "pulp_time": f"{elapsed:.0f}s/{max_time}s",
                    "pulp_status": status_msg
                })
            except:
                pass
        
        # Usar solver con límites más agresivos
        status = None
        try:
            # Actualizar progreso cada 5s
            import threading
            def periodic_update():
                update_progress()
                if time.time() - start_time < 60:  # Solo 1 minuto máximo
                    timer = threading.Timer(3.0, periodic_update)
                    timer.daemon = True
                    timer.start()
            
            periodic_update()
            
            # Solver con límites muy agresivos para terminar rápido
            solver = pl.PULP_CBC_CMD(
                msg=True,
                timeLimit=solver_time,
                threads=0,
            )
            try:
                status = prob.solve(solver)
                print(f"[PULP] Solver status: {status}")
            except Exception as e:
                print(f"[PULP] Error con solver configurado: {str(e)[:100]}")
                _write_partial_result(job_id, None, None, None, meta={"error": str(e)})
                # Fallback a solver simple con timeout muy corto
                try:
                    print(f"[PULP] Intentando solver simple")
                    simple_solver = pl.PULP_CBC_CMD(timeLimit=solver_time, msg=True, threads=0)
                    status = prob.solve(simple_solver)
                    print(f"[PULP] Solver simple status: {status}")
                except Exception as e2:
                    print(f"[PULP] Error final: {str(e2)[:100]}")
                    _write_partial_result(job_id, None, None, None, meta={"error": str(e2)})
                    return {}, "SOLVER_ERROR"
        except KeyboardInterrupt:
            print(f"[PULP] Interrumpido por usuario")
            return {}, "PULP_INTERRUPTED"
        
        solve_time = time.time() - start_time
        
        # Verificar si excedió el tiempo máximo
        if solve_time > 180:  # Más de 3 minutos
            print(f"[PULP] Timeout manual después de {solve_time:.1f}s")
            # Intentar extraer solución parcial si existe
            try:
                for shift in shifts_list:
                    try:
                        value = int(shift_vars[shift].varValue or 0)
                        if value > 0:
                            assignments[shift] = value
                    except (TypeError, AttributeError):
                        pass
                if assignments:
                    total_agents = sum(assignments.values())
                    print(f"[PULP] Solución parcial extraída: {total_agents} agentes")
                    return assignments, "PULP_TIMEOUT_PARTIAL"
            except:
                pass
            return {}, "PULP_TIMEOUT"
        
        # Progreso final
        try:
            from .extensions import scheduler as store
            store.update_progress(job_id, {
                "pulp_progress": "100%",
                "pulp_time": f"{solve_time:.0f}s"
            })
        except:
            pass
        
        # Extraer solución EXACTA del original
        assignments = {}
        status = getattr(prob, 'status', -1)
        
        if status == pl.LpStatusOptimal:
            for shift in shifts_list:
                try:
                    value = int(shift_vars[shift].varValue or 0)
                    if value > 0:
                        assignments[shift] = value
                except (TypeError, AttributeError):
                    pass
            method = "PULP_OPTIMAL"
            print(f"[PULP] Solución óptima encontrada en {solve_time:.1f}s")
        else:
            # Intentar extraer solución aunque el status no sea óptimo
            for shift in shifts_list:
                try:
                    value = int(shift_vars[shift].varValue or 0)
                    if value > 0:
                        assignments[shift] = value
                except (TypeError, AttributeError):
                    pass
            method = f"PULP_STATUS_{status}"
            print(f"[PULP] Status {status} en {solve_time:.1f}s")
        
        total_agents = sum(assignments.values()) if assignments else 0
        print(f"[PULP] Completado: {total_agents} agentes en {len(assignments)} turnos")
        
        return assignments, method
        
    except Exception as e:
        import traceback
        print(f"[PULP] Error: {e}")
        print(f"[PULP] Traceback: {traceback.format_exc()[:200]}")
        return {}, f"PULP_ERROR"

@single_model
def optimize_jean_search(shifts_coverage, demand_matrix, *, cfg=None, target_coverage=98.0, max_iterations=5, job_id=None):
    """Búsqueda iterativa JEAN EXACTA del original."""
    print("[JEAN] Iniciando búsqueda JEAN")
    cfg = merge_config(cfg)
    
    if not PULP_AVAILABLE:
        print("[JEAN] PuLP no disponible")
        return {}, "JEAN_NO_PULP"
    
    start_time = time.time()
    max_time = 180  # Máximo 3 minutos para JEAN
    
    original_factor = cfg["agent_limit_factor"]
    best_assignments = {}
    best_method = ""
    best_score = float("inf")
    best_coverage = 0
    
    # Secuencia de factores EXACTA del legacy Streamlit
    factor_sequence = [30, 25, 20, 15, 12, 10, 8]  # EXACTA del legacy
    factor_sequence = [f for f in factor_sequence if f <= original_factor]
    
    if not factor_sequence:
        factor_sequence = [original_factor]

    print(f"[JEAN] Secuencia de factores: {factor_sequence}")

    for iteration, factor in enumerate(factor_sequence[:max_iterations]):
        # Verificar timeout
        if time.time() - start_time > max_time:
            print(f"[JEAN] Timeout alcanzado ({max_time}s)")
            break

        print(f"[JEAN] Iteración {iteration + 1}: factor {factor}")

        # Snapshot inicial antes de resolver la iteración
        try:
            if job_id is not None:
                D, H = demand_matrix.shape
                pat_small = _compact_patterns_for({}, shifts_coverage, D, H)
                _write_partial_result(
                    job_id,
                    {},
                    pat_small,
                    demand_matrix,
                    meta={
                        "iteration": iteration + 1,
                        "factor": factor,
                        "day_labels": [f"Día {i+1}" for i in range(D)],
                        "hour_labels": list(range(H)),
                    },
                )
        except Exception:
            pass

        # Configuración temporal
        temp_cfg = cfg.copy()
        temp_cfg["agent_limit_factor"] = factor
        temp_cfg["solver_time"] = 45  # 45s por iteración

        try:
            assignments, method = optimize_with_pulp(shifts_coverage, demand_matrix, cfg=temp_cfg, job_id=job_id)
            results = analyze_results(assignments, shifts_coverage, demand_matrix) if assignments else None

            # Guardar snapshot con resultados si existen
            try:
                if job_id is not None:
                    D, H = demand_matrix.shape
                    pat_small = _compact_patterns_for(assignments if results else {}, shifts_coverage, D, H)
                    _write_partial_result(
                        job_id,
                        assignments if results else {},
                        pat_small,
                        demand_matrix,
                        meta={
                            "iteration": iteration + 1,
                            "factor": factor,
                            "day_labels": [f"Día {i+1}" for i in range(D)],
                            "hour_labels": list(range(H)),
                        },
                    )
            except Exception:
                pass

            if results:
                cov = results["coverage_percentage"]
                score = results["overstaffing"] + results["understaffing"]
                print(f"[JEAN] Factor {factor}: cobertura {cov:.1f}%, score {score:.1f}")

                if cov >= target_coverage:
                    if score < best_score or not best_assignments:
                        best_assignments, best_method = assignments, f"JEAN_F{factor}"
                        best_score = score
                        best_coverage = cov
                        print(f"[JEAN] Nueva mejor solución: score {score:.1f}")
                elif cov > best_coverage:
                    best_assignments, best_method, best_coverage = assignments, f"JEAN_F{factor}", cov
                    best_score = score
                    print(f"[JEAN] Mejor cobertura parcial: {cov:.1f}%")
        except Exception as e:
            print(f"[JEAN] Error en iteración {iteration + 1}: {e}")
            continue
    
    elapsed = time.time() - start_time
    print(f"[JEAN] Completado en {elapsed:.1f}s: score {best_score:.1f}, cobertura {best_coverage:.1f}%")
    
    return best_assignments, best_method or "JEAN_NO_SOLUTION"
