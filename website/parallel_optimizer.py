# -*- coding: utf-8 -*-
"""
Optimizador Paralelo - Ejecuta PuLP y Greedy independientemente en paralelo
"""
import threading
import time
import tempfile
import json
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .scheduler_core import (
    load_demand_matrix_from_df, 
    analyze_demand_matrix,
    generate_shifts_coverage,
    analyze_results,
    generate_all_heatmaps,
    export_detailed_schedule
)
from .optimizer_pulp import optimize_with_pulp, optimize_jean_search
from .optimizer_greedy import optimize_with_greedy, optimize_greedy_fast

def update_job_progress(job_id, progress_info):
    """Actualizar progreso del job."""
    if job_id:
        try:
            from .extensions import scheduler as store
            store.update_progress(job_id, progress_info)
        except Exception as e:
            # Silenciar errores de contexto, solo imprimir para debug
            pass

def run_pulp_optimization(shifts_coverage, demand_matrix, cfg, job_id=None, app_context=None):
    """Ejecutar optimización PuLP independiente."""
    print("[PARALLEL] Iniciando PuLP")
    start_time = time.time()
    
    try:
        update_job_progress(job_id, {"pulp_status": "RUNNING"})
        
        # Determinar método según perfil
        profile = cfg.get("optimization_profile", "Equilibrado (Recomendado)")
        
        if profile == "JEAN":
            assignments, method = optimize_jean_search(
                shifts_coverage, demand_matrix, 
                cfg=cfg, 
                target_coverage=cfg.get("TARGET_COVERAGE", 98.0),
                job_id=job_id
            )
        elif profile == "JEAN Personalizado":
            # Para JEAN Personalizado, usar búsqueda JEAN también
            assignments, method = optimize_jean_search(
                shifts_coverage, demand_matrix, 
                cfg=cfg, 
                target_coverage=cfg.get("TARGET_COVERAGE", 98.0),
                job_id=job_id
            )
        else:
            # Perfiles estándar usan PuLP directo
            assignments, method = optimize_with_pulp(
                shifts_coverage, demand_matrix, 
                cfg=cfg, 
                job_id=job_id
            )
        
        # Analizar resultados
        metrics = analyze_results(assignments, shifts_coverage, demand_matrix) if assignments else None
        
        elapsed = time.time() - start_time
        total_agents = sum(assignments.values()) if assignments else 0
        
        result = {
            "assignments": assignments,
            "metrics": metrics,
            "status": method,
            "execution_time": elapsed,
            "total_agents": total_agents,
            "heatmaps": {}
        }
        
        print(f"[PULP] Completado en {elapsed:.1f}s: {total_agents} agentes")
        update_job_progress(job_id, {"pulp_status": "COMPLETED", "pulp_agents": total_agents})
        
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[PULP] Error en {elapsed:.1f}s: {e}")
        update_job_progress(job_id, {"pulp_status": "ERROR"})
        
        return {
            "assignments": {},
            "metrics": None,
            "status": f"PULP_ERROR: {str(e)[:50]}",
            "execution_time": elapsed,
            "total_agents": 0,
            "heatmaps": {}
        }

def run_greedy_optimization(shifts_coverage, demand_matrix, cfg, job_id=None, app_context=None):
    """Ejecutar optimización Greedy independiente."""
    print("[PARALLEL] Iniciando Greedy")
    start_time = time.time()
    
    try:
        update_job_progress(job_id, {"greedy_status": "RUNNING"})
        
        assignments, method = optimize_with_greedy(
            shifts_coverage, demand_matrix, 
            cfg=cfg, 
            job_id=job_id
        )
        
        # Analizar resultados
        metrics = analyze_results(assignments, shifts_coverage, demand_matrix) if assignments else None
        
        elapsed = time.time() - start_time
        total_agents = sum(assignments.values()) if assignments else 0
        
        result = {
            "assignments": assignments,
            "metrics": metrics,
            "status": method,
            "execution_time": elapsed,
            "total_agents": total_agents,
            "heatmaps": {}
        }
        
        print(f"[GREEDY] Completado en {elapsed:.1f}s: {total_agents} agentes")
        update_job_progress(job_id, {"greedy_status": "COMPLETED", "greedy_agents": total_agents})
        
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[GREEDY] Error en {elapsed:.1f}s: {e}")
        update_job_progress(job_id, {"greedy_status": "ERROR"})
        
        return {
            "assignments": {},
            "metrics": None,
            "status": f"GREEDY_ERROR: {str(e)[:50]}",
            "execution_time": elapsed,
            "total_agents": 0,
            "heatmaps": {}
        }

def save_partial_results(job_id, pulp_result=None, greedy_result=None, demand_analysis=None, cfg=None):
    """Guardar resultados parciales para que la UI los muestre."""
    if not job_id:
        return
    
    try:
        from .extensions import scheduler as store
        
        # Crear resultado parcial
        partial_result = {
            "analysis": demand_analysis or {},
            "assignments": {},
            "metrics": None,
            "status": "PARTIAL",
            "pulp_results": pulp_result or {"assignments": {}, "metrics": None, "status": "PENDING", "heatmaps": {}},
            "greedy_results": greedy_result or {"assignments": {}, "metrics": None, "status": "PENDING", "heatmaps": {}},
            "effective_config": cfg or {},
            "heatmaps": {}
        }
        
        # Usar el mejor resultado disponible como principal
        if pulp_result and pulp_result.get("assignments"):
            partial_result["assignments"] = pulp_result["assignments"]
            partial_result["metrics"] = pulp_result["metrics"]
            partial_result["status"] = f"PARTIAL_{pulp_result['status']}"
        elif greedy_result and greedy_result.get("assignments"):
            partial_result["assignments"] = greedy_result["assignments"]
            partial_result["metrics"] = greedy_result["metrics"]
            partial_result["status"] = f"PARTIAL_{greedy_result['status']}"
        
        # Marcar como terminado en el store (sin contexto Flask)
        try:
            store.mark_finished(job_id, partial_result, None, None)
        except Exception:
            pass
        
        # Guardar snapshot en disco
        try:
            path = os.path.join(tempfile.gettempdir(), f"scheduler_result_{job_id}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(partial_result, fh, ensure_ascii=False, default=str)
        except Exception:
            pass
        
        print(f"[PARTIAL] Resultados parciales guardados para job {job_id}")
        
    except Exception as e:
        print(f"[PARTIAL] Error guardando resultados: {e}")

def run_parallel_optimization(file_stream, config=None, generate_charts=False, job_id=None):
    """Ejecutar optimización paralela completa."""
    print("[PARALLEL] Iniciando optimización paralela")
    
    # Capturar el contexto de Flask para los threads
    app_context = None
    try:
        from flask import current_app
        app_context = current_app._get_current_object()
    except Exception:
        pass
    
    if job_id:
        update_job_progress(job_id, {"stage": "Iniciando optimización paralela"})
    
    try:
        # Aplicar configuración y perfil
        from .profiles import apply_profile
        cfg = apply_profile(config)
        
        # Cargar y procesar demanda
        print("[PARALLEL] Cargando demanda")
        import pandas as pd
        df = pd.read_excel(file_stream)
        demand_matrix = load_demand_matrix_from_df(df)
        analysis = analyze_demand_matrix(demand_matrix)
        
        update_job_progress(job_id, {"stage": "Generando patrones"})
        
        # Generar patrones de turnos
        print("[PARALLEL] Generando patrones")
        shifts_coverage = generate_shifts_coverage(cfg=cfg)
        print(f"[PARALLEL] Generados {len(shifts_coverage)} patrones")
        
        update_job_progress(job_id, {"stage": "Iniciando optimizadores paralelos"})
        
        # Variables para resultados
        pulp_result = None
        greedy_result = None
        
        # Ejecutar optimizadores en paralelo
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Lanzar ambos optimizadores
            pulp_future = executor.submit(run_pulp_optimization, shifts_coverage, demand_matrix, cfg, job_id, app_context)
            greedy_future = executor.submit(run_greedy_optimization, shifts_coverage, demand_matrix, cfg, job_id, app_context)
            
            # Procesar resultados conforme van completándose
            for future in as_completed([pulp_future, greedy_future], timeout=300):  # 5 minutos máximo
                try:
                    result = future.result(timeout=120)  # 2 minutos por optimizador
                    
                    if future == pulp_future:
                        pulp_result = result
                        print("[PARALLEL] PuLP completado")
                    elif future == greedy_future:
                        greedy_result = result
                        print("[PARALLEL] Greedy completado")
                    
                    # Guardar resultados parciales cada vez que uno termine
                    save_partial_results(job_id, pulp_result, greedy_result, analysis, cfg)
                    
                except Exception as e:
                    print(f"[PARALLEL] Error en optimizador: {e}")
                    if future == pulp_future:
                        pulp_result = {"assignments": {}, "metrics": None, "status": f"ERROR: {e}", "heatmaps": {}}
                    elif future == greedy_future:
                        greedy_result = {"assignments": {}, "metrics": None, "status": f"ERROR: {e}", "heatmaps": {}}
        
        # Asegurar que tenemos resultados por defecto
        if pulp_result is None:
            pulp_result = {"assignments": {}, "metrics": None, "status": "NOT_EXECUTED", "heatmaps": {}}
        if greedy_result is None:
            greedy_result = {"assignments": {}, "metrics": None, "status": "NOT_EXECUTED", "heatmaps": {}}
        
        print("[PARALLEL] Ambos optimizadores completados")
        update_job_progress(job_id, {"stage": "Procesando resultados finales"})
        
        # Determinar mejor resultado
        best_assignments = {}
        best_metrics = None
        best_status = "NO_SOLUTION"
        
        # Priorizar PuLP si tiene resultados, sino Greedy
        if pulp_result.get("assignments"):
            best_assignments = pulp_result["assignments"]
            best_metrics = pulp_result["metrics"]
            best_status = pulp_result["status"]
        elif greedy_result.get("assignments"):
            best_assignments = greedy_result["assignments"]
            best_metrics = greedy_result["metrics"]
            best_status = greedy_result["status"]
        
        # Generar heatmaps si se solicitan
        heatmaps = {}
        pulp_heatmaps = {}
        greedy_heatmaps = {}
        
        if generate_charts:
            update_job_progress(job_id, {"stage": "Generando gráficos"})
            
            # Heatmaps generales
            demand_maps = generate_all_heatmaps(demand_matrix)
            for key, fig in demand_maps.items():
                if key == "demand":
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    fig.savefig(tmp.name, format="png", bbox_inches="tight")
                    tmp.flush()
                    tmp.close()
                    filename = os.path.basename(tmp.name)
                    heatmaps[key] = f"/heatmap/{filename}"
                if hasattr(fig, 'close'):
                    fig.close()
            
            # Heatmaps específicos de PuLP
            if pulp_result.get("metrics") and pulp_result.get("assignments"):
                pulp_maps = generate_all_heatmaps(
                    demand_matrix,
                    pulp_result["metrics"].get("total_coverage"),
                    pulp_result["metrics"].get("diff_matrix")
                )
                for key, fig in pulp_maps.items():
                    if key != "demand":
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                        fig.savefig(tmp.name, format="png", bbox_inches="tight")
                        tmp.flush()
                        tmp.close()
                        filename = os.path.basename(tmp.name)
                        pulp_heatmaps[key] = f"/heatmap/{filename}"
                    if hasattr(fig, 'close'):
                        fig.close()
            
            # Heatmaps específicos de Greedy
            if greedy_result.get("metrics") and greedy_result.get("assignments"):
                greedy_maps = generate_all_heatmaps(
                    demand_matrix,
                    greedy_result["metrics"].get("total_coverage"),
                    greedy_result["metrics"].get("diff_matrix")
                )
                for key, fig in greedy_maps.items():
                    if key != "demand":
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                        fig.savefig(tmp.name, format="png", bbox_inches="tight")
                        tmp.flush()
                        tmp.close()
                        filename = os.path.basename(tmp.name)
                        greedy_heatmaps[key] = f"/heatmap/{filename}"
                    if hasattr(fig, 'close'):
                        fig.close()
        
        # Actualizar heatmaps en resultados
        pulp_result["heatmaps"] = pulp_heatmaps
        greedy_result["heatmaps"] = greedy_heatmaps
        
        # Exportar archivos si se solicita
        excel_bytes = None
        csv_bytes = None
        if cfg.get("export_files", False) and best_assignments:
            update_job_progress(job_id, {"stage": "Exportando archivos"})
            excel_bytes, csv_bytes = export_detailed_schedule(best_assignments, shifts_coverage)
        
        # Resultado final
        def _convert_numpy(obj):
            """Convertir arrays numpy a listas."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert_numpy(v) for v in obj]
            return obj
        
        final_result = {
            "analysis": _convert_numpy(analysis),
            "assignments": best_assignments,
            "metrics": _convert_numpy(best_metrics),
            "status": best_status,
            "heatmaps": heatmaps,
            "pulp_results": _convert_numpy(pulp_result),
            "greedy_results": _convert_numpy(greedy_result),
            "effective_config": _convert_numpy(cfg)
        }
        
        print("[PARALLEL] Optimización paralela completada")
        update_job_progress(job_id, {"stage": "Completado"})
        
        return final_result, excel_bytes, csv_bytes
        
    except Exception as e:
        print(f"[PARALLEL] Error crítico: {e}")
        return {"error": str(e)}, None, None