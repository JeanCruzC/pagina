# -*- coding: utf-8 -*-
"""
Fallback rápido para cuando el solver se cuelga.
Usa solo greedy con timeouts muy agresivos.
"""
import time
import threading
from .scheduler import (
    optimize_schedule_greedy_enhanced,
    analyze_results,
    merge_config,
    generate_shifts_coverage_optimized
)

def quick_optimization_fallback(file_stream, config=None, job_id=None):
    """Optimización rápida usando solo greedy con timeouts agresivos."""
    print("[FALLBACK] Iniciando optimización rápida")
    
    try:
        from .scheduler import (
            apply_configuration,
            load_demand_matrix_from_df,
            analyze_demand_matrix,
            monitor_memory_usage
        )
        import pandas as pd
        import numpy as np
        
        cfg = apply_configuration(config)
        
        # Leer demanda
        df = pd.read_excel(file_stream)
        demand_matrix = load_demand_matrix_from_df(df)
        analysis = analyze_demand_matrix(demand_matrix)
        
        # Generar patrones con límites muy restrictivos
        patterns = {}
        cfg["max_patterns"] = 1000  # Máximo 1000 patrones
        cfg["batch_size"] = 500     # Batches pequeños
        
        for batch in generate_shifts_coverage_optimized(
            demand_matrix,
            max_patterns=1000,
            batch_size=500,
            cfg=cfg,
        ):
            patterns.update(batch)
            if len(patterns) >= 1000:
                break
        
        print(f"[FALLBACK] Generados {len(patterns)} patrones")
        
        # Solo usar greedy
        assignments, status = optimize_schedule_greedy_enhanced(
            patterns, demand_matrix, cfg=cfg, job_id=job_id
        )
        
        # Calcular métricas
        metrics = analyze_results(assignments, patterns, demand_matrix)
        
        result = {
            "analysis": analysis,
            "assignments": assignments,
            "metrics": metrics,
            "status": f"FALLBACK_{status}",
            "pulp_results": {
                "assignments": assignments,
                "metrics": metrics,
                "status": f"FALLBACK_{status}",
                "heatmaps": {}
            },
            "greedy_results": {
                "assignments": assignments,
                "metrics": metrics,
                "status": status,
                "heatmaps": {}
            },
            "effective_config": cfg,
            "message": "Optimización rápida completada (fallback mode)"
        }
        
        print(f"[FALLBACK] Completado: {len(assignments)} turnos")
        return result, None, None
        
    except Exception as e:
        print(f"[FALLBACK] Error: {e}")
        return {"error": str(e)}, None, None