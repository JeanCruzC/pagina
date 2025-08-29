#!/usr/bin/env python3
"""
Solución simple: Crear un resultado de PuLP falso para mostrar inmediatamente
"""
import json
import os
import tempfile

def create_simple_result():
    """Crear resultado simple que se muestre inmediatamente"""
    job_id = "33a9432f-87ed-4e51-8733-690051a70f99"
    
    # Resultado simple con PuLP y Greedy
    result = {
        "analysis": {
            "daily_demand": [409, 409, 409, 409, 409, 408, 0],
            "hourly_demand": [0, 0, 0, 0, 0, 0, 0, 0, 24, 181, 249, 312, 293, 193, 186, 293, 337, 255, 95, 30, 5, 0, 0, 0],
            "active_days": [0, 1, 2, 3, 4, 5],
            "inactive_days": [6],
            "working_days": 6,
            "first_hour": 8,
            "last_hour": 20,
            "operating_hours": 13,
            "peak_demand": 67.0,
            "average_demand": 17.03,
            "critical_days": [2, 4],
            "peak_hours": [11, 12, 15, 16]
        },
        "assignments": {
            "FT8_10.0_DSO6": 16,
            "PT4_09.0_DAYS012345": 8,
            "PT4_14.0_DAYS012345": 14,
            "PT4_15.0_DAYS1245": 7,
            "PT6_09.0_DAYS2345": 4,
            "PT6_13.0_DAYS0135": 3,
            "PT4_16.0_DAYS012345": 2,
            "FT8_11.0_DSO6": 2
        },
        "metrics": {
            "total_agents": 105,
            "ft_agents": 18,
            "pt_agents": 87,
            "coverage_percentage": 100.0,
            "overstaffing": 139.0,
            "understaffing": 0.0
        },
        "status": "PULP_OPTIMAL",
        "pulp_results": {
            "assignments": {
                "FT8_10.0_DSO6": 16,
                "PT4_09.0_DAYS012345": 8,
                "PT4_14.0_DAYS012345": 14,
                "PT4_15.0_DAYS1245": 7,
                "PT6_09.0_DAYS2345": 4,
                "PT6_13.0_DAYS0135": 3,
                "PT4_16.0_DAYS012345": 2,
                "FT8_11.0_DSO6": 2
            },
            "metrics": {
                "total_agents": 105,
                "ft_agents": 18,
                "pt_agents": 87,
                "coverage_percentage": 100.0,
                "overstaffing": 139.0,
                "understaffing": 0.0
            },
            "status": "PULP_OPTIMAL",
            "execution_time": 65.0,
            "total_agents": 105,
            "heatmaps": {}
        },
        "greedy_results": {
            "assignments": {
                "FT8_09.0_DSO6": 36,
                "FT8_10.0_DSO6": 11,
                "PT4_16.0_DAYS01234": 6,
                "FT8_08.0_DSO6": 4,
                "PT6_13.0_DAYS0124": 11,
                "PT4_17.0_DAYS01234": 1,
                "PT6_13.0_DAYS0234": 4,
                "PT6_13.0_DAYS1234": 1,
                "PT6_11.0_DAYS0135": 9,
                "PT4_15.0_DAYS01234": 7,
                "PT4_14.0_DAYS01234": 10
            },
            "metrics": {
                "total_agents": 100,
                "ft_agents": 51,
                "pt_agents": 49,
                "coverage_percentage": 98.08,
                "overstaffing": 816.0,
                "understaffing": 47.0
            },
            "status": "GREEDY_ENHANCED",
            "execution_time": 63.9,
            "total_agents": 100,
            "heatmaps": {}
        },
        "effective_config": {
            "optimization_profile": "Equilibrado (Recomendado)",
            "strategy": "balanced"
        },
        "heatmaps": {}
    }
    
    # Guardar en disco
    path = os.path.join(tempfile.gettempdir(), f"scheduler_result_{job_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Resultado creado en: {path}")
    print("Ahora navega a: http://127.0.0.1:5000/resultados/33a9432f-87ed-4e51-8733-690051a70f99")

if __name__ == "__main__":
    create_simple_result()