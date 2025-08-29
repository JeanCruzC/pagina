#!/usr/bin/env python3
"""
Test directo de PuLP para verificar que funciona correctamente
"""
import pandas as pd
import tempfile
import json
import os
from website.scheduler_core import load_demand_matrix_from_df, generate_shifts_coverage
from website.optimizer_pulp import optimize_with_pulp
from website.profiles import apply_profile

def test_pulp_direct():
    """Test directo de PuLP sin Flask"""
    print("[TEST] Iniciando test directo de PuLP")
    
    # Cargar archivo real de demanda
    excel_path = "legacy/Requerido.xlsx"
    if not os.path.exists(excel_path):
        print(f"[TEST] ERROR: No se encuentra {excel_path}")
        return False
    
    df = pd.read_excel(excel_path)
    print(f"[TEST] Cargado {excel_path} con {len(df)} filas")
    print(f"[TEST] Columnas: {list(df.columns)}")
    print(f"[TEST] Primeras 5 filas:")
    print(df.head())
    
    # Cargar matriz de demanda
    demand_matrix = load_demand_matrix_from_df(df)
    print(f"[TEST] Matriz de demanda: {demand_matrix.shape}")
    print(f"[TEST] Demanda total: {demand_matrix.sum()}")
    
    # Configuración básica
    config = {
        "optimization_profile": "Equilibrado (Recomendado)",
        "solver_time": 60,  # 1 minuto máximo
        "use_pulp": True
    }
    cfg = apply_profile(config)
    print(f"[TEST] Configuración aplicada: {cfg.get('optimization_profile')}")
    
    # Generar patrones de turnos
    shifts_coverage = generate_shifts_coverage(cfg=cfg)
    print(f"[TEST] Generados {len(shifts_coverage)} patrones de turnos")
    
    # Ejecutar PuLP
    print("[TEST] Ejecutando PuLP...")
    assignments, method = optimize_with_pulp(
        shifts_coverage, 
        demand_matrix, 
        cfg=cfg
    )
    
    print(f"[TEST] PuLP completado: {method}")
    print(f"[TEST] Asignaciones: {len(assignments)} turnos")
    
    if assignments:
        total_agents = sum(assignments.values())
        print(f"[TEST] Total agentes: {total_agents}")
        
        # Mostrar algunos turnos
        sorted_assignments = sorted(assignments.items(), key=lambda x: x[1], reverse=True)
        print("[TEST] Top 5 turnos:")
        for shift, count in sorted_assignments[:5]:
            print(f"  {shift}: {count} agentes")
        
        # Guardar resultado en archivo temporal
        result = {
            "assignments": assignments,
            "method": method,
            "total_agents": total_agents,
            "shifts_count": len(assignments)
        }
        
        temp_path = os.path.join(tempfile.gettempdir(), "test_pulp_result.json")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"[TEST] Resultado guardado en: {temp_path}")
        return True
    else:
        print("[TEST] ERROR: No se generaron asignaciones")
        return False

if __name__ == "__main__":
    success = test_pulp_direct()
    if success:
        print("[TEST] ✅ PuLP funciona correctamente")
    else:
        print("[TEST] ❌ PuLP falló")