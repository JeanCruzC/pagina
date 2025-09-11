#!/usr/bin/env python3
"""
Script independiente para ejecutar Grid Search exhaustivo de hiperparámetros.
Uso: python scripts/run_grid_search.py <archivo_demanda.xlsx>
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from website.scheduler import (
    load_demand_matrix_from_df,
    generate_shifts_coverage_optimized,
    _search_hyperparameters
)

def main():
    if len(sys.argv) != 2:
        print("Uso: python scripts/run_grid_search.py <archivo_demanda.xlsx>")
        sys.exit(1)
    
    demand_file = sys.argv[1]
    
    if not os.path.exists(demand_file):
        print(f"Error: Archivo {demand_file} no encontrado")
        sys.exit(1)
    
    print(f"[Grid Search Script] Cargando demanda desde {demand_file}")
    
    # Cargar demanda
    df = pd.read_excel(demand_file)
    demand_matrix = load_demand_matrix_from_df(df)
    
    print(f"[Grid Search Script] Demanda cargada: {demand_matrix.shape} (días x horas)")
    print(f"[Grid Search Script] Demanda total: {demand_matrix.sum()} agentes-hora")
    
    # Generar patrones (configuración completa)
    cfg = {
        "use_ft": True,
        "use_pt": True,
        "allow_8h": True,
        "allow_10h8": True,
        "allow_pt_4h": True,
        "allow_pt_5h": True,
        "allow_pt_6h": True,
        "solver_time": 60,
        "random_seed": 42
    }
    
    print("[Grid Search Script] Generando patrones de turnos...")
    patterns = {}
    for batch in generate_shifts_coverage_optimized(
        demand_matrix,
        allow_8h=True,
        allow_10h8=True,
        allow_pt_4h=True,
        allow_pt_5h=True,
        allow_pt_6h=True,
        keep_percentage=1.0,
        max_patterns=None
    ):
        patterns.update(batch)
    
    print(f"[Grid Search Script] Patrones generados: {len(patterns)}")
    
    # Crear directorio de datos si no existe
    os.makedirs("data", exist_ok=True)
    
    # Ejecutar Grid Search
    print("[Grid Search Script] Iniciando búsqueda exhaustiva...")
    best_solution, best_metrics = _search_hyperparameters(patterns, demand_matrix, cfg)
    
    if best_solution:
        print("\n[Grid Search Script] ¡Búsqueda completada!")
        print(f"Mejor solver: {best_metrics['solver']}")
        print(f"Cobertura real: {best_metrics['coverage_real']:.2f}%")
        print(f"Exceso: {best_metrics['excess']}")
        print(f"Déficit: {best_metrics['deficit']}")
        print(f"Agentes totales: {best_metrics['total_agents']}")
        print(f"Score final: {best_metrics['score']:.3f}")
        print("\nResultados detallados guardados en data/grid_search_results.csv")
    else:
        print("[Grid Search Script] No se encontraron soluciones válidas")
        sys.exit(1)

if __name__ == "__main__":
    main()