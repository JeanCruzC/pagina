#!/usr/bin/env python3
"""
Test del sistema paralelo de optimización
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from website.parallel_optimizer import run_parallel_optimization
from io import BytesIO
import pandas as pd
import numpy as np

def create_test_excel():
    """Crear archivo Excel de prueba."""
    # Crear datos de demanda de prueba
    data = []
    for day in range(1, 8):  # Días 1-7
        for hour in range(24):  # Horas 0-23
            demand = max(0, 10 + 5 * np.sin(hour * np.pi / 12) + np.random.normal(0, 2))
            if day in [6, 7]:  # Fin de semana con menos demanda
                demand *= 0.7
            data.append({
                'Día': day,
                'Horario': f"{hour:02d}:00",
                'Suma de Agentes Requeridos Erlang': max(1, int(demand))
            })
    
    df = pd.DataFrame(data)
    
    # Guardar en BytesIO
    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    
    return excel_buffer

def test_parallel_optimization():
    """Probar optimización paralela."""
    print("=== TEST OPTIMIZACIÓN PARALELA ===")
    
    # Crear archivo de prueba
    print("Creando archivo Excel de prueba...")
    excel_file = create_test_excel()
    
    # Configuración de prueba
    config = {
        "optimization_profile": "Equilibrado (Recomendado)",
        "use_ft": True,
        "use_pt": True,
        "solver_time": 60,
        "export_files": False
    }
    
    print("Iniciando optimización paralela...")
    print(f"Configuración: {config}")
    
    try:
        result, excel_bytes, csv_bytes = run_parallel_optimization(
            excel_file, 
            config=config, 
            generate_charts=False, 
            job_id="test_001"
        )
        
        print("\n=== RESULTADOS ===")
        print(f"Status: {result.get('status', 'UNKNOWN')}")
        
        # Resultados PuLP
        pulp_results = result.get('pulp_results', {})
        print(f"\nPuLP Status: {pulp_results.get('status', 'NO_EJECUTADO')}")
        pulp_assignments = pulp_results.get('assignments', {})
        pulp_agents = sum(pulp_assignments.values()) if pulp_assignments else 0
        print(f"PuLP Agentes: {pulp_agents}")
        
        # Resultados Greedy
        greedy_results = result.get('greedy_results', {})
        print(f"\nGreedy Status: {greedy_results.get('status', 'NO_EJECUTADO')}")
        greedy_assignments = greedy_results.get('assignments', {})
        greedy_agents = sum(greedy_assignments.values()) if greedy_assignments else 0
        print(f"Greedy Agentes: {greedy_agents}")
        
        # Métricas principales
        metrics = result.get('metrics')
        if metrics:
            print(f"\nCobertura: {metrics.get('coverage_percentage', 0):.1f}%")
            print(f"Total Agentes: {metrics.get('total_agents', 0)}")
            print(f"Exceso: {metrics.get('overstaffing', 0)}")
            print(f"Déficit: {metrics.get('understaffing', 0)}")
        
        print("\n✅ Test completado exitosamente")
        return True
        
    except Exception as e:
        print(f"\n❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_parallel_optimization()
    sys.exit(0 if success else 1)