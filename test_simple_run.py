#!/usr/bin/env python3
import os
import sys
import tempfile
import json

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from website import create_app

def test_simple_optimization():
    """Test básico de optimización sin Flask context."""
    
    # Crear datos de prueba
    import pandas as pd
    import numpy as np
    
    # Crear demanda simple
    demand_data = []
    for day in range(1, 8):  # 7 días
        for hour in range(24):  # 24 horas
            # Demanda más alta en horas laborales
            if 8 <= hour <= 17:
                demand = 15 + np.random.randint(0, 10)
            else:
                demand = 5 + np.random.randint(0, 5)
            
            demand_data.append({
                'Día': day,
                'Horario': hour,
                'Suma de Agentes Requeridos Erlang': demand
            })
    
    df = pd.DataFrame(demand_data)
    
    # Guardar en archivo temporal
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        df.to_excel(tmp.name, index=False)
        excel_path = tmp.name
    
    print(f"Archivo de prueba creado: {excel_path}")
    
    # Probar optimización directa
    try:
        from website.parallel_optimizer import run_parallel_optimization
        from io import BytesIO
        
        with open(excel_path, 'rb') as f:
            file_bytes = f.read()
        
        config = {
            'optimization_profile': 'Equilibrado (Recomendado)',
            'solver_time': 30,
            'use_pulp': True,
            'use_greedy': True
        }
        
        print("Iniciando optimización de prueba...")
        result, excel_bytes, csv_bytes = run_parallel_optimization(
            BytesIO(file_bytes), 
            config=config, 
            generate_charts=False,
            job_id="test_job"
        )
        
        print("Optimización completada!")
        print(f"Status: {result.get('status', 'Unknown')}")
        print(f"Agentes totales: {result.get('metrics', {}).get('total_agents', 0)}")
        
        # Guardar resultado para inspección
        result_path = os.path.join(tempfile.gettempdir(), "test_result.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"Resultado guardado en: {result_path}")
        
        return True
        
    except Exception as e:
        print(f"Error en optimización: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Limpiar archivo temporal
        try:
            os.unlink(excel_path)
        except:
            pass

if __name__ == "__main__":
    success = test_simple_optimization()
    print(f"Test {'EXITOSO' if success else 'FALLIDO'}")