#!/usr/bin/env python3
"""
Test simple del scheduler sin emojis para evitar problemas de codificación.
"""

import os
import sys
import tempfile
from io import BytesIO

# Agregar el directorio del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from website import create_app
from website import scheduler as engine

def test_basic_scheduler():
    """Test básico del scheduler."""
    
    # Crear app de prueba
    app = create_app()
    
    with app.app_context():
        # Crear archivo Excel de prueba simple
        import pandas as pd
        
        # Datos de prueba: demanda simple para 7 días, 24 horas
        test_data = []
        for dia in range(1, 8):  # Días 1-7
            for hora in range(24):  # Horas 0-23
                # Demanda simple: más alta durante horas laborales
                if 8 <= hora <= 17:
                    demanda = 3 if dia <= 5 else 1  # Más demanda en días laborales
                else:
                    demanda = 1
                
                test_data.append({
                    'Día': dia,
                    'Horario': f"{hora:02d}:00",
                    'Suma de Agentes Requeridos Erlang': demanda
                })
        
        df = pd.DataFrame(test_data)
        
        # Guardar en BytesIO para simular archivo subido
        excel_buffer = BytesIO()
        df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)
        
        # Configuración de prueba
        config = {
            'optimization_profile': 'Equilibrado (Recomendado)',
            'solver_time': 30,  # Tiempo corto para prueba
            'use_ft': True,
            'use_pt': True,
            'allow_8h': True,
            'allow_pt_4h': True,
            'allow_pt_6h': True
        }
        
        print("[TEST] Iniciando test de optimización...")
        print(f"[TEST] Datos de prueba: {len(test_data)} registros")
        print(f"[TEST] Perfil: {config['optimization_profile']}")
        
        try:
            # Llamar a la función principal
            result, excel_bytes, csv_bytes = engine.run_complete_optimization(
                excel_buffer, config=config, generate_charts=False
            )
            
            print("[TEST] Optimización completada!")
            print(f"[TEST] Status: {result.get('status', 'UNKNOWN')}")
            
            if 'assignments' in result:
                total_agents = sum(result['assignments'].values())
                print(f"[TEST] Total agentes: {total_agents}")
                print(f"[TEST] Turnos asignados: {len(result['assignments'])}")
            
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"[TEST] Cobertura: {metrics.get('coverage_percentage', 0):.1f}%")
                print(f"[TEST] Exceso: {metrics.get('overstaffing', 0)}")
                print(f"[TEST] Déficit: {metrics.get('understaffing', 0)}")
            
            if excel_bytes:
                print(f"[TEST] Excel generado: {len(excel_bytes)} bytes")
            
            if csv_bytes:
                print(f"[TEST] CSV generado: {len(csv_bytes)} bytes")
            
            print("[TEST] Test completado exitosamente!")
            return True
            
        except Exception as e:
            print(f"[TEST] Error en test: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_basic_scheduler()
    sys.exit(0 if success else 1)