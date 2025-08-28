#!/usr/bin/env python3
"""
Test del flujo completo: POST -> worker -> status -> results
"""

import os
import sys
import time
import requests
import pandas as pd
from io import BytesIO

# Agregar el directorio del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_complete_flow():
    """Test del flujo completo de optimización."""
    
    base_url = "http://127.0.0.1:5000"
    
    # Crear datos de prueba
    test_data = []
    for dia in range(1, 8):
        for hora in range(24):
            if 8 <= hora <= 17:
                demanda = 3 if dia <= 5 else 1
            else:
                demanda = 1
            
            test_data.append({
                'Día': dia,
                'Horario': f"{hora:02d}:00",
                'Suma de Agentes Requeridos Erlang': demanda
            })
    
    df = pd.DataFrame(test_data)
    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    
    print("[TEST] Iniciando test del flujo completo...")
    
    try:
        # 1. POST al generador
        print("[TEST] 1. Enviando archivo Excel...")
        files = {'excel': ('test.xlsx', excel_buffer.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
        data = {
            'optimization_profile': 'Equilibrado (Recomendado)',
            'solver_time': '30',
            'use_ft': 'true',
            'use_pt': 'true'
        }
        
        response = requests.post(f"{base_url}/generador", files=files, data=data)
        
        if response.status_code != 202:
            print(f"[TEST] ERROR: POST failed with status {response.status_code}")
            print(f"[TEST] Response: {response.text}")
            return False
        
        job_data = response.json()
        job_id = job_data.get('job_id')
        
        if not job_id:
            print("[TEST] ERROR: No job_id in response")
            return False
        
        print(f"[TEST] Job creado: {job_id}")
        
        # 2. Polling del status
        print("[TEST] 2. Polling status...")
        max_attempts = 60  # 60 segundos máximo
        
        for attempt in range(max_attempts):
            time.sleep(1)
            
            status_response = requests.get(f"{base_url}/generador/status/{job_id}")
            
            if status_response.status_code != 200:
                print(f"[TEST] ERROR: Status check failed with {status_response.status_code}")
                continue
            
            status_data = status_response.json()
            status = status_data.get('status')
            
            print(f"[TEST] Attempt {attempt + 1}: status = {status}")
            
            if status == 'finished':
                redirect_url = status_data.get('redirect')
                print(f"[TEST] ✅ Job finished! Redirect: {redirect_url}")
                
                # 3. Verificar resultados
                print("[TEST] 3. Verificando resultados...")
                results_response = requests.get(f"{base_url}{redirect_url}")
                
                if results_response.status_code == 200:
                    print("[TEST] ✅ Resultados accesibles")
                    return True
                else:
                    print(f"[TEST] ERROR: Results page failed with {results_response.status_code}")
                    return False
            
            elif status == 'error':
                error_msg = status_data.get('error', 'Unknown error')
                print(f"[TEST] ❌ Job failed with error: {error_msg}")
                return False
            
            elif status in ['running', 'unknown']:
                continue  # Seguir esperando
            
            else:
                print(f"[TEST] ERROR: Unexpected status: {status}")
                return False
        
        print("[TEST] ❌ Timeout waiting for job completion")
        return False
        
    except Exception as e:
        print(f"[TEST] ERROR: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("[TEST] Asegúrate de que el servidor esté corriendo en http://127.0.0.1:5000")
    print("[TEST] Ejecuta: python run.py")
    input("[TEST] Presiona Enter cuando el servidor esté listo...")
    
    success = test_complete_flow()
    
    if success:
        print("[TEST] 🎉 Test completado exitosamente!")
    else:
        print("[TEST] ❌ Test falló")
    
    sys.exit(0 if success else 1)