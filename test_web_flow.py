#!/usr/bin/env python3
"""
Test del flujo completo de la aplicación web para verificar dónde se pierden los resultados de PuLP
"""
import os
import tempfile
import json
import time
from io import BytesIO

def test_web_flow():
    """Test del flujo completo web"""
    print("[WEB_TEST] Iniciando test del flujo web completo")
    
    # Simular el flujo completo
    from website import create_app
    from website.parallel_optimizer import run_parallel_optimization
    
    app = create_app()
    
    with app.app_context():
        print("[WEB_TEST] Contexto Flask creado")
        
        # Cargar archivo real
        excel_path = "legacy/Requerido.xlsx"
        if not os.path.exists(excel_path):
            print(f"[WEB_TEST] ERROR: No se encuentra {excel_path}")
            return False
        
        with open(excel_path, "rb") as f:
            file_bytes = f.read()
        
        print(f"[WEB_TEST] Archivo cargado: {len(file_bytes)} bytes")
        
        # Configuración de prueba
        config = {
            "optimization_profile": "Equilibrado (Recomendado)",
            "solver_time": 60,
            "use_pulp": True,
            "use_greedy": True
        }
        
        job_id = "test_web_flow_123"
        
        print("[WEB_TEST] Ejecutando optimización paralela...")
        
        # Ejecutar optimización
        result, excel_bytes, csv_bytes = run_parallel_optimization(
            BytesIO(file_bytes),
            config=config,
            generate_charts=False,
            job_id=job_id
        )
        
        print(f"[WEB_TEST] Optimización completada")
        print(f"[WEB_TEST] Resultado keys: {list(result.keys()) if result else 'None'}")
        
        if result:
            print(f"[WEB_TEST] Status: {result.get('status')}")
            print(f"[WEB_TEST] Assignments: {len(result.get('assignments', {}))}")
            
            # Verificar resultados de PuLP
            pulp_results = result.get('pulp_results', {})
            print(f"[WEB_TEST] PuLP results keys: {list(pulp_results.keys())}")
            print(f"[WEB_TEST] PuLP assignments: {len(pulp_results.get('assignments', {}))}")
            print(f"[WEB_TEST] PuLP status: {pulp_results.get('status')}")
            
            # Verificar resultados de Greedy
            greedy_results = result.get('greedy_results', {})
            print(f"[WEB_TEST] Greedy results keys: {list(greedy_results.keys())}")
            print(f"[WEB_TEST] Greedy assignments: {len(greedy_results.get('assignments', {}))}")
            print(f"[WEB_TEST] Greedy status: {greedy_results.get('status')}")
            
            # Verificar archivo en disco
            temp_path = os.path.join(tempfile.gettempdir(), f"scheduler_result_{job_id}.json")
            if os.path.exists(temp_path):
                print(f"[WEB_TEST] Archivo en disco encontrado: {temp_path}")
                with open(temp_path, "r", encoding="utf-8") as f:
                    disk_result = json.load(f)
                print(f"[WEB_TEST] Disco - PuLP assignments: {len(disk_result.get('pulp_results', {}).get('assignments', {}))}")
                print(f"[WEB_TEST] Disco - Greedy assignments: {len(disk_result.get('greedy_results', {}).get('assignments', {}))}")
            else:
                print(f"[WEB_TEST] ERROR: No se encontró archivo en disco")
            
            # Verificar store
            try:
                from website.extensions import scheduler as store
                payload = store.get_payload(job_id)
                if payload:
                    print(f"[WEB_TEST] Store payload encontrado")
                    store_result = payload.get('result', {})
                    print(f"[WEB_TEST] Store - PuLP assignments: {len(store_result.get('pulp_results', {}).get('assignments', {}))}")
                    print(f"[WEB_TEST] Store - Greedy assignments: {len(store_result.get('greedy_results', {}).get('assignments', {}))}")
                else:
                    print(f"[WEB_TEST] ERROR: No se encontró payload en store")
            except Exception as e:
                print(f"[WEB_TEST] ERROR accediendo store: {e}")
            
            return True
        else:
            print("[WEB_TEST] ERROR: No se obtuvo resultado")
            return False

if __name__ == "__main__":
    success = test_web_flow()
    if success:
        print("[WEB_TEST] Test completado exitosamente")
    else:
        print("[WEB_TEST] Test falló")