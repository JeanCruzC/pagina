#!/usr/bin/env python3
"""
Script para forzar la carga de resultados en el store de Flask
"""
import sys
import os
import tempfile
import json

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from website import create_app

def force_load_results():
    """Cargar resultados desde archivos temporales al store de Flask."""
    
    app = create_app()
    
    with app.app_context():
        from website.extensions import scheduler as store
        
        temp_dir = tempfile.gettempdir()
        loaded_count = 0
        
        for filename in os.listdir(temp_dir):
            if filename.startswith('scheduler_result_') and filename.endswith('.json'):
                job_id = filename.replace('scheduler_result_', '').replace('.json', '')
                filepath = os.path.join(temp_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    
                    # Verificar que tiene resultados válidos
                    has_results = bool(
                        result_data.get('assignments') or 
                        result_data.get('greedy_results', {}).get('assignments') or 
                        result_data.get('pulp_results', {}).get('assignments')
                    )
                    
                    if has_results:
                        # Forzar carga en el store
                        store.mark_finished(job_id, result_data, None, None)
                        print(f"✓ Cargado job {job_id}")
                        loaded_count += 1
                    else:
                        print(f"✗ Job {job_id} sin resultados válidos")
                        
                except Exception as e:
                    print(f"✗ Error cargando {job_id}: {e}")
        
        print(f"\nTotal cargados: {loaded_count}")
        
        # Verificar el store
        print("\nJobs en store:")
        try:
            s = store._s()
            for job_id, job_info in s.get('jobs', {}).items():
                status = job_info.get('status', 'unknown')
                print(f"  {job_id}: {status}")
        except Exception as e:
            print(f"Error verificando store: {e}")

if __name__ == "__main__":
    force_load_results()