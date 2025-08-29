#!/usr/bin/env python3
"""
Script para forzar la visualización de resultados
"""
import tempfile
import json
import os

def check_temp_results():
    """Buscar archivos de resultados en temp."""
    temp_dir = tempfile.gettempdir()
    result_files = []
    
    for filename in os.listdir(temp_dir):
        if filename.startswith('scheduler_result_') and filename.endswith('.json'):
            filepath = os.path.join(temp_dir, filename)
            job_id = filename.replace('scheduler_result_', '').replace('.json', '')
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                result_files.append({
                    'job_id': job_id,
                    'filepath': filepath,
                    'has_results': bool(data.get('assignments') or 
                                      data.get('greedy_results', {}).get('assignments') or 
                                      data.get('pulp_results', {}).get('assignments'))
                })
            except Exception as e:
                print(f"Error leyendo {filepath}: {e}")
    
    return result_files

def main():
    print("Buscando archivos de resultados en temp...")
    results = check_temp_results()
    
    if not results:
        print("No se encontraron archivos de resultados.")
        return
    
    print(f"Encontrados {len(results)} archivos:")
    for r in results:
        print(f"  Job ID: {r['job_id']}")
        print(f"  Archivo: {r['filepath']}")
        print(f"  Tiene resultados: {r['has_results']}")
        print(f"  URL: http://127.0.0.1:5000/resultados/{r['job_id']}")
        print()

if __name__ == "__main__":
    main()