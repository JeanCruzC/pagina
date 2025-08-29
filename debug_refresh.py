#!/usr/bin/env python3
"""
Debug del endpoint refresh para ver por qué no detecta los resultados de PuLP
"""
import json
import os
import tempfile

def debug_refresh():
    """Debug del refresh endpoint"""
    job_id = "8bcb1fd0-ad9b-4ce8-a615-3f857c67ca9f"
    path = os.path.join(tempfile.gettempdir(), f"scheduler_result_{job_id}.json")
    
    print(f"[DEBUG] Verificando archivo: {path}")
    print(f"[DEBUG] Existe: {os.path.exists(path)}")
    
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            disk_result = json.load(f)
        
        print(f"[DEBUG] Keys principales: {list(disk_result.keys())}")
        
        # Verificar PuLP
        pulp_results = disk_result.get("pulp_results", {})
        print(f"[DEBUG] pulp_results keys: {list(pulp_results.keys())}")
        pulp_assignments = pulp_results.get("assignments", {})
        print(f"[DEBUG] pulp_assignments type: {type(pulp_assignments)}")
        print(f"[DEBUG] pulp_assignments length: {len(pulp_assignments) if pulp_assignments else 0}")
        print(f"[DEBUG] bool(pulp_assignments): {bool(pulp_assignments)}")
        
        # Verificar Greedy
        greedy_results = disk_result.get("greedy_results", {})
        print(f"[DEBUG] greedy_results keys: {list(greedy_results.keys())}")
        greedy_assignments = greedy_results.get("assignments", {})
        print(f"[DEBUG] greedy_assignments type: {type(greedy_assignments)}")
        print(f"[DEBUG] greedy_assignments length: {len(greedy_assignments) if greedy_assignments else 0}")
        print(f"[DEBUG] bool(greedy_assignments): {bool(greedy_assignments)}")
        
        # Simular la lógica del endpoint
        has_pulp = bool(disk_result.get("pulp_results", {}).get("assignments"))
        has_greedy = bool(disk_result.get("greedy_results", {}).get("assignments"))
        
        print(f"[DEBUG] Lógica endpoint - has_pulp: {has_pulp}, has_greedy: {has_greedy}")
        
        # Mostrar algunos assignments
        if pulp_assignments:
            print(f"[DEBUG] Primeros 3 PuLP assignments:")
            for i, (k, v) in enumerate(list(pulp_assignments.items())[:3]):
                print(f"  {k}: {v}")
        
        if greedy_assignments:
            print(f"[DEBUG] Primeros 3 Greedy assignments:")
            for i, (k, v) in enumerate(list(greedy_assignments.items())[:3]):
                print(f"  {k}: {v}")

if __name__ == "__main__":
    debug_refresh()