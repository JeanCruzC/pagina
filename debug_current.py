#!/usr/bin/env python3
"""
Debug del archivo actual para ver por qué no se detectan las asignaciones de PuLP
"""
import json
import os
import tempfile

def debug_current():
    """Debug del archivo actual"""
    job_id = "cf854eb3-e566-4042-892f-f3de42deec84"
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
        
        # Simular la lógica del endpoint EXACTA
        has_pulp = bool(disk_result.get("pulp_results", {}).get("assignments"))
        print(f"[DEBUG] Lógica endpoint - has_pulp: {has_pulp}")
        
        # Mostrar algunos assignments
        if pulp_assignments:
            print(f"[DEBUG] Primeros 3 PuLP assignments:")
            for i, (k, v) in enumerate(list(pulp_assignments.items())[:3]):
                print(f"  {k}: {v}")
        else:
            print(f"[DEBUG] ERROR: pulp_assignments está vacío o es None")
            print(f"[DEBUG] pulp_assignments = {pulp_assignments}")

if __name__ == "__main__":
    debug_current()