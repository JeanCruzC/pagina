#!/usr/bin/env python3
"""
Test del template para verificar que recibe los datos de PuLP correctamente
"""
import json
import os
import tempfile
from website import create_app

def test_template():
    """Test del template con datos reales"""
    app = create_app()
    
    with app.app_context():
        # Cargar datos del archivo real
        job_id = "8bcb1fd0-ad9b-4ce8-a615-3f857c67ca9f"
        path = os.path.join(tempfile.gettempdir(), f"scheduler_result_{job_id}.json")
        
        if not os.path.exists(path):
            print(f"[TEST] ERROR: No se encuentra {path}")
            return False
        
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)
        
        print(f"[TEST] Datos cargados correctamente")
        print(f"[TEST] Keys: {list(result.keys())}")
        
        # Verificar PuLP
        pulp_results = result.get("pulp_results", {})
        pulp_assignments = pulp_results.get("assignments", {})
        print(f"[TEST] PuLP assignments: {len(pulp_assignments)}")
        print(f"[TEST] PuLP status: {pulp_results.get('status')}")
        
        # Verificar Greedy
        greedy_results = result.get("greedy_results", {})
        greedy_assignments = greedy_results.get("assignments", {})
        print(f"[TEST] Greedy assignments: {len(greedy_assignments)}")
        print(f"[TEST] Greedy status: {greedy_results.get('status')}")
        
        # Simular el template
        from flask import render_template_string
        
        template = """
        PuLP Results: {{ resultado.pulp_results.assignments|length if resultado.pulp_results.assignments else 0 }}
        Greedy Results: {{ resultado.greedy_results.assignments|length if resultado.greedy_results.assignments else 0 }}
        PuLP Status: {{ resultado.pulp_results.status if resultado.pulp_results else 'None' }}
        Greedy Status: {{ resultado.greedy_results.status if resultado.greedy_results else 'None' }}
        """
        
        rendered = render_template_string(template, resultado=result)
        print(f"[TEST] Template renderizado:")
        print(rendered)
        
        return True

if __name__ == "__main__":
    test_template()