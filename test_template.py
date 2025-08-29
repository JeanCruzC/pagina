#!/usr/bin/env python3
"""Test del template con datos reales"""
import json
import tempfile
import os
from website import create_app
from flask import render_template_string

# Leer datos reales
result_file = os.path.join(tempfile.gettempdir(), "scheduler_result_6e947d47-5590-477c-b707-c836a8d00cbc.json")

if os.path.exists(result_file):
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    app = create_app()
    
    with app.app_context():
        # Test simple del template
        template = """
        PuLP assignments: {{ resultado.pulp_results.get('assignments') | length if resultado.pulp_results.get('assignments') else 'None' }}
        Greedy assignments: {{ resultado.greedy_results.get('assignments') | length if resultado.greedy_results.get('assignments') else 'None' }}
        
        PuLP condition: {{ resultado.pulp_results and resultado.pulp_results.get('assignments') }}
        Greedy condition: {{ resultado.greedy_results and resultado.greedy_results.get('assignments') }}
        """
        
        result = render_template_string(template, resultado=data)
        print("=== TEMPLATE TEST ===")
        print(result)
        
        # Test específico de las condiciones
        print("\n=== CONDICIONES ===")
        print(f"resultado.pulp_results existe: {bool(data.get('pulp_results'))}")
        print(f"resultado.pulp_results.get('assignments') existe: {bool(data.get('pulp_results', {}).get('assignments'))}")
        print(f"Condición completa PuLP: {bool(data.get('pulp_results')) and bool(data.get('pulp_results', {}).get('assignments'))}")
        
        print(f"resultado.greedy_results existe: {bool(data.get('greedy_results'))}")
        print(f"resultado.greedy_results.get('assignments') existe: {bool(data.get('greedy_results', {}).get('assignments'))}")
        print(f"Condición completa Greedy: {bool(data.get('greedy_results')) and bool(data.get('greedy_results', {}).get('assignments'))}")

else:
    print("Archivo de resultados no encontrado")