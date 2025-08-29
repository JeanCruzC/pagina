#!/usr/bin/env python3
"""Test rápido para ver los resultados"""
import json
import tempfile
import os

# Leer el archivo de resultados más reciente
result_file = os.path.join(tempfile.gettempdir(), "scheduler_result_6e947d47-5590-477c-b707-c836a8d00cbc.json")

if os.path.exists(result_file):
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=== ESTRUCTURA DE DATOS ===")
    print(f"Keys principales: {list(data.keys())}")
    
    if 'pulp_results' in data:
        print(f"\nPuLP results keys: {list(data['pulp_results'].keys())}")
        print(f"PuLP assignments existe: {'assignments' in data['pulp_results']}")
        if 'assignments' in data['pulp_results']:
            print(f"PuLP assignments count: {len(data['pulp_results']['assignments'])}")
            print(f"PuLP total agents: {data['pulp_results'].get('total_agents', 'N/A')}")
    
    if 'greedy_results' in data:
        print(f"\nGreedy results keys: {list(data['greedy_results'].keys())}")
        print(f"Greedy assignments existe: {'assignments' in data['greedy_results']}")
        if 'assignments' in data['greedy_results']:
            print(f"Greedy assignments count: {len(data['greedy_results']['assignments'])}")
            print(f"Greedy total agents: {data['greedy_results'].get('total_agents', 'N/A')}")
    
    print("\n=== RESUMEN ===")
    print(f"PuLP: {data['pulp_results']['metrics']['total_agents']} agentes, {data['pulp_results']['metrics']['coverage_percentage']:.1f}% cobertura")
    print(f"Greedy: {data['greedy_results']['metrics']['total_agents']} agentes, {data['greedy_results']['metrics']['coverage_percentage']:.1f}% cobertura")
else:
    print("Archivo de resultados no encontrado")