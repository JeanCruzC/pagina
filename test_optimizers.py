#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test rápido de optimizadores
"""
import numpy as np
from website.optimizer_pulp import optimize_with_pulp
from website.optimizer_greedy import optimize_with_greedy

def test_optimizers():
    """Test básico de ambos optimizadores."""
    print("=== Test de Optimizadores ===")
    
    # Crear demanda simple
    demand = np.ones((7, 24)) * 2  # 2 agentes por hora
    
    # Crear algunos turnos de prueba
    shifts = {}
    
    # Turno de 8 horas, lunes a viernes
    pattern_8h = np.zeros(168)  # 7 días * 24 horas
    for day in range(5):  # Lunes a viernes
        for hour in range(8, 16):  # 8am a 4pm
            pattern_8h[day * 24 + hour] = 1
    shifts['FT_8h_M-F'] = pattern_8h
    
    # Turno de 6 horas, fines de semana
    pattern_6h = np.zeros(168)
    for day in [5, 6]:  # Sábado y domingo
        for hour in range(10, 16):  # 10am a 4pm
            pattern_6h[day * 24 + hour] = 1
    shifts['PT_6h_Weekend'] = pattern_6h
    
    print(f"Demanda total: {demand.sum()}")
    print(f"Turnos disponibles: {len(shifts)}")
    
    # Test PuLP
    print("\n--- Test PuLP ---")
    try:
        pulp_result, pulp_method = optimize_with_pulp(shifts, demand)
        print(f"PuLP Result: {pulp_method}")
        print(f"Assignments: {pulp_result}")
        print(f"Total agents: {sum(pulp_result.values()) if pulp_result else 0}")
    except Exception as e:
        print(f"PuLP Error: {e}")
    
    # Test Greedy
    print("\n--- Test Greedy ---")
    try:
        greedy_result, greedy_method = optimize_with_greedy(shifts, demand)
        print(f"Greedy Result: {greedy_method}")
        print(f"Assignments: {greedy_result}")
        print(f"Total agents: {sum(greedy_result.values()) if greedy_result else 0}")
    except Exception as e:
        print(f"Greedy Error: {e}")
    
    print("\n=== Test Completado ===")

if __name__ == "__main__":
    test_optimizers()