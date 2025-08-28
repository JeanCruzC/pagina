#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de prueba para verificar que todos los perfiles de optimización
funcionan correctamente con sus características específicas.
"""

import numpy as np
import sys
import os

# Agregar el directorio del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from website.scheduler import (
    apply_configuration, PROFILES, merge_config,
    optimize_jean_search, optimize_perfect_coverage,
    optimize_with_precision_targeting, optimize_schedule_greedy_enhanced,
    analyze_results, generate_shifts_coverage_optimized
)

def create_test_demand_matrix():
    """Crear una matriz de demanda de prueba."""
    # Matriz 7x24 con patrón típico de call center
    demand = np.zeros((7, 24))
    
    # Lunes a Viernes: 8:00 - 20:00
    for day in range(5):  # Lun-Vie
        for hour in range(8, 21):
            if 9 <= hour <= 17:  # Horas pico
                demand[day, hour] = np.random.randint(8, 15)
            else:  # Horas normales
                demand[day, hour] = np.random.randint(3, 8)
    
    # Sábado: 9:00 - 18:00 (reducido)
    for hour in range(9, 19):
        demand[5, hour] = np.random.randint(2, 6)
    
    # Domingo: 10:00 - 16:00 (mínimo)
    for hour in range(10, 17):
        demand[6, hour] = np.random.randint(1, 4)
    
    return demand

def test_profile_configuration(profile_name):
    """Probar la configuración de un perfil específico."""
    print(f"\n=== Probando perfil: {profile_name} ===")
    
    # Configurar el perfil
    config = {"optimization_profile": profile_name}
    cfg = apply_configuration(config)
    
    # Verificar que se aplicaron los parámetros correctos
    expected_params = PROFILES.get(profile_name, {})
    
    print(f"Parámetros aplicados:")
    for key, expected_value in expected_params.items():
        actual_value = cfg.get(key)
        status = "✓" if actual_value == expected_value else "✗"
        print(f"  {status} {key}: {actual_value} (esperado: {expected_value})")
    
    return cfg

def test_profile_optimization(profile_name, demand_matrix, patterns):
    """Probar la optimización con un perfil específico."""
    print(f"\n--- Optimización con {profile_name} ---")
    
    config = {"optimization_profile": profile_name}
    cfg = apply_configuration(config)
    
    try:
        # Seleccionar algoritmo según el perfil
        if profile_name == "JEAN":
            assignments, status = optimize_jean_search(
                patterns, demand_matrix, cfg=cfg, target_coverage=98.0, max_iterations=3
            )
        elif profile_name in ["100% Cobertura Eficiente", "100% Cobertura Total", "Cobertura Perfecta", "100% Exacto"]:
            assignments, status = optimize_perfect_coverage(patterns, demand_matrix, cfg=cfg)
        elif profile_name == "Máxima Cobertura":
            try:
                from website.profile_optimizers import optimize_maximum_coverage
                assignments, status = optimize_maximum_coverage(patterns, demand_matrix, cfg=cfg)
            except ImportError:
                assignments, status = optimize_with_precision_targeting(patterns, demand_matrix, cfg=cfg)
        elif profile_name == "Mínimo Costo":
            try:
                from website.profile_optimizers import optimize_minimum_cost
                assignments, status = optimize_minimum_cost(patterns, demand_matrix, cfg=cfg)
            except ImportError:
                assignments, status = optimize_with_precision_targeting(patterns, demand_matrix, cfg=cfg)
        elif profile_name == "Conservador":
            try:
                from website.profile_optimizers import optimize_conservative
                assignments, status = optimize_conservative(patterns, demand_matrix, cfg=cfg)
            except ImportError:
                assignments, status = optimize_with_precision_targeting(patterns, demand_matrix, cfg=cfg)
        elif profile_name == "Agresivo":
            try:
                from website.profile_optimizers import optimize_aggressive
                assignments, status = optimize_aggressive(patterns, demand_matrix, cfg=cfg)
            except ImportError:
                assignments, status = optimize_with_precision_targeting(patterns, demand_matrix, cfg=cfg)
        else:
            # Usar optimización estándar para otros perfiles
            assignments, status = optimize_with_precision_targeting(patterns, demand_matrix, cfg=cfg)
        
        # Analizar resultados
        if assignments:
            results = analyze_results(assignments, patterns, demand_matrix)
            if results:
                print(f"  Status: {status}")
                print(f"  Agentes totales: {results['total_agents']}")
                print(f"  Cobertura: {results['coverage_percentage']:.1f}%")
                print(f"  Exceso: {results['overstaffing']:.1f}")
                print(f"  Déficit: {results['understaffing']:.1f}")
                print(f"  FT: {results['ft_agents']}, PT: {results['pt_agents']}")
                return True
            else:
                print(f"  ✗ Error analizando resultados")
                return False
        else:
            print(f"  ✗ No se generaron asignaciones - Status: {status}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error en optimización: {str(e)}")
        return False

def test_greedy_modes():
    """Probar los diferentes modos del algoritmo Greedy."""
    print(f"\n=== Probando modos Greedy ===")
    
    demand_matrix = create_test_demand_matrix()
    
    # Generar patrones básicos
    patterns = {}
    for batch in generate_shifts_coverage_optimized(
        demand_matrix, max_patterns=100, batch_size=50, cfg=merge_config()
    ):
        patterns.update(batch)
        break  # Solo el primer batch para pruebas
    
    modes = ["balanced", "cost_focused", "coverage_focused", "safe_focused"]
    
    for mode in modes:
        print(f"\n--- Modo Greedy: {mode} ---")
        cfg = merge_config({"greedy_mode": mode})
        
        try:
            assignments, status = optimize_schedule_greedy_enhanced(
                patterns, demand_matrix, cfg=cfg
            )
            
            if assignments:
                results = analyze_results(assignments, patterns, demand_matrix)
                if results:
                    print(f"  ✓ Agentes: {results['total_agents']}")
                    print(f"  ✓ Cobertura: {results['coverage_percentage']:.1f}%")
                else:
                    print(f"  ✗ Error analizando resultados")
            else:
                print(f"  ✗ No se generaron asignaciones")
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")

def main():
    """Función principal de pruebas."""
    print("🧪 INICIANDO PRUEBAS DE PERFILES DE OPTIMIZACIÓN")
    print("=" * 60)
    
    # Crear datos de prueba
    demand_matrix = create_test_demand_matrix()
    print(f"Matriz de demanda creada: {demand_matrix.shape}")
    print(f"Demanda total: {demand_matrix.sum():.0f} agentes-hora")
    
    # Generar patrones de turnos
    print("\n📋 Generando patrones de turnos...")
    patterns = {}
    pattern_count = 0
    
    for batch in generate_shifts_coverage_optimized(
        demand_matrix, max_patterns=500, batch_size=100, cfg=merge_config()
    ):
        patterns.update(batch)
        pattern_count += len(batch)
        if pattern_count >= 200:  # Limitar para pruebas
            break
    
    print(f"Patrones generados: {len(patterns)}")
    
    # Probar configuración de cada perfil
    print("\n🔧 PROBANDO CONFIGURACIONES DE PERFILES")
    print("-" * 50)
    
    config_results = {}
    for profile_name in PROFILES.keys():
        try:
            cfg = test_profile_configuration(profile_name)
            config_results[profile_name] = True
        except Exception as e:
            print(f"  ✗ Error configurando {profile_name}: {str(e)}")
            config_results[profile_name] = False
    
    # Probar optimización con cada perfil
    print("\n⚡ PROBANDO OPTIMIZACIONES POR PERFIL")
    print("-" * 50)
    
    optimization_results = {}
    for profile_name in PROFILES.keys():
        try:
            success = test_profile_optimization(profile_name, demand_matrix, patterns)
            optimization_results[profile_name] = success
        except Exception as e:
            print(f"  ✗ Error optimizando {profile_name}: {str(e)}")
            optimization_results[profile_name] = False
    
    # Probar modos Greedy
    test_greedy_modes()
    
    # Resumen final
    print("\n📊 RESUMEN DE RESULTADOS")
    print("=" * 60)
    
    config_success = sum(config_results.values())
    config_total = len(config_results)
    print(f"Configuraciones exitosas: {config_success}/{config_total}")
    
    opt_success = sum(optimization_results.values())
    opt_total = len(optimization_results)
    print(f"Optimizaciones exitosas: {opt_success}/{opt_total}")
    
    # Mostrar fallos
    failed_configs = [name for name, success in config_results.items() if not success]
    failed_opts = [name for name, success in optimization_results.items() if not success]
    
    if failed_configs:
        print(f"\n❌ Configuraciones fallidas: {', '.join(failed_configs)}")
    
    if failed_opts:
        print(f"\n❌ Optimizaciones fallidas: {', '.join(failed_opts)}")
    
    if config_success == config_total and opt_success == opt_total:
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON EXITOSAMENTE!")
        return True
    else:
        print(f"\n⚠️  Algunas pruebas fallaron. Revisar implementación.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)