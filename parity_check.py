#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de verificación de paridad con Streamlit original
Compara mismo Excel, mismo perfil entre app1 y esta página
"""

import pandas as pd
from website.profiles import apply_profile
from website.scheduler import (
    load_demand_matrix_from_df, 
    generate_shifts_coverage_corrected,
    optimize_jean_search,
    analyze_results
)

def main():
    print("=== VERIFICACIÓN DE PARIDAD ===")
    
    # Configuración JEAN exacta
    cfg = {
        "optimization_profile": "JEAN",
        "use_ft": True,
        "use_pt": True,
        "allow_8h": True,
        "allow_10h8": True,
        "allow_pt_4h": True,
        "allow_pt_6h": True,
        "allow_pt_5h": False,
        "break_from_start": 2.0,
        "break_from_end": 2.0
    }
    
    # Aplicar perfil JEAN
    cfg = apply_profile(cfg)
    print(f"Perfil aplicado: {cfg.get('optimization_profile')}")
    print(f"Parámetros JEAN: factor={cfg.get('agent_limit_factor')}, penalty={cfg.get('excess_penalty')}")
    
    # Cargar Excel (ajusta la ruta según tu archivo)
    try:
        df = pd.read_excel("Requerido_Ntech (1).xlsx")
        print(f"Excel cargado: {len(df)} filas")
    except FileNotFoundError:
        print("ERROR: No se encontró 'Requerido_Ntech (1).xlsx'")
        print("Coloca el archivo Excel en el directorio raíz o ajusta la ruta")
        return
    
    # Cargar matriz de demanda
    D = load_demand_matrix_from_df(df)
    print(f"Matriz de demanda: {D.shape}, total={D.sum()}")
    
    # Generar patrones
    S = generate_shifts_coverage_corrected(D, cfg=cfg)
    
    ft_count = sum(1 for k in S if k.startswith("FT"))
    pt_count = sum(1 for k in S if k.startswith("PT"))
    total_patterns = len(S)
    
    print(f"Patrones generados:")
    print(f"  FT: {ft_count}")
    print(f"  PT: {pt_count}")
    print(f"  TOTAL: {total_patterns}")
    
    # Verificar si coincide con Streamlit (~1584)
    if 1500 <= total_patterns <= 1700:
        print("✅ TOTAL de patrones coincide con rango esperado (1500-1700)")
    else:
        print(f"⚠️  TOTAL de patrones ({total_patterns}) fuera del rango esperado")
    
    # Ejecutar optimización JEAN
    print("\n=== OPTIMIZACIÓN JEAN ===")
    assignments, status = optimize_jean_search(
        S, D, cfg=cfg
    )
    
    print(f"Status: {status}")
    print(f"Asignaciones: {len(assignments)} turnos")
    
    if assignments:
        results = analyze_results(assignments, S, D)
        if results:
            print(f"\n=== RESULTADOS ===")
            print(f"Agentes totales: {results['total_agents']}")
            print(f"  FT: {results['ft_agents']}")
            print(f"  PT: {results['pt_agents']}")
            print(f"Cobertura: {results['coverage_percentage']:.1f}%")
            print(f"Exceso: {results['overstaffing']}")
            print(f"Déficit: {results['understaffing']}")
            print(f"Score JEAN: {results['overstaffing'] + results['understaffing']}")
            
            # Verificaciones de paridad
            print(f"\n=== VERIFICACIONES ===")
            if results['coverage_percentage'] >= 98.0:
                print("✅ Cobertura >= 98%")
            else:
                print(f"⚠️  Cobertura {results['coverage_percentage']:.1f}% < 98%")
                
            if results['overstaffing'] + results['understaffing'] < 50:
                print("✅ Score JEAN aceptable")
            else:
                print(f"⚠️  Score JEAN alto: {results['overstaffing'] + results['understaffing']}")
        else:
            print("ERROR: No se pudieron analizar los resultados")
    else:
        print("ERROR: No se obtuvieron asignaciones")

if __name__ == "__main__":
    main()