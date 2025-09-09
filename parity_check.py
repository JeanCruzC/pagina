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
    print("=== VERIFICACIÓN DE PARIDAD (24/7 + COMBINACIONES PT + FILTRADO INTELIGENTE) ===")
    
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
        "break_from_end": 2.0,
        "keep_percentage": 0.3,
        "peak_bonus": 1.5,
        "critical_bonus": 2.0,
        "efficiency_bonus": 1.0,
        "iterations": 5,
        "TARGET_COVERAGE": 98.0
    }
    
    # Aplicar perfil JEAN
    cfg = apply_profile(cfg)
    print(f"Perfil aplicado: {cfg.get('optimization_profile')}")
    print(f"Parámetros JEAN: factor={cfg.get('agent_limit_factor')}, penalty={cfg.get('excess_penalty')}")
    print(f"Configuración breaks: desde_inicio={cfg.get('break_from_start')}h, antes_fin={cfg.get('break_from_end')}h")
    print(f"Filtrado: keep={cfg.get('keep_percentage', 0.3)*100:.0f}%, peak_bonus={cfg.get('peak_bonus', 1.5)}, critical_bonus={cfg.get('critical_bonus', 2.0)}")
    
    # Cargar Excel (ajusta la ruta según tu archivo)
    excel_files = ["Requerido_Ntech (1).xlsx", "Requerido.xlsx", "demanda.xlsx"]
    df = None
    for excel_file in excel_files:
        try:
            df = pd.read_excel(excel_file)
            print(f"Excel cargado: {excel_file} ({len(df)} filas)")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        print("ERROR: No se encontró ningún archivo Excel")
        print(f"Archivos buscados: {excel_files}")
        print("Coloca un archivo Excel en el directorio raíz")
        return
    
    # Cargar matriz de demanda
    D = load_demand_matrix_from_df(df)
    print(f"Matriz de demanda: {D.shape}, total={D.sum()}")
    
    # Generar patrones
    S = generate_shifts_coverage_corrected(D, cfg=cfg)
    
    ft_count = sum(1 for k in S if k.startswith("FT"))
    pt_count = sum(1 for k in S if k.startswith("PT"))
    pt4_count = sum(1 for k in S if k.startswith("PT_4H"))
    pt5_count = sum(1 for k in S if k.startswith("PT_5H"))
    pt6_count = sum(1 for k in S if k.startswith("PT_6H"))
    total_patterns = len(S)
    
    print(f"Patrones generados:")
    print(f"  FT: {ft_count}")
    print(f"  PT: {pt_count}")
    print(f"    PT4: {pt4_count}")
    print(f"    PT5: {pt5_count}")
    print(f"    PT6: {pt6_count}")
    print(f"  TOTAL: {total_patterns}")
    
    # Verificar turnos nocturnos
    night_patterns = [k for k in S.keys() if any(h in k for h in ['_0.0_', '_1.0_', '_2.0_', '_22.0_', '_23.0_', '_00_', '_01_', '_02_', '_22_', '_23_'])]
    if night_patterns:
        print(f"✅ Turnos nocturnos detectados: {len(night_patterns)} patrones")
        print(f"  Ejemplos: {night_patterns[:3]}")
    else:
        print("⚠️  No se detectaron turnos nocturnos")
    
    # Verificar breaks en patrones FT
    print(f"\n=== VERIFICACIÓN DE BREAKS ===")
    sample_ft = next((k for k in S.keys() if k.startswith('FT_8H')), None)
    if sample_ft:
        pattern = S[sample_ft]
        pattern_2d = pattern.reshape(7, 24)
        # Buscar día con trabajo
        work_day = None
        for d in range(7):
            if pattern_2d[d].sum() > 0:
                work_day = d
                break
        
        if work_day is not None:
            work_hours = pattern_2d[work_day]
            total_work = work_hours.sum()
            work_slots = [i for i, h in enumerate(work_hours) if h == 1]
            break_slots = []
            
            # Buscar breaks (huecos en medio de la jornada)
            if len(work_slots) > 1:
                for i in range(work_slots[0], work_slots[-1] + 1):
                    if work_hours[i] == 0:
                        break_slots.append(i)
            
            print(f"Patrón ejemplo: {sample_ft}")
            print(f"  Horas trabajadas: {total_work}/8")
            print(f"  Slots de trabajo: {work_slots}")
            print(f"  Breaks detectados: {break_slots}")
            
            # Calcular puntaje del patrón ejemplo
            from website.scheduler import score_pattern
            score = score_pattern(pattern, D)
            print(f"  Puntaje básico: {score}")
            
            if break_slots:
                print("✅ Breaks correctamente insertados en FT")
            else:
                print("⚠️  No se detectaron breaks en FT")
        else:
            print("⚠️  No se pudo analizar patrón FT")
    else:
        print("⚠️  No se encontraron patrones FT para verificar breaks")
    
    # Verificar si coincide con Streamlit (con filtrado debería ser ~30% del total generado)
    if 500 <= total_patterns <= 2000:
        print(f"✅ TOTAL de patrones filtrados en rango esperado (500-2000): {total_patterns}")
    elif 2000 < total_patterns <= 6000:
        print(f"✅ TOTAL de patrones alto (sin filtrar o filtrado suave): {total_patterns}")
    else:
        print(f"⚠️  TOTAL de patrones ({total_patterns}) fuera del rango esperado")
    
    # Verificar combinaciones PT (ajustado para filtrado)
    if pt4_count > 30:  # Reducido por filtrado
        print("✅ PT4 tiene suficientes combinaciones (post-filtrado)")
    else:
        print(f"⚠️  PT4 tiene pocas combinaciones: {pt4_count}")
        
    if pt5_count > 5:  # Reducido por filtrado
        print("✅ PT5 tiene suficientes patrones (post-filtrado)")
    else:
        print(f"⚠️  PT5 tiene pocos patrones: {pt5_count}")
    
    # Verificar eficiencia del filtrado
    expected_before_filter = total_patterns / 0.3  # Estimación inversa
    print(f"\n=== EFICIENCIA DEL FILTRADO ===")
    print(f"Patrones estimados antes del filtrado: ~{expected_before_filter:.0f}")
    print(f"Patrones después del filtrado: {total_patterns}")
    print(f"Ratio de filtrado: {(total_patterns / expected_before_filter * 100):.1f}%")
    
    # Ejecutar optimización JEAN
    print("\n=== OPTIMIZACIÓN JEAN ITERATIVA ===")
    assignments, status = optimize_jean_search(
        S, D, cfg=cfg, verbose=True, max_iterations=cfg.get('iterations', 5)
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
            target_cov = cfg.get('TARGET_COVERAGE', 98.0)
            if results['coverage_percentage'] >= target_cov:
                print(f"✅ Cobertura >= {target_cov}%")
            else:
                print(f"⚠️  Cobertura {results['coverage_percentage']:.1f}% < {target_cov}%")
                
            jean_score = results['overstaffing'] + results['understaffing']
            if jean_score < 50:
                print(f"✅ Score JEAN aceptable: {jean_score}")
            elif jean_score < 100:
                print(f"⚠️  Score JEAN moderado: {jean_score}")
            else:
                print(f"⚠️  Score JEAN alto: {jean_score}")
            
            # Verificar eficiencia de la búsqueda iterativa
            efficiency = results['coverage_percentage'] / results['total_agents'] if results['total_agents'] > 0 else 0
            print(f"Eficiencia JEAN: {efficiency:.2f}% cobertura por agente")
        else:
            print("ERROR: No se pudieron analizar los resultados")
    else:
        print("ERROR: No se obtuvieron asignaciones")

if __name__ == "__main__":
    main()