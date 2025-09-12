#!/usr/bin/env python3
"""
Test script simple para probar el archivo Requerido.xlsx con HPO + Cascada 100%
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'website'))

from website.scheduler import run_complete_optimization

def test_requerido():
    """Prueba el archivo Requerido.xlsx con HPO + Cascada 100%"""
    
    # Configuración para HPO + Cascada 100%
    config = {
        "optimization_profile": "HPO + Cascada 100%",
        "use_ft": True,
        "use_pt": True,
        "allow_8h": True,
        "allow_10h8": True,
        "allow_pt_4h": True,
        "allow_pt_6h": True,
        "allow_pt_5h": False,
        "solver_time": 240,
        "iterations": 30,
        "TARGET_COVERAGE": 100.0,
        "agent_limit_factor": 15,
        "excess_penalty": 5.0,
        "peak_bonus": 2.0,
        "critical_bonus": 2.5,
        "allow_excess": False,  # No permitir exceso
        "verbose": True
    }
    
    # Abrir archivo
    file_path = "legacy/Requerido.xlsx"
    
    print(f"[TEST] Probando archivo: {file_path}")
    print(f"[TEST] Perfil: {config['optimization_profile']}")
    print(f"[TEST] Configuracion: FT={config['use_ft']}, PT={config['use_pt']}")
    print("-" * 60)
    
    try:
        with open(file_path, 'rb') as f:
            result = run_complete_optimization(
                f,
                config=config,
                generate_charts=True,
                return_payload=True
            )
        
        if result and 'metrics' in result:
            metrics = result['metrics']
            
            print(f"[RESULTADO] Perfil ejecutado: {result.get('effective_profile', 'N/A')}")
            print(f"[RESULTADO] Status: {result.get('status', 'N/A')}")
            print(f"[RESULTADO] Total Agentes: {metrics.get('agents', 0)}")
            print(f"[RESULTADO] FT: {metrics.get('ft', 0)} - PT: {metrics.get('pt', 0)}")
            print(f"[RESULTADO] Cobertura Pura: {metrics.get('coverage_pure', 0):.1f}%")
            print(f"[RESULTADO] Cobertura Real: {metrics.get('coverage_real', 0):.1f}%")
            print(f"[RESULTADO] Exceso: {metrics.get('excess', 0)}")
            print(f"[RESULTADO] Deficit: {metrics.get('deficit', 0)}")
            
            # Verificar si hay exceso o déficit
            excess = metrics.get('excess', 0)
            deficit = metrics.get('deficit', 0)
            coverage_real = metrics.get('coverage_real', 0)
            
            print("-" * 60)
            if excess > 0:
                print(f"[EXCESO] HAY EXCESO: {excess} unidades")
                print(f"[EXCESO] La cobertura real deberia ser menor a 100% ({coverage_real:.1f}%)")
            else:
                print("[OK] NO HAY EXCESO")
                
            if deficit > 0:
                print(f"[DEFICIT] HAY DEFICIT: {deficit} unidades")
                print(f"[DEFICIT] La cobertura real deberia ser menor a 100% ({coverage_real:.1f}%)")
            else:
                print("[OK] NO HAY DEFICIT")
                
            if excess == 0 and deficit == 0:
                print("[PERFECTO] Cobertura exacta al 100% sin exceso ni deficit")
            elif coverage_real < 100.0 and (excess > 0 or deficit > 0):
                print(f"[FORMULA OK] Cobertura real {coverage_real:.1f}% refleja el exceso/deficit")
            else:
                print(f"[REVISAR] Cobertura real {coverage_real:.1f}% con exceso={excess}, deficit={deficit}")
                
        else:
            print("[ERROR] No se obtuvieron metricas del resultado")
            
    except Exception as e:
        print(f"[ERROR] Error ejecutando optimizacion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_requerido()