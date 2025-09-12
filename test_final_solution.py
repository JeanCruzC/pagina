#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prueba final de la solución completa
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'website'))

from website.scheduler import run_complete_optimization
import io

def test_final_solution():
    """Prueba final con la configuración real de la web"""
    print("=== PRUEBA FINAL DE LA SOLUCIÓN COMPLETA ===\n")
    
    # Configuración exacta que viene de la web
    cfg = {
        "optimization_profile": "HPO + Cascada 100%",
        "hpo_trials": 12,
        "solver_time": 240,
        "TARGET_COVERAGE": 100.0,
        "coverage_method": "efficiency",  # Ahora por defecto
        "penalty_factor": 1.0,
        "use_ft": True,
        "use_pt": True,
        "allow_8h": True,
        "allow_pt_4h": True,
        "allow_pt_6h": True,
    }
    
    print(f"Configuración de prueba: {cfg}")
    print()
    
    # Usar archivo Excel real
    with open('test_demand.xlsx', 'rb') as f:
        excel_file = io.BytesIO(f.read())
    
    result = run_complete_optimization(
        excel_file,
        config=cfg,
        return_payload=True
    )
    
    if result and "metrics" in result:
        metrics = result["metrics"]
        print("=== RESULTADO FINAL ===")
        print(f"Status: {result.get('status', 'N/A')}")
        print(f"Perfil efectivo: {result.get('effective_profile', 'N/A')}")
        print(f"Agentes totales: {metrics.get('agents', 0)}")
        print(f"  - FT: {metrics.get('ft', 0)}")
        print(f"  - PT: {metrics.get('pt', 0)}")
        print(f"Cobertura pura: {metrics.get('coverage_pure', 0):.1f}%")
        print(f"Cobertura real: {metrics.get('coverage_real', 0):.1f}%")
        print(f"Exceso: {metrics.get('excess', 0)}")
        print(f"Déficit: {metrics.get('deficit', 0)}")
        
        # Verificar que funciona correctamente
        cov_real = metrics.get('coverage_real', 0)
        exceso = metrics.get('excess', 0)
        
        print("\n=== VERIFICACIÓN ===")
        if exceso > 0 and cov_real < 100:
            print(f"✅ ÉXITO: El sistema penaliza correctamente el exceso")
            print(f"   Con {exceso} de exceso, la cobertura real es {cov_real:.1f}% (no 100%)")
            print(f"   Esto significa que las nuevas fórmulas están funcionando")
        elif exceso > 0 and cov_real >= 100:
            print(f"❌ PROBLEMA: Aún muestra {cov_real:.1f}% con exceso {exceso}")
            print(f"   Las nuevas fórmulas no se están aplicando correctamente")
        elif exceso == 0:
            print(f"ℹ️  Sin exceso para probar: cobertura {cov_real:.1f}%, exceso {exceso}")
            print(f"   El sistema está funcionando eficientemente sin exceso")
        
        print(f"\n=== INTERPRETACIÓN ===")
        print(f"Antes (método original): Cualquier exceso = 100% cobertura")
        print(f"Ahora (método efficiency): Exceso penaliza la cobertura real")
        print(f"Resultado: {cov_real:.1f}% refleja la eficiencia real del staffing")
        
    else:
        print("❌ Error: No se obtuvieron métricas del resultado")

if __name__ == "__main__":
    test_final_solution()