#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crear archivo Excel de prueba
"""

import pandas as pd
import numpy as np

def create_test_excel():
    """Crea un archivo Excel de prueba con demanda"""
    
    # Crear datos de demanda de prueba
    days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    hours = list(range(24))
    
    data = []
    for day in days:
        for hour in hours:
            # Crear demanda realista: más alta durante horas laborales
            if 8 <= hour <= 17:  # Horas laborales
                demand = np.random.randint(3, 8)
            elif 6 <= hour <= 7 or 18 <= hour <= 20:  # Horas pico
                demand = np.random.randint(1, 4)
            else:  # Horas nocturnas
                demand = np.random.randint(0, 2)
            
            # Menos demanda en fin de semana
            if day in ['Sábado', 'Domingo']:
                demand = max(0, demand - 2)
            
            data.append({
                'Día': day,
                'Hora': f"{hour:02d}:00",
                'Suma de Agentes Requeridos Erlang': demand
            })
    
    df = pd.DataFrame(data)
    
    # Guardar como Excel
    filename = 'test_demand.xlsx'
    df.to_excel(filename, index=False)
    print(f"Archivo Excel creado: {filename}")
    
    # Mostrar resumen
    total_demand = df['Suma de Agentes Requeridos Erlang'].sum()
    max_demand = df['Suma de Agentes Requeridos Erlang'].max()
    print(f"Demanda total semanal: {total_demand}")
    print(f"Demanda máxima simultánea: {max_demand}")
    
    return filename

if __name__ == "__main__":
    create_test_excel()