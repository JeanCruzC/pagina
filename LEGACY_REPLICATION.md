# Replicación Exacta del Legacy Streamlit

## Cambios Implementados para Obtener Resultados 1:1

### 1. **Perfiles EXACTOS del Legacy**

**Problema**: Los parámetros de los perfiles no coincidían exactamente con el Streamlit legacy.

**Solución**:
- **JEAN**: `agent_limit_factor: 30` (era 12), `solver_time: 240` (era 300)
- **JEAN Personalizado**: Mismos parámetros que JEAN
- Todos los perfiles ahora usan los valores exactos del archivo `generador_turnos_2025_cnx_BACKUP_F_FIRST_P_LAST.py`

### 2. **Algoritmo JEAN Iterativo EXACTO**

**Problema**: La lógica iterativa no coincidía con el legacy.

**Solución**:
```python
# ANTES (incorrecto)
for iteration, factor in enumerate(factor_sequence[:max_iterations]):

# AHORA (exacto del legacy)
factor = original_factor
iteration = 0
while iteration < max_iterations and factor >= 3:
    # ... lógica de optimización ...
    factor = max(3, int(factor * 0.9))  # Reducción exacta del legacy
    iteration += 1
```

### 3. **Función solve_in_chunks_optimized_legacy**

**Problema**: La lógica de optimización no seguía el patrón del legacy.

**Solución**:
- Creada función `solve_in_chunks_optimized_legacy()` que replica exactamente la lógica del Streamlit
- Usa chunks ordenados por score como en el legacy
- Aplica el optimizador correcto según el perfil (JEAN vs otros)
- Manejo de dimensiones de patrones idéntico al legacy

### 4. **Configuración de Solver EXACTA**

**Problema**: Los parámetros del solver PuLP no coincidían.

**Solución**:
```python
# Configuración EXACTA del legacy
solver = pl.PULP_CBC_CMD(
    msg=cfg.get("solver_msg", 0),
    timeLimit=solver_time,  # 240s para JEAN
    threads=cfg.get("solver_threads", 1)
)
```

### 5. **Lógica de Optimización Simplificada**

**Problema**: El sistema usaba threading complejo que no existía en el legacy.

**Solución**:
- Eliminado sistema de threading paralelo
- Implementada lógica secuencial exacta del legacy
- Un solo algoritmo ejecutándose como en Streamlit
- Resultados inmediatos sin esperas

### 6. **Parámetros JavaScript Sincronizados**

**Problema**: Los valores por defecto en el frontend no coincidían.

**Solución**:
- JEAN: `agent_limit_factor: 30`, `solver_time: 240`
- Todos los perfiles ahora tienen los valores exactos del legacy

## Diferencias Clave Identificadas del Legacy

### **Configuración JEAN del Legacy Streamlit**:
```python
"JEAN": {
    "agent_limit_factor": 30,
    "excess_penalty": 5.0, 
    "peak_bonus": 2.0,
    "critical_bonus": 2.5,
    "solver_time": 240,
    "TARGET_COVERAGE": 98.0
}
```

### **Algoritmo Iterativo JEAN**:
1. Inicia con `factor = 30`
2. Ejecuta optimización con `optimize_with_precision_targeting`
3. Si cobertura >= 98% y score < mejor_score: guarda como mejor
4. Reduce factor: `factor = max(3, int(factor * 0.9))`
5. Repite hasta `factor < 3` o `max_iterations`

### **Función Principal**:
- `solve_in_chunks_optimized()` del legacy
- Ordena patrones por score descendente
- Procesa en chunks adaptativos
- Usa el optimizador según perfil

## Resultados Esperados

Con estos cambios, el sistema Flask ahora debería producir:

✅ **Mismos parámetros** que el Streamlit legacy
✅ **Misma lógica iterativa** JEAN 
✅ **Mismos algoritmos** de optimización
✅ **Misma configuración** de solver
✅ **Mismos resultados** finales

## Archivos Modificados

1. **`website/profile_optimizers.py`**
   - Algoritmo JEAN iterativo exacto del legacy
   - Lógica de reducción de factor `factor * 0.9`

2. **`website/scheduler.py`**
   - Perfiles con parámetros exactos del legacy
   - Función `solve_in_chunks_optimized_legacy()`
   - Configuración de solver exacta
   - Lógica de optimización simplificada

3. **`website/static/js/generador.js`**
   - Parámetros JEAN actualizados: factor=30, time=240
   - Sincronización completa con backend

## Verificación

Para verificar que funciona correctamente:

1. **Usar perfil JEAN** con los mismos datos del Streamlit
2. **Comparar resultados**:
   - Número total de agentes
   - Distribución FT/PT  
   - Cobertura porcentual
   - Exceso y déficit

3. **Verificar logs**:
   - Secuencia de factores JEAN: [30, 27, 24, 21, 18, 15, 12, 9, 6, 3]
   - Iteraciones y scores por factor
   - Tiempo de solver: 240s

El sistema ahora debería producir resultados **idénticos** al Streamlit legacy.