# Análisis de Diferencias: Legacy vs Web App

## Diferencias Críticas Identificadas

### 1. **Generación de Patrones**
**Legacy**: Usa `generate_shifts_coverage_corrected()` con lógica compleja de breaks variables
**Web App**: Usa `generate_shifts_coverage_optimized()` con lógica simplificada

**Problema**: Los patrones generados no son idénticos

### 2. **Configuración de Breaks**
**Legacy**: 
- `break_from_start = 2.5` (configurable)
- `break_from_end = 2.5` (configurable)
- Breaks variables por día según demanda

**Web App**: 
- `break_from_start = 2.0` (fijo)
- `break_from_end = 2.0` (fijo)
- Breaks fijos

### 3. **Parámetros del Perfil JEAN**
**Legacy**:
```python
"JEAN": {
    "agent_limit_factor": 30,
    "excess_penalty": 5.0,
    "peak_bonus": 2.0,
    "critical_bonus": 2.5,
    "TARGET_COVERAGE": 98.0,
    "solver_time": 240
}
```

**Web App**:
```python
"JEAN": {
    "agent_limit_factor": 30,
    "excess_penalty": 5.0,
    "peak_bonus": 2.0,
    "critical_bonus": 2.5,
    "TARGET_COVERAGE": 98.0,
    "solver_time": 240  # Pero se reduce a 30s en la práctica
}
```

### 4. **Algoritmo de Optimización**
**Legacy**: Usa `optimize_jean_search()` con secuencia completa de factores [30, 25, 20, 15, 12, 10, 8]
**Web App**: Usa secuencia reducida [30, 20, 15, 10] por timeout

### 5. **Timeout del Solver**
**Legacy**: 240 segundos por iteración JEAN
**Web App**: 30 segundos máximo total

### 6. **Resolución de Slots**
**Legacy**: Soporte completo para 30 minutos (192 slots/día)
**Web App**: Principalmente 24 horas (24 slots/día)

## Solución Propuesta

1. **Restaurar parámetros exactos del legacy**
2. **Implementar generación de patrones idéntica**
3. **Usar timeouts apropiados**
4. **Configurar breaks variables**