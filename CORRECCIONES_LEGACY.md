# Correcciones para Paridad con Legacy Streamlit

## Problema Identificado
Los resultados de la web app no coincidían exactamente con el legacy Streamlit. Las capturas muestran:
- **Legacy**: ~105-108 agentes con cobertura ~98-100%
- **Web App**: Resultados diferentes

## Correcciones Implementadas

### 1. **Configuración por Defecto (scheduler.py)**
```python
# ANTES
"solver_time": 300,
"break_from_start": 2.0,
"break_from_end": 2.0,

# DESPUÉS - EXACTO del legacy
"solver_time": 240,  # EXACTO del legacy
"break_from_start": 2.5,  # EXACTO del legacy
"break_from_end": 2.5,   # EXACTO del legacy
```

### 2. **Timeout del Solver PuLP (optimizer_pulp.py)**
```python
# ANTES - Límite artificial
solver_time = min(cfg.get("solver_time", 300), 30)  # Máximo 30 segundos

# DESPUÉS - EXACTO del legacy
solver_time = cfg.get("solver_time", 240)  # EXACTO del legacy - sin límite artificial
```

### 3. **Configuración del Solver CBC**
```python
# ANTES - Configuración agresiva
solver = pl.PULP_CBC_CMD(
    msg=1,
    timeLimit=30,  # Muy corto
    gapRel=0.15,   # Gap muy permisivo
    threads=2      # Pocos threads
)

# DESPUÉS - EXACTA del legacy
solver = pl.PULP_CBC_CMD(
    msg=1,
    timeLimit=solver_time,  # EXACTO del legacy
    gapRel=0.05,           # Gap del legacy
    threads=4              # Threads del legacy
)
```

### 4. **Secuencia de Factores JEAN**
```python
# ANTES - Secuencia reducida
factor_sequence = [30, 20, 15, 10]  # Secuencia más corta

# DESPUÉS - EXACTA del legacy
factor_sequence = [30, 25, 20, 15, 12, 10, 8]  # EXACTA del legacy
```

### 5. **Perfil JEAN en PROFILES**
Confirmado que los parámetros son exactos:
```python
"JEAN": {
    "agent_limit_factor": 30,
    "excess_penalty": 5.0,
    "peak_bonus": 2.0,
    "critical_bonus": 2.5,
    "TARGET_COVERAGE": 98.0,
    "solver_time": 240,  # EXACTO del legacy
}
```

## Diferencias Clave Corregidas

| Aspecto | Legacy | Web App (Antes) | Web App (Después) |
|---------|--------|-----------------|-------------------|
| solver_time | 240s | 30s | 240s ✅ |
| break_from_start | 2.5h | 2.0h | 2.5h ✅ |
| break_from_end | 2.5h | 2.0h | 2.5h ✅ |
| Secuencia JEAN | [30,25,20,15,12,10,8] | [30,20,15,10] | [30,25,20,15,12,10,8] ✅ |
| Gap CBC | 0.05 | 0.15 | 0.05 ✅ |
| Threads CBC | 4 | 2 | 4 ✅ |

## Resultados Esperados

Con estas correcciones, la web app debería producir resultados **idénticos** al legacy:
- **Total agentes**: 105-108 (según demanda)
- **Cobertura**: ≥98%
- **Perfil JEAN**: Minimización de exceso+déficit
- **Turnos**: Combinación de FT y PT según configuración

## Verificación

Ejecutar el script de prueba:
```bash
python test_legacy_parity.py
```

Este script verifica que los resultados estén dentro del rango esperado del legacy Streamlit.

## Estado

✅ **COMPLETADO** - Todas las correcciones implementadas para lograr paridad 1:1 con el legacy Streamlit.