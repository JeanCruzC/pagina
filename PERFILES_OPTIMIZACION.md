# Perfiles de Optimización - Documentación Completa

Este documento describe todas las funcionalidades y características específicas de cada perfil de optimización implementado en el sistema.

## 1. Equilibrado (Recomendado)

**Descripción**: Balance óptimo entre cobertura y costo

**Características**:
- `agent_limit_factor`: 12 (balance moderado)
- `excess_penalty`: 2.0 (penalización moderada del exceso)
- `peak_bonus`: 1.5 (bonificación moderada en horas pico)
- `critical_bonus`: 2.0 (bonificación moderada en días críticos)
- `solver_time`: 300 segundos
- `precision_mode`: False (velocidad sobre precisión extrema)

**Funcionalidad**:
- Usa optimización estándar con PuLP
- Busca equilibrio entre número de agentes y cobertura
- Tolera exceso moderado para evitar déficit
- Recomendado para la mayoría de casos de uso

## 2. Conservador

**Descripción**: Minimiza riesgos, permite más agentes

**Características**:
- `agent_limit_factor`: 30 (permite muchos más agentes)
- `excess_penalty`: 0.5 (tolera exceso)
- `peak_bonus`: 1.0 (bonificación mínima)
- `critical_bonus`: 1.2 (bonificación baja)
- `solver_time`: 240 segundos
- `safety_margin`: True (margen de seguridad)

**Funcionalidad**:
- Implementa `optimize_conservative()` específico
- Penaliza fuertemente el déficit (factor 10000)
- Permite hasta 20% de exceso sobre la demanda total
- Límites generosos de agentes (factor - 8)
- Ideal para operaciones críticas donde el déficit es inaceptable

## 3. Agresivo

**Descripción**: Maximiza eficiencia, tolera déficit menor

**Características**:
- `agent_limit_factor`: 15 (restrictivo)
- `excess_penalty`: 0.05 (muy tolerante al exceso)
- `peak_bonus`: 1.5 (bonificación moderada)
- `critical_bonus`: 2.0 (bonificación moderada)
- `solver_time`: 180 segundos (rápido)
- `fast_solve`: True (gap de optimalidad 5%)

**Funcionalidad**:
- Implementa `optimize_aggressive()` específico
- Prioriza minimizar agentes (factor 100 en objetivo)
- Permite hasta 5% de déficit controlado
- Límite estricto de exceso (3% máximo)
- Resolución rápida con gap amplio
- Ideal para reducir costos operativos

## 4. Máxima Cobertura

**Descripción**: Prioriza cobertura completa sobre costo

**Características**:
- `agent_limit_factor`: 7 (muy generoso)
- `excess_penalty`: 0.005 (casi no penaliza exceso)
- `peak_bonus`: 3.0 (alta bonificación)
- `critical_bonus`: 4.0 (muy alta bonificación)
- `solver_time`: 400 segundos (tiempo extendido)
- `target_coverage`: 99.5%

**Funcionalidad**:
- Implementa `optimize_maximum_coverage()` específico
- Prioridad máxima en eliminar déficit (factor 1,000,000)
- Bonificaciones adicionales por cobertura en días/horas críticas
- Límites muy generosos de agentes
- Multiplica bonificaciones por importancia relativa de días/horas
- Ideal cuando la cobertura es más importante que el costo

## 5. Mínimo Costo

**Descripción**: Minimiza número de agentes

**Características**:
- `agent_limit_factor`: 35 (muy restrictivo)
- `excess_penalty`: 0.8 (penaliza exceso moderadamente)
- `peak_bonus`: 0.8 (bonificación baja)
- `critical_bonus`: 1.0 (sin bonificación especial)
- `solver_time`: 200 segundos
- `allow_deficit`: True (permite déficit controlado)

**Funcionalidad**:
- Implementa `optimize_minimum_cost()` específico
- Prioriza minimizar agentes (factor 1000 en objetivo)
- Solo usa variables de déficit (no exceso)
- Permite hasta 15% de déficit si está configurado
- Límite muy restrictivo de agentes totales
- Ideal para minimizar costos de personal

## 6. 100% Cobertura Eficiente

**Descripción**: Cobertura completa con mínimo exceso

**Características**:
- `agent_limit_factor`: 6 (muy generoso)
- `excess_penalty`: 0.01 (penalización muy baja)
- `peak_bonus`: 3.5 (alta bonificación)
- `critical_bonus`: 4.5 (muy alta bonificación)
- `solver_time`: 500 segundos
- `target_coverage`: 100.0%
- `max_excess_ratio`: 0.02 (máximo 2% exceso)

**Funcionalidad**:
- Implementa `optimize_perfect_coverage()` específico
- Prohibe cualquier déficit (restricción estricta)
- Limita exceso al 2% de la demanda total
- Penalización alta del exceso (factor 1000)
- Ideal para cobertura perfecta con eficiencia

## 7. 100% Cobertura Total

**Descripción**: Cobertura completa sin restricciones de exceso

**Características**:
- `agent_limit_factor`: 5 (extremadamente generoso)
- `excess_penalty`: 0.001 (penalización mínima)
- `peak_bonus`: 4.0 (muy alta bonificación)
- `critical_bonus`: 5.0 (máxima bonificación)
- `solver_time`: 600 segundos
- `target_coverage`: 100.0%
- `allow_excess`: True

**Funcionalidad**:
- Implementa `optimize_perfect_coverage()` específico
- Prohibe cualquier déficit (restricción estricta)
- Sin restricciones de exceso
- Límites muy generosos de agentes (total_demand / 3)
- Ideal cuando se necesita cobertura garantizada sin importar el costo

## 8. Cobertura Perfecta

**Descripción**: Balance entre cobertura perfecta y eficiencia

**Características**:
- `agent_limit_factor`: 8 (generoso)
- `excess_penalty`: 0.01 (penalización baja)
- `peak_bonus`: 3.0 (alta bonificación)
- `critical_bonus`: 4.0 (muy alta bonificación)
- `solver_time`: 450 segundos
- `target_coverage`: 99.8%

**Funcionalidad**:
- Implementa `optimize_perfect_coverage()` específico
- Balance entre eliminar déficit y controlar exceso
- Penalización moderada del exceso (factor 100)
- Ideal para cobertura casi perfecta con control de costos

## 9. 100% Exacto

**Descripción**: Cobertura exacta sin déficit ni exceso

**Características**:
- `agent_limit_factor`: 6 (generoso)
- `excess_penalty`: 0.005 (penalización muy baja)
- `peak_bonus`: 4.0 (muy alta bonificación)
- `critical_bonus`: 5.0 (máxima bonificación)
- `solver_time`: 600 segundos
- `target_coverage`: 100.0%
- `zero_excess`: True
- `zero_deficit`: True

**Funcionalidad**:
- Implementa `optimize_perfect_coverage()` específico
- Prohibe completamente déficit y exceso (factor 1,000,000)
- Restricciones estrictas: déficit = 0, exceso = 0
- Gap de optimalidad muy estricto (0.1%)
- Ideal para cobertura matemáticamente exacta

## 10. JEAN

**Descripción**: Búsqueda iterativa para minimizar exceso+déficit

**Características**:
- `agent_limit_factor`: 30 (inicial, se reduce iterativamente)
- `excess_penalty`: 5.0 (alta penalización)
- `peak_bonus`: 2.0 (bonificación moderada)
- `critical_bonus`: 2.5 (bonificación moderada-alta)
- `TARGET_COVERAGE`: 98.0%
- `iterative_search`: True
- `search_iterations`: 5

**Funcionalidad**:
- Implementa `optimize_jean_search()` específico
- Búsqueda iterativa con secuencia de factores: [30, 27, 24, 21, 18, 15, 12, 9, 6, 3]
- Minimiza la suma de exceso + déficit
- Mantiene cobertura ≥ 98%
- Para cuando encuentra solución óptima o completa iteraciones
- Basado en el algoritmo legacy de Streamlit

## 11. JEAN Personalizado

**Descripción**: JEAN con configuración personalizada de turnos

**Características**:
- Hereda parámetros de JEAN
- `custom_shifts`: True (carga turnos desde JSON)
- `ft_pt_strategy`: True (estrategia 2 fases)
- `slot_duration_minutes`: Configurable (30 min por defecto)

**Funcionalidad**:
- Carga patrones personalizados desde archivo JSON
- Estrategia 2 fases: FT sin exceso → PT para completar
- Refinamiento con búsqueda JEAN si no es óptimo
- Soporte para configuraciones complejas de turnos
- Ideal para organizaciones con patrones específicos

## 12. Personalizado

**Descripción**: Configuración manual de parámetros

**Características**:
- `agent_limit_factor`: 25 (configurable por usuario)
- `excess_penalty`: 0.5 (configurable por usuario)
- `peak_bonus`: 1.5 (configurable por usuario)
- `critical_bonus`: 2.0 (configurable por usuario)
- `user_defined`: True

**Funcionalidad**:
- Permite configuración manual de todos los parámetros
- Usa optimización estándar con parámetros personalizados
- Interfaz de sliders para ajuste en tiempo real
- Ideal para usuarios expertos que necesitan control total

## 13. Aprendizaje Adaptativo

**Descripción**: IA que evoluciona automáticamente

**Características**:
- `agent_limit_factor`: 8 (inicial, evoluciona)
- `excess_penalty`: 0.01 (inicial, evoluciona)
- `peak_bonus`: 3.0 (inicial, evoluciona)
- `critical_bonus`: 4.0 (inicial, evoluciona)
- `learning_enabled`: True
- `adaptive`: True

**Funcionalidad**:
- Implementa `optimize_adaptive_learning()` específico
- Sistema de aprendizaje evolutivo que mejora en cada ejecución
- Guarda historial de ejecuciones en `optimization_learning.json`
- Ajusta parámetros automáticamente según resultados previos
- Estrategias evolutivas: aggressive, moderate, fine_tune, explore
- Calcula firmas de demanda para patrones similares
- Ideal para operaciones recurrentes que mejoran con el tiempo

## Algoritmos Greedy Adaptativos

Cada perfil también adapta el comportamiento del algoritmo Greedy:

### Modos Greedy:
- **cost_focused**: Minimiza agentes, tolera más exceso
- **coverage_focused**: Maximiza cobertura, bonifica cobertura adicional
- **safe_focused**: Enfoque conservador con multiplicadores moderados
- **balanced**: Balance estándar entre todos los factores

### Criterios de Parada Adaptativos:
- **cost_focused**: Para antes (threshold 1.0)
- **coverage_focused**: Continúa más tiempo (threshold 0.1)
- **otros**: Threshold estándar (0.5)

## Selección de Resultados por Perfil

El sistema selecciona automáticamente el mejor resultado según el perfil:

- **Mínimo Costo/Agresivo**: Prefiere menor número de agentes
- **Máxima Cobertura/100% Total**: Prioriza PuLP para mejor cobertura
- **JEAN/JEAN Personalizado**: Siempre usa resultado PuLP
- **Otros**: Balance general entre PuLP y Greedy

## Configuraciones Específicas por Perfil

Cada perfil configura automáticamente:
- Tiempo de solver optimizado
- Modo de precisión apropiado
- Restricciones específicas
- Bonificaciones y penalizaciones
- Estrategias de resolución
- Criterios de parada
- Límites de memoria y patrones

Esta implementación completa asegura que cada perfil tenga comportamientos únicos y optimizados para sus casos de uso específicos.