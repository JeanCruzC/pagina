# ⏱️ Tiempos de Ejecución - Schedules Generator

## 🎯 Tiempos Estimados por Algoritmo

### **Greedy (Heurístico)**
- ⚡ **Tiempo**: 1-2 minutos
- 🎯 **Características**: Rápido, buena solución
- 📊 **Progreso**: Visible en tiempo real

### **PuLP (Programación Lineal)**
- 🧠 **Tiempo**: 2-4 minutos
- 🎯 **Características**: Más lento, solución óptima
- 📊 **Progreso**: Basado en tiempo transcurrido

## 📋 Indicadores de Progreso

### **Estado "Ejecutando..."**
- Barra de progreso animada
- Tiempo estimado mostrado
- Auto-refresh cada 5 segundos

### **Estado "Completado"**
- Badge verde
- Tiempo real de ejecución
- Resultados visibles

## ⚠️ Importante

1. **NO hagas Ctrl+C** durante la ejecución
2. **Espera a que termine** naturalmente
3. **La página se actualiza** automáticamente
4. **PuLP puede tardar hasta 4 minutos** en casos complejos

## 🔄 Si necesitas cancelar

1. Usa el botón "Cancelar" en la interfaz (si está disponible)
2. O cierra la pestaña del navegador
3. Como último recurso: `taskkill /f /im python.exe`

## 📊 Comparación de Resultados

- **Greedy**: Más agentes, ejecución rápida
- **PuLP**: Menos agentes, solución matemáticamente óptima
- **Ambos**: Se ejecutan en paralelo para comparar