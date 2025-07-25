# Schedules Generator

This project creates optimized work schedules using Streamlit.

## Setup

1. Install the dependencies listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   The list includes the `pulp` package used for solving the optimization problem.

2. Launch the Streamlit application:

   ```bash
   streamlit run "generador_turnos_2025_cnx_BACKUP_F_FIRST_P_LAST (1).py"
   ```

   When prompted, upload the demand Excel file (see assumption below).

3. Choose the **JEAN** profile from the sidebar to minimise the sum of excess and deficit while keeping coverage near 100%.
4. Select **JEAN Personalizado** to choose the working days, hours per day and break placement. All other solver parameters use the JEAN profile automatically.

## Excel Input

The expected Excel file `Requerido.xlsx` must contain a column named `Día` with values from 1 to 7 and a column `Suma de Agentes Requeridos Erlang` representing the hourly staffing requirements.

## Time Slot Resolution

Schedules are represented using 7×24 matrices. **Each slot corresponds to one hour** and this remains the resolution used by the optimizer.  However `load_shift_patterns()` can parse configurations with smaller slots as long as the value divides 60 (for example 30 or 15 minutes).  When the optional `slot_duration_minutes` argument is supplied it overrides any per‑shift setting.  The examples show both hourly and 30‑minute resolutions.

With hourly slots the JEAN profile can produce over a thousand possible patterns when all shift types are enabled (around 1360 for a full seven‑day demand).
Coverage calculations now use compact integer arrays to keep memory usage low during optimization.

Shifts that start late in the evening automatically continue into the next
day, so cross‑midnight schedules are fully supported.

## Perfil JEAN

Incluye un perfil de optimización llamado **JEAN** que minimiza la suma de
exceso y déficit de agentes. El algoritmo prueba diferentes valores de
`agent_limit_factor` y conserva la asignación con la menor suma de exceso y
déficit siempre que la cobertura alcance el objetivo (al menos 98 %).

## Perfil JEAN Personalizado

Permite configurar de forma independiente los turnos **Full Time** y **Part Time**.
Puedes ajustar los días laborables, la duración de la jornada y la ventana de
break para cada tipo. El resto de parámetros del solver se fijan automáticamente
según el perfil **JEAN**. Para Part Time la duración del break puede fijarse en
0 horas si así lo requiere la normativa. Si se habilitan ambos contratos a la vez,
el optimizador ejecuta una estrategia en dos fases que asigna los turnos **Full Time**
sin exceso y luego completa la cobertura con los **Part Time**. A continuación
refina el resultado usando la búsqueda de puntuación **JEAN** para mejorar la
 cobertura y el score.

The solver reports a `coverage_percentage` metric. This value is
demand‑weighted: the total covered demand hours are divided by the total
required hours, providing a more accurate view of how much of the
workload is satisfied.

## JSON Template

The **JEAN Personalizado** sidebar allows loading a configuration template in
JSON format. Upload a file through the *Plantilla JSON* control to pre-fill all
shift parameters and hide the sliders.

Example `shift_config_template.json`:

```json
{
  "use_ft": true,
  "use_pt": true,
  "ft_work_days": 5,
  "ft_shift_hours": 8,
  "ft_break_duration": 1,
  "ft_break_from_start": 2,
  "ft_break_from_end": 2,
  "pt_work_days": 5,
  "pt_shift_hours": 6,
  "pt_break_duration": 1,
  "pt_break_from_start": 2,
  "pt_break_from_end": 2
}
```

Any missing field in the template defaults to the standard slider values.

An additional example is available at `examples/shift_config.json`. It
defines a shift named `FT_12_9_6` with three segments (12 h, 9 h and
6 h) distributed across the specified working days.

Example `examples/shift_config.json`:

```json
{
  "shifts": [
    {
      "name": "FT_12_9_6",
      "pattern": {"work_days": [0, 1, 2], "segments": [12, 9, 6]},
      "break": 1
    }
  ]
}
```

The loader also understands a **v2** format where each shift specifies the
resolution of the start times, the number of segments per duration and a break
window. Upload a file following this structure when using **JEAN Personalizado**
to predefine the available patterns.

Example `examples/shift_config_v2.json`:

```json
{
  "shifts": [
    {
      "name": "FT_12_9_6",
      "slot_duration_minutes": 30,
      "pattern": {
        "work_days": 6,
        "segments": [
          {"hours": 12, "count": 2},
          {"hours": 9,  "count": 2},
          {"hours": 6,  "count": 2}
        ]
      },
      "break": {
        "enabled": true,
        "length_minutes": 60,
        "earliest_after_start": 120,
        "latest_before_end": 120
      }
    }
  ]
}
```

In **v2** each segment may specify a `count` field to produce several patterns
with the same duration. The generator also validates that weekly hours do not
exceed 48 h for full time or 24 h for part time shifts.

A single file may also combine the JEAN slider parameters with the
`shifts` array in **v2** format. See `examples/shift_config_jean_v2.json`
for a complete example.

## Limiting Generated Patterns

`load_shift_patterns()` and the internal `generate_shifts_coverage_optimized()`
functions accept an optional `max_patterns` argument. When omitted the loader
now estimates how many patterns fit in roughly 4&nbsp;GB of the available
memory and caps generation automatically.  Patterns are generated and solved
in batches (2000 patterns by default), sorted by a quick heuristic score, so even 50&nbsp;000+ combinations
can be handled without exhausting RAM.  `generate_shifts_coverage_optimized()`
still honours the `batch_size` option to emit patterns in smaller chunks.

## Memory-aware Generation

Several helper functions monitor memory usage during generation and
optimization.  `monitor_memory_usage()` reports the current RAM usage,
`adaptive_chunk_size()` scales the solver chunk size accordingly and
`emergency_cleanup()` frees memory when consumption exceeds 85 %.  The
loader can also derive optimal start hours via `get_smart_start_hours()`
and limit permutations per shift with `max_patterns_per_shift`.

## Testing

After installing the dependencies, run the test suite with:

```bash
PYTHONPATH=. pytest -q
```
