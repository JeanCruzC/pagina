import io, base64, time, gc, hashlib
import numpy as np
try:
    import pandas as pd
except Exception:  # pragma: no cover
    class _PDPlaceholder:
        class DataFrame:  # type: ignore
            ...
    pd = _PDPlaceholder()
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    matplotlib = None
    plt = None
from collections import Counter
try:  # pragma: no cover
    from flask import current_app
except Exception:  # pragma: no cover
    current_app = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    from .profiles import PROFILES, resolve_profile_name, apply_profile
except Exception:  # pragma: no cover
    from website.profiles import PROFILES, resolve_profile_name, apply_profile

# === Robust import of profile optimizer selector ===
try:
    from .profile_optimizers import get_profile_optimizer as _get_profile_optimizer
except Exception:
    try:
        from profile_optimizers import get_profile_optimizer as _get_profile_optimizer
    except Exception as e:
        _get_profile_optimizer = None
        _get_profile_optimizer_err = e


def get_profile_optimizer(profile_name: str):
    """
    Devuelve un optimizador callable. Nunca lanza ImportError.
    Si no hay módulo, hace fallback a solve_in_chunks_optimized.
    """

    if _get_profile_optimizer is not None:
        return _get_profile_optimizer(profile_name)

    def _fallback(shifts_coverage, demand_matrix, *, cfg=None, job_id=None, **_):
        _cfg = cfg or {}
        return (
            solve_in_chunks_optimized(
                shifts_coverage,
                demand_matrix,
                optimization_profile=profile_name,
                use_ft=_cfg.get("use_ft", True),
                use_pt=_cfg.get("use_pt", True),
                TARGET_COVERAGE=_cfg.get("TARGET_COVERAGE", 98.0),
                agent_limit_factor=_cfg.get("agent_limit_factor", 12),
                excess_penalty=_cfg.get("excess_penalty", 2.0),
                peak_bonus=_cfg.get("peak_bonus", 1.5),
                critical_bonus=_cfg.get("critical_bonus", 2.0),
            ),
            "FALLBACK_CHUNKS",
        )

    return _fallback

try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    pulp = None
    PULP_AVAILABLE = False


def _build_cbc_solver(cfg):
    """
    Construye un PULP_CBC_CMD compatible con PuLP>=2.7/2.8.
    - timeLimit: respeta cfg['solver_time'] (0 => sin límite -> None)
    - msg: respeta cfg['solver_msg']
    - options: pasa 'randomSeed 42', 'threads N' como CADENAS
    """
    time_limit = int(cfg.get("solver_time", 120))
    msg = bool(cfg.get("solver_msg", True))

    # Opcional: si manejas hilos desde config; si no, se ignora.
    threads = int(cfg.get("solver_threads", 1))

    # Semilla determinística (misma que usabas): 42 por defecto
    seed = int(cfg.get("random_seed", 42))

    opts = []
    # IMPORTANTE: options deben ser strings tipo "clave valor"
    if seed is not None:
        opts.append(f"randomSeed {seed}")
    if threads and threads > 1:
        opts.append(f"threads {threads}")

    return pulp.PULP_CBC_CMD(
        msg=msg,
        timeLimit=(time_limit if time_limit > 0 else None),
        options=opts,
    )


def load_demand_matrix_from_df(df: pd.DataFrame) -> np.ndarray:
    """
    Carga una matriz 7x24 desde un DataFrame con columnas de día y hora.
    Acepta día como nombre (Lunes..Domingo) o número (1..7) y hora como 11 o '11:00'.
    También intenta varios nombres de columna habituales.
    """
    import numpy as np
    days_idx = {
        "lunes": 0,
        "martes": 1,
        "miércoles": 2,
        "miercoles": 2,
        "jueves": 3,
        "viernes": 4,
        "sábado": 5,
        "sabado": 5,
        "domingo": 6,
    }
    mat = np.zeros((7, 24), dtype=float)

    cols = {c.lower(): c for c in df.columns}
    day_col = cols.get("día") or cols.get("dia") or cols.get("day") or list(df.columns)[0]
    time_col = cols.get("horario") or cols.get("hora") or cols.get("hour") or list(df.columns)[1]
    val_col = (
        cols.get("suma de agentes requeridos erlang")
        or cols.get("agentes")
        or cols.get("valor")
        or list(df.columns)[-1]
    )

    for _, row in df.iterrows():
        d_raw = row.get(day_col)
        t_raw = row.get(time_col)
        v_raw = row.get(val_col)
        if pd.isna(d_raw) or pd.isna(t_raw) or pd.isna(v_raw):
            continue

        # Día -> índice 0..6
        if isinstance(d_raw, (int, float)):
            d_idx = int(d_raw) - 1
        else:
            d_idx = days_idx.get(str(d_raw).strip().lower())
        if d_idx is None or not (0 <= d_idx <= 6):
            continue

        # Hora -> 0..23
        s = str(t_raw).strip()
        try:
            h = int(s.split(":")[0]) if ":" in s else int(float(s))
        except Exception:
            continue
        if not (0 <= h <= 23):
            continue

        try:
            v = float(v_raw)
        except Exception:
            continue
        mat[d_idx, h] = v

    # Igual al parser de Streamlit: redondeo y casting a int
    return np.rint(mat).astype(int)


# Utilidad para validar patrones según los toggles del perfil
def _allowed_by_toggles(name: str, cfg: dict) -> bool:
    # Bloques por familia
    if name.startswith('FT') and not cfg.get('use_ft', True):
        return False
    if name.startswith('PT') and not cfg.get('use_pt', True):
        return False

    # Compatibilidad de prefijos (nueva y “legacy”)
    FT8  = name.startswith(('FT8_', 'FT_8H_'))
    FT10 = name.startswith(('FT10p8_', 'FT_10H_', 'FT_10H8_'))
    PT4  = name.startswith(('PT4_', 'PT_4H_'))
    PT6  = name.startswith(('PT6_', 'PT_6H_'))
    PT5  = name.startswith(('PT5_', 'PT_5H_'))

    if FT8:  return bool(cfg.get('allow_8h', False))
    if FT10: return bool(cfg.get('allow_10h8', False))
    if PT4:  return bool(cfg.get('allow_pt_4h', False))
    if PT6:  return bool(cfg.get('allow_pt_6h', False))
    if PT5:  return bool(cfg.get('allow_pt_5h', False))

    # Si no matchea ningún formato conocido, lo bloqueamos.
    return False


# --- Helpers de filtrado por familias/FT-PT ---
def _pattern_id(pid_or_dict):
    if isinstance(pid_or_dict, dict):
        return pid_or_dict.get("id") or pid_or_dict.get("name") or pid_or_dict.get("pattern")
    return str(pid_or_dict)


def _allow_pid(pid, cfg):
    pid = str(pid)
    # Bloqueo por grupo FT/PT
    if not cfg.get("use_ft", True) and pid.startswith("FT_"):
        return False
    if not cfg.get("use_pt", True) and pid.startswith("PT_"):
        return False

    # Familias FT
    if pid.startswith("FT_8H") and not cfg.get("allow_8h", False):
        return False
    if pid.startswith("FT_10H") and not cfg.get("allow_10h8", False):
        return False

    # Familias PT
    if pid.startswith("PT_4H") and not cfg.get("allow_pt_4h", False):
        return False
    if pid.startswith("PT_5H") and not cfg.get("allow_pt_5h", False):
        return False
    if pid.startswith("PT_6H") and not cfg.get("allow_pt_6h", False):
        return False

    return True


def _filter_patterns_by_cfg(patterns, cfg):
    """
    patterns: dict[str, dict|array]  (id -> patrón)
    Devuelve solo los patrones permitidos por cfg (familias/FT-PT).
    """
    if not patterns:
        return patterns
    filtered = {}
    for pid, pdef in patterns.items():
        if _allow_pid(pid, cfg):
            filtered[pid] = pdef
    return filtered


def _is_allowed_pid(pid: str, cfg: dict) -> bool:
    allow_8h   = cfg.get("allow_8h", True)
    allow_10h8 = cfg.get("allow_10h8", False)
    allow_pt_4h = cfg.get("allow_pt_4h", True)
    allow_pt_5h = cfg.get("allow_pt_5h", False)
    allow_pt_6h = cfg.get("allow_pt_6h", True)

    # FT (ambos estilos)
    if pid.startswith("FT"):
        if pid.startswith(("FT8_", "FT_8H_")) and not allow_8h:
            return False
        if pid.startswith(("FT10p8_", "FT_10H_", "FT_10H8_")) and not allow_10h8:
            return False
        return True

    # PT (ambos estilos)
    if pid.startswith("PT"):
        if pid.startswith(("PT4_", "PT_4H_")) and not allow_pt_4h:
            return False
        if pid.startswith(("PT5_", "PT_5H_")) and not allow_pt_5h:
            return False
        if pid.startswith(("PT6_", "PT_6H_")) and not allow_pt_6h:
            return False
        return True

    return True


def _build_pattern(days, durations, start_hour, break_len, break_from_start, break_from_end, slot_factor=1):
    """Construye matriz semanal con breaks usando lógica original exacta"""
    slots_per_day = 24 * slot_factor
    pattern = np.zeros((7, slots_per_day), dtype=np.int8)
    
    for day, dur in zip(days, durations):
        # Marcar horas trabajadas
        for s in range(int(dur * slot_factor)):
            slot = int(start_hour * slot_factor) + s
            d_off, idx = divmod(slot, slots_per_day)
            pattern[(day + d_off) % 7, idx] = 1
        
        # Insertar break en la mitad del turno
        if break_len:
            b_start = int((start_hour + break_from_start) * slot_factor)
            b_end = int((start_hour + dur - break_from_end) * slot_factor)
            b_slot = b_start + (b_end - b_start) // 2 if b_start < b_end else b_start
            
            for b in range(int(break_len * slot_factor)):
                slot = b_slot + b
                d_off, idx = divmod(slot, slots_per_day)
                pattern[(day + d_off) % 7, idx] = 0
    
    return pattern.flatten()

def generate_weekly_pattern_simple(start_hour, duration, working_days):
    """Wrapper para compatibilidad - sin breaks"""
    return _build_pattern(working_days, [duration] * len(working_days), start_hour, 0, 0, 0)

def get_valid_break_times(start_hour, duration, cfg):
    """Obtiene todas las franjas válidas de break para un turno."""
    valid_breaks = []
    earliest_break = start_hour + cfg.get("break_from_start", 2.0)
    latest_break = start_hour + duration - cfg.get("break_from_end", 2.0) - 1
    current_time = earliest_break
    while current_time <= latest_break:
        # Solo permitir breaks en horas en punto o media
        if current_time % 0.5 == 0:
            valid_breaks.append(round(current_time, 1))
        current_time += 0.5
    return valid_breaks[:7]  # máximo 7 opciones

def generate_weekly_pattern_with_break(start_hour, duration, working_days, dso_day, break_start, break_len=1):
    """Genera un patrón semanal con un break específico (corrigiendo cruce de medianoche)."""
    pattern = np.zeros((7, 24), dtype=np.int8)
    for day in working_days:
        if day == dso_day:
            continue
        # Marcar horas trabajadas
        for h in range(duration):
            t = start_hour + h
            d_off, idx = divmod(int(t), 24)
            pattern[(day + d_off) % 7, idx] = 1
        # Insertar break en la hora especificada
        for b in range(int(break_len)):
            t = break_start + b
            d_off, idx = divmod(int(t), 24)
            pattern[(day + d_off) % 7, idx] = 0
    return pattern.flatten()

def generate_weekly_pattern_with_break_compat(start_hour, duration, working_days, break_len=1, break_from_start=2.0, break_from_end=2.0):
    """Wrapper para compatibilidad - con breaks"""
    return _build_pattern(working_days, [duration] * len(working_days), start_hour, break_len, break_from_start, break_from_end)


def generate_weekly_pattern_pt5(start_hour, working_days):
    """PT5H: 4 días de 5h + 1 día de 4h (24h/sem)"""
    if not working_days:
        return np.zeros(7 * 24, dtype=np.int8)
    four_hour_day = working_days[-1]
    durations = [4 if d == four_hour_day else 5 for d in working_days]
    return _build_pattern(working_days, durations, start_hour, 0, 0, 0)

def generate_weekly_pattern_10h8(start_hour, working_days, eight_hour_day, break_len=1, break_from_start=2.0, break_from_end=2.0):
    """FT 10h8: 4 días de 10h + 1 día de 8h con breaks"""
    durations = [8 if d == eight_hour_day else 10 for d in working_days]
    return _build_pattern(working_days, durations, start_hour, break_len, break_from_start, break_from_end)


def score_pattern(pat, demand_matrix) -> int:
    pat_arr = np.asarray(pat)
    dm_arr = np.asarray(demand_matrix)
    if pat_arr.ndim == 1:
        slots = pat_arr.reshape(dm_arr.shape)
        covered = np.minimum(slots, dm_arr).sum()
        return int(covered)
    pat_arr = pat_arr.astype(np.uint8)
    dm_arr = dm_arr.astype(np.uint8)
    return int(np.unpackbits(np.bitwise_and(pat_arr, dm_arr)).sum())

def score_and_filter_patterns(patterns: dict, demand: np.ndarray, keep_percentage=0.3,
                             peak_bonus=1.5, critical_bonus=2.0, efficiency_bonus=1.0):
    """Pattern scoring with exact APP1 logic - efficiency as additive bonus"""
    if demand is None or not patterns:
        return patterns
    
    # Calculate totals by day and hour
    daily = demand.sum(axis=1)
    hourly = demand.sum(axis=0)
    
    # 2 days with highest total demand
    critical_days = set(np.argsort(daily)[-2:]) if daily.size > 1 else {int(np.argmax(daily))}
    
    # Hours >= P75 of demand
    peak_hours = set(np.where(hourly >= np.percentile(hourly[hourly > 0], 75) if np.any(hourly > 0) else 0)[0])
    
    scores = []
    for name, pat in patterns.items():
        try:
            pat_mat = np.array(pat).reshape(demand.shape)
            coverage = np.minimum(pat_mat, demand)
            
            # Base score: coverage sum
            score = coverage.sum()
            
            # Peak hours bonus (multiplicative)
            if peak_hours:
                score += coverage[:, list(peak_hours)].sum() * peak_bonus
            
            # Critical days bonus (multiplicative)
            if critical_days:
                score += coverage[list(critical_days), :].sum() * critical_bonus
            
            # Efficiency bonus (additive, not multiplicative like before)
            total_hours = pat_mat.sum()
            if total_hours > 0:
                efficiency = coverage.sum() / total_hours
                score = score + efficiency * efficiency_bonus  # Additive like APP1
            
            scores.append((score, name))
        except Exception:
            # If error, assign minimum score
            scores.append((0, name))
    
    # Sort by score descending
    scores.sort(reverse=True, key=lambda x: x[0])
    
    # Keep top X%
    keep_n = max(1, int(len(scores) * keep_percentage))
    top_names = {name for _, name in scores[:keep_n]}
    
    print(f"[FILTER] Pattern filtering: {len(patterns)} -> {len(top_names)} (top {keep_percentage*100:.0f}%)")
    
    return {name: patterns[name] for name in top_names}


def generate_shift_patterns(demand_matrix, *, top_k=20, cfg=None):
    cfg = cfg or {}
    active_days = cfg.get("ACTIVE_DAYS", range(demand_matrix.shape[0]))
    allow_8h = cfg.get("allow_8h", True)
    H = demand_matrix.shape[1]
    shift_len = 8 if allow_8h else 8
    dm_packed = np.packbits(demand_matrix > 0, axis=1).astype(np.uint8)
    patterns = []
    for day in active_days:
        for start in range(H - shift_len + 1):
            mat = np.zeros_like(demand_matrix)
            mat[day, start : start + shift_len] = 1
            packed = np.packbits(mat > 0, axis=1).astype(np.uint8)
            score = score_pattern(packed, dm_packed)
            pid = f"D{day}_H{start}"
            patterns.append((score, pid, packed))
    patterns.sort(key=lambda x: x[0], reverse=True)
    return patterns[:top_k]


def generate_shifts_coverage_corrected(demand_matrix=None, *, cfg=None):
    cfg = cfg or {}
    demand = demand_matrix if demand_matrix is not None else np.zeros((7,24))
    active_days = [d for d in range(7) if demand[d].sum() > 0]
    
    # Determine first and last hour with demand - use full range without 6-20 restriction
    nonzero_hours = np.where(demand.sum(axis=0) > 0)[0]
    first_hour = int(nonzero_hours[0]) if len(nonzero_hours) > 0 else 0
    last_hour = int(nonzero_hours[-1] + 1) if len(nonzero_hours) > 0 else 24
    # Allow starts from first hour with demand to last (no 6-20 clipping)
    start_h_min = first_hour 
    start_h_max = last_hour 
    base_start_hours = np.arange(start_h_min, start_h_max, 0.5)
    
    use_ft = cfg.get("use_ft", True)
    use_pt = cfg.get("use_pt", True)
    allow_8h = cfg.get("allow_8h", True)
    allow_10h8 = cfg.get("allow_10h8", True)
    allow_pt_4h = cfg.get("allow_pt_4h", True)
    allow_pt_6h = cfg.get("allow_pt_6h", True)
    allow_pt_5h = cfg.get("allow_pt_5h", False)
    
    shifts = {}
    seen_patterns = set()
    
    def add_pattern(name, pattern):
        key = hashlib.md5(pattern).digest()
        if key not in seen_patterns:
            seen_patterns.add(key)
            shifts[name] = pattern
            return True
        return False

    # FT 8H: Generate patterns for all active days (even if < 6)
    if use_ft and allow_8h and active_days:
        for start in base_start_hours:  # every 30 minutes across full demand range
            if len(active_days) >= 6:
                # Generate all 6-day work combinations (one day off)
                for dso in active_days + ([None] if len(active_days) == 6 else []):
                    wd = [d for d in active_days if d != dso][:6]
                    if len(wd) >= 6:
                        break_options = get_valid_break_times(start, 8, cfg)
                        for brk in break_options:
                            pattern = generate_weekly_pattern_with_break(start, 8, wd, dso, brk)
                            dso_suffix = f"_DSO{dso}" if dso is not None else ""
                            shift_name = f"FT8_{start:04.1f}_DAYS{''.join(map(str, wd))}_BRK{brk:04.1f}{dso_suffix}"
                            add_pattern(shift_name, pattern)
            else:
                # Generate pattern working ALL days with demand (<=5 days)
                wd = list(active_days)  # use all active days
                break_options = get_valid_break_times(start, 8, cfg)
                for brk in break_options:
                    pattern = generate_weekly_pattern_with_break(start, 8, wd, None, brk)
                    shift_name = f"FT8_{start:04.1f}_DAYS{''.join(map(str, wd))}_BRK{brk:04.1f}"
                    add_pattern(shift_name, pattern)

    # FT 10H8: 5 working days, 4 days of 10h + 1 day of 8h (48h/week), with break
    if use_ft and allow_10h8 and len(active_days) >= 5:
        for start in base_start_hours[::2]:  # every 1.0 hour (skip half-hour starts)
            for dso in active_days:
                working_days = sorted([d for d in active_days if d != dso])[:5]
                if len(working_days) >= 5:
                    for eight_day in working_days:
                        durations = [8 if d == eight_day else 10 for d in working_days]
                        pattern = _build_pattern(working_days, durations, start, 1,
                                                 cfg.get("break_from_start", 2.0),
                                                 cfg.get("break_from_end", 2.0))
                        shift_name = f"FT10p8_{start:04.1f}_DSO{dso}_8{eight_day}"
                        add_pattern(shift_name, pattern)

    from itertools import combinations, permutations
    
    # PT 4H: allow 4h shifts with all possible working days (<=24h/week)
    if use_pt and allow_pt_4h:
        for start in base_start_hours[::2]:  # every 1 hour
            for num_days in range(1, len(active_days)+1):
                if 4 * num_days <= 24:
                    for days_combo in combinations(active_days, num_days):
                        pattern = _build_pattern(list(days_combo), [4] * num_days, start, 0, 0, 0)
                        shift_name = f"PT4_{start:04.1f}_DAYS{''.join(str(d) for d in days_combo)}"
                        add_pattern(shift_name, pattern)

    # PT 6H: all possible working days (<=24h/week)
    if use_pt and allow_pt_6h:
        for start in base_start_hours[::3]:  # every 1.5 hours
            for num_days in range(1, len(active_days)+1):
                if 6 * num_days <= 24:
                    for days_combo in combinations(active_days, num_days):
                        pattern = _build_pattern(list(days_combo), [6] * num_days, start, 0, 0, 0)
                        shift_name = f"PT6_{start:04.1f}_DAYS{''.join(str(d) for d in days_combo)}"
                        add_pattern(shift_name, pattern)

    # PT 5H: 5h shifts with all possible working days (up to 25h/week)
    if use_pt and allow_pt_5h:
        for start in base_start_hours[::3]:  # every 1.5 hours
            for num_days in range(1, len(active_days)+1):
                if 5 * num_days <= 25:
                    for days_combo in combinations(active_days, num_days):
                        if 5 * num_days <= 24:
                            # Standard 5h pattern
                            pattern = _build_pattern(list(days_combo), [5] * num_days, start, 0, 0, 0)
                        else:
                            # Use PT5 pattern with mixed hours to stay within 24h
                            pattern = generate_weekly_pattern_pt5(start, list(days_combo))
                        shift_name = f"PT5_{start:04.1f}_DAYS{''.join(str(d) for d in days_combo)}"
                        add_pattern(shift_name, pattern)

    return shifts


def load_shift_patterns(template_cfg, demand_matrix=None, *, max_patterns=None, keep_percentage=0.3, slot_duration_minutes=60, smart_start_hours=False, cfg=None):
    """Carga patrones personalizados desde configuración JSON JEAN Personalizado"""
    cfg = cfg or {}
    shifts = template_cfg.get("shifts", [])
    if not shifts:
        return {}
    
    patterns = {}
    seen_patterns = set()
    
    def add_pattern(name, pattern):
        key = hashlib.md5(pattern).digest()
        if key not in seen_patterns:
            seen_patterns.add(key)
            patterns[name] = pattern
            return True
        return False
    
    # Generate start hours according to granularity
    slot_factor = 60 / slot_duration_minutes
    if smart_start_hours and demand_matrix is not None:
        # Prioritize hours with high demand
        hourly_demand = demand_matrix.sum(axis=0)
        peak_hours = np.where(hourly_demand >= np.percentile(hourly_demand[hourly_demand > 0], 60))[0]
        start_hours = [h + m/60.0 for h in peak_hours for m in range(0, 60, int(slot_duration_minutes))]
        # No operational range limit - allow all hours with demand
    else:
        # Standard hours every slot_duration_minutes - use full demand range
        nonzero_hours = np.where(demand_matrix.sum(axis=0) > 0)[0] if demand_matrix is not None else range(24)
        first_hour = int(nonzero_hours[0]) if len(nonzero_hours) > 0 else 0
        last_hour = int(nonzero_hours[-1] + 1) if len(nonzero_hours) > 0 else 24
        start_hours = [h + m/60.0 for h in range(first_hour, last_hour) for m in range(0, 60, int(slot_duration_minutes))]
    
    for shift_def in shifts:
        shift_name = shift_def.get("name", "CUSTOM")
        pattern_def = shift_def.get("pattern", {})
        break_def = shift_def.get("break", {})
        
        # Filtrar por toggles de UI
        if shift_name.startswith("FT") and not cfg.get("use_ft", True):
            continue
        if shift_name.startswith("PT") and not cfg.get("use_pt", True):
            continue
        
        # Días de trabajo
        work_days_def = pattern_def.get("work_days", [])
        if isinstance(work_days_def, int):
            # Generar todas las combinaciones de N días
            work_days_combos = list(combinations(range(7), work_days_def))
        else:
            # Usar días específicos
            work_days_combos = [work_days_def]
        
        # Segmentos de duración
        segments = pattern_def.get("segments", [])
        if isinstance(segments, list) and segments:
            # Si segments es lista de números, usar directamente
            if all(isinstance(s, (int, float)) for s in segments):
                duration_combos = [segments]
            # Si segments es lista de dicts con "hours" y "count"
            elif all(isinstance(s, dict) for s in segments):
                duration_list = []
                for seg in segments:
                    hours = seg.get("hours", 8)
                    count = seg.get("count", 1)
                    duration_list.extend([hours] * count)
                duration_combos = list(permutations(duration_list))
            else:
                duration_combos = [segments]
        else:
            # Fallback: 8 horas por defecto
            duration_combos = [[8] * len(work_days_combos[0]) if work_days_combos else [8]]
        
        # Configuración de break
        break_enabled = break_def.get("enabled", True) if isinstance(break_def, dict) else bool(break_def)
        break_len = break_def.get("length_minutes", 60) / 60.0 if break_enabled else 0
        break_from_start = break_def.get("earliest_after_start", 120) / 60.0
        break_from_end = break_def.get("latest_before_end", 120) / 60.0
        
        # Generar patrones para cada combinación
        pattern_count = 0
        for work_days in work_days_combos:
            for durations in duration_combos:
                # Validar horas semanales
                weekly_hours = sum(durations)
                if shift_name.startswith("FT") and weekly_hours > 48:
                    continue
                if shift_name.startswith("PT") and weekly_hours > 24:
                    continue
                
                for start_hour in start_hours:
                    if max_patterns and pattern_count >= max_patterns:
                        break
                    
                    try:
                        pat = _build_pattern(
                            list(work_days), list(durations), start_hour,
                            break_len, break_from_start, break_from_end, slot_factor
                        )
                        
                        name = f"{shift_name}_{start_hour:04.1f}_{''.join(str(d) for d in work_days)}"
                        if add_pattern(name, pat):
                            pattern_count += 1
                    except Exception as e:
                        print(f"[CUSTOM] Error generating pattern {shift_name}: {e}")
                        continue
                
                if max_patterns and pattern_count >= max_patterns:
                    break
            if max_patterns and pattern_count >= max_patterns:
                break
    
    print(f"[CUSTOM] Custom patterns generated: {len(patterns)}")
    
    # Apply score filtering if specified
    if keep_percentage < 1.0 and demand_matrix is not None:
        patterns = score_and_filter_patterns(
            patterns, demand_matrix, keep_percentage=keep_percentage
        )
    
    return patterns


def get_smart_start_hours(demand_matrix, slot_duration_minutes=60):
    """Get smart start hours based on demand - no 8-20 restriction like APP1"""
    hourly_demand = demand_matrix.sum(axis=0)
    # Hours with demand >= percentile 60
    threshold = np.percentile(hourly_demand[hourly_demand > 0], 60) if np.any(hourly_demand > 0) else 0
    peak_hours = np.where(hourly_demand >= threshold)[0]
    
    # Generate slots every slot_duration_minutes
    start_hours = []
    for h in peak_hours:
        for m in range(0, 60, int(slot_duration_minutes)):
            start_hour = h + m/60.0
            # No operational range restriction - allow night/early morning shifts
            start_hours.append(start_hour)
    
    return sorted(set(start_hours))


def generate_shifts_coverage_optimized(
    demand_matrix,
    *,
    max_patterns=None,
    quality_threshold=0,
    allow_8h=True,
    allow_10h8=False,
    allow_pt_4h=True,
    allow_pt_5h=True,
    allow_pt_6h=False,
    batch_size=2000,
    keep_percentage=0.3,
    peak_bonus=1.5,
    critical_bonus=2.0,
    efficiency_bonus=1.0,
):
    """Genera patrones optimizados usando _build_pattern con detección de duplicados"""
    cfg = {
        "use_ft": allow_8h or allow_10h8,
        "use_pt": allow_pt_4h or allow_pt_5h or allow_pt_6h,
        "allow_8h": allow_8h,
        "allow_10h8": allow_10h8,
        "allow_pt_4h": allow_pt_4h,
        "allow_pt_5h": allow_pt_5h,
        "allow_pt_6h": allow_pt_6h,
    }
    
    # Generar patrones con detección de duplicados integrada
    raw = generate_shifts_coverage_corrected(demand_matrix, cfg=cfg)
    
    # Filtrar por calidad mínima
    quality_patterns = {}
    for name, pat in raw.items():
        if score_pattern(pat, demand_matrix) >= quality_threshold:
            quality_patterns[name] = pat
    
    print(f"[GEN] Patrones únicos generados: {len(quality_patterns)}")
    
    # Apply intelligent score filtering only if keep_percentage < 1.0
    if keep_percentage < 1.0:
        filtered_patterns = score_and_filter_patterns(
            quality_patterns, demand_matrix, 
            keep_percentage=keep_percentage,
            peak_bonus=peak_bonus,
            critical_bonus=critical_bonus,
            efficiency_bonus=efficiency_bonus
        )
    else:
        filtered_patterns = quality_patterns
    
    # Entregar en batches
    selected = 0
    batch = {}
    for name, pat in filtered_patterns.items():
        batch[name] = pat
        selected += 1
        if len(batch) >= batch_size:
            yield batch
            batch = {}
        if max_patterns and selected >= max_patterns:
            break
    if batch:
        yield batch


def memory_limit_patterns(slots_per_day: int) -> int:
    """Return how many patterns fit in roughly 4GB of RAM."""
    if slots_per_day <= 0:
        return 0
    if psutil is None:
        return 10000
    available = psutil.virtual_memory().available
    cap = min(available, 4 * 1024 ** 3)
    return int(cap // (7 * slots_per_day))

def monitor_memory_usage() -> float:
    """Return current memory usage percentage."""
    if psutil is None:
        return 0.0
    return psutil.virtual_memory().percent

def adaptive_chunk_size(base: int = 5000) -> int:
    """Return dynamic chunk size based on memory pressure."""
    if psutil is None:
        return base
    usage = monitor_memory_usage()
    if usage > 80:
        return max(1000, base // 4)
    if usage > 60:
        return max(2000, base // 2)
    return base

def emergency_cleanup(threshold: float = 85.0) -> bool:
    """Trigger garbage collection if memory usage exceeds threshold."""
    if psutil is None:
        return False
    if monitor_memory_usage() >= threshold:
        gc.collect()
        return True
    return False


def save_execution_result(demand_matrix, cfg, coverage_percentage, total_agents, execution_time, overstaffing=0, understaffing=0):
    """Save execution results for adaptive learning."""
    import json
    import os
    from datetime import datetime
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    history_file = os.path.join(data_dir, "adaptive_history.json")
    
    # Load existing history
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception:
            history = []
    
    # Create new result entry
    result = {
        "timestamp": datetime.now().isoformat(),
        "demand_total": int(demand_matrix.sum()),
        "demand_peak": int(demand_matrix.max()),
        "agent_limit_factor": cfg.get("agent_limit_factor", 30),
        "excess_penalty": cfg.get("excess_penalty", 0.5),
        "peak_bonus": cfg.get("peak_bonus", 0.75),
        "critical_bonus": cfg.get("critical_bonus", 1.0),
        "coverage_percentage": coverage_percentage,
        "total_agents": total_agents,
        "execution_time": execution_time,
        "overstaffing": overstaffing,
        "understaffing": understaffing,
        "score": overstaffing + understaffing
    }
    
    # Add to history (keep last 50 results)
    history.append(result)
    history = history[-50:]
    
    # Save updated history
    try:
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"[ADAPTIVE] Result saved: coverage {coverage_percentage:.1f}%, agents {total_agents}")
    except Exception as e:
        print(f"[ADAPTIVE] Error saving result: {e}")


def get_adaptive_params(demand_matrix):
    """Get adaptive parameters based on execution history."""
    import json
    import os
    
    history_file = os.path.join("data", "adaptive_history.json")
    
    # Default parameters
    default_params = {
        "agent_limit_factor": 25,
        "excess_penalty": 2.0,
        "peak_bonus": 1.5,
        "critical_bonus": 2.0
    }
    
    if not os.path.exists(history_file):
        print("[ADAPTIVE] No previous history, using default parameters")
        return default_params
    
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except Exception:
        print("[ADAPTIVE] Error reading history, using default parameters")
        return default_params
    
    if not history:
        return default_params
    
    # Find similar demand patterns (±20% total demand)
    current_total = demand_matrix.sum()
    current_peak = demand_matrix.max()
    
    similar_results = []
    for result in history:
        demand_diff = abs(result["demand_total"] - current_total) / max(current_total, 1)
        peak_diff = abs(result["demand_peak"] - current_peak) / max(current_peak, 1)
        
        if demand_diff <= 0.2 and peak_diff <= 0.3:  # Similar patterns
            similar_results.append(result)
    
    if not similar_results:
        # Use best overall results
        similar_results = sorted(history, key=lambda x: x["score"])[:5]
    
    if not similar_results:
        return default_params
    
    # Find best performing parameters
    best_result = min(similar_results, key=lambda x: x["score"])
    
    adaptive_params = {
        "agent_limit_factor": best_result["agent_limit_factor"],
        "excess_penalty": best_result["excess_penalty"],
        "peak_bonus": best_result["peak_bonus"],
        "critical_bonus": best_result["critical_bonus"]
    }
    
    print(f"[ADAPTIVE] Adaptive parameters based on {len(similar_results)} similar results")
    print(f"[ADAPTIVE] Factor: {adaptive_params['agent_limit_factor']}, Excess: {adaptive_params['excess_penalty']:.1f}")
    
    return adaptive_params


def solve_in_chunks_optimized(shifts_coverage, demand_matrix, base_chunk_size=10000, **kwargs):
    """Optimización por chunks con soporte para restricciones JEAN"""
    cfg = kwargs
    agent_cap = cfg.get("agent_cap")
    allow_excess = cfg.get("allow_excess", True)
    
    scored = []
    seen = set()
    for name, pat in shifts_coverage.items():
        key = hashlib.md5(pat).digest()
        if key in seen:
            continue
        seen.add(key)
        scored.append((name, pat, score_pattern(pat, demand_matrix)))
    scored.sort(key=lambda x: x[2], reverse=True)
    
    # Filtrar por score si hay muchos patrones
    if len(scored) > 8000:
        keep = int(len(scored) * 0.6)
        scored = scored[:keep]

    assignments_total = {}
    coverage = np.zeros_like(demand_matrix)
    days = demand_matrix.shape[0]
    total_agents = 0
    idx = 0
    
    while idx < len(scored):
        # Verificar límite de agentes JEAN
        if agent_cap and total_agents >= agent_cap:
            print(f"[CHUNKS] Límite de agentes alcanzado: {total_agents}/{agent_cap}")
            break
            
        chunk_size = adaptive_chunk_size(base_chunk_size)
        chunk = {n: p for n, p, _ in scored[idx : idx + chunk_size]}
        
        # Calcular demanda restante
        if allow_excess:
            remaining = np.maximum(0, demand_matrix - coverage)
        else:
            # Modo JEAN: evitar exceso
            remaining = np.clip(demand_matrix - coverage, 0, None)
            
        if not np.any(remaining):
            break
            
        # Configuración para el chunk
        chunk_cfg = cfg.copy()
        if agent_cap:
            chunk_cfg["agent_limit_factor"] = max(5, (agent_cap - total_agents) // max(1, len(chunk)))
            
        assigns, _ = optimize_portfolio(chunk, remaining, cfg=chunk_cfg)
        
        for n, v in assigns.items():
            assignments_total[n] = assignments_total.get(n, 0) + v
            slots = len(chunk[n]) // days
            coverage += chunk[n].reshape(days, slots) * v
            total_agents += v
            
            # Verificar límite de agentes
            if agent_cap and total_agents >= agent_cap:
                break
                
        idx += chunk_size
        gc.collect()
        
        # Verificar si se completó la cobertura
        if not np.any(np.maximum(0, demand_matrix - coverage)):
            break
            
    print(f"[CHUNKS] Completado: {total_agents} agentes asignados")
    # Logs de métricas para chunks
    if assignments_total:
        chunk_results = analyze_results(assignments_total, {n: p for n, p, _ in scored}, demand_matrix)
        if chunk_results:
            print(f"[MÉTRICAS CHUNKS] Cobertura_real={chunk_results.get('coverage_real', 0):.1f}%, Cobertura_pura={chunk_results['coverage_percentage']:.1f}%, Exceso={chunk_results['overstaffing']}, Déficit={chunk_results['understaffing']}, Agentes={chunk_results['total_agents']}")
    return assignments_total


def optimize_with_precision_targeting(
    shifts_coverage,
    demand_matrix,
    *,
    cfg=None,
    iteration_time_limit=None,
    agent_limit_factor=30,
    excess_penalty=0,
    peak_bonus=0,
    critical_bonus=0,
    TIME_SOLVER=None,
):
    if not PULP_AVAILABLE:
        return {}, "NO_PULP"
    cfg = cfg or {}
    shifts_list = list(shifts_coverage.keys())
    prob = pulp.LpProblem("PrecisionTargeting", pulp.LpMinimize)

    D, H = demand_matrix.shape
    total_dem = int(demand_matrix.sum())
    peak_dem = int(demand_matrix.max())

    shift_vars = {}
    for i, s in enumerate(shifts_list):
        pat = np.array(shifts_coverage[s])
        pat_2d = pat.reshape(D, len(pat) // D)
        pat_hours = int(pat_2d.sum())
        
        # Límite dinámico por patrón
        if cfg.get("agent_cap"):
            # En modo JEAN, límite más restrictivo
            ub = max(1, min(cfg["agent_cap"] // max(1, len(shifts_list) // 10), 
                           int(np.ceil(total_dem / max(pat_hours, 1)))))
        else:
            ub = max(0, min(max(3, peak_dem), int(np.ceil(total_dem / max(pat_hours, 1)))))
        
        shift_vars[s] = pulp.LpVariable(f"x_{i}", 0, ub, pulp.LpInteger)

    deficit_vars = {
        (d, h): pulp.LpVariable(f"def_{d}_{h}", 0, None)
        for d in range(D)
        for h in range(H)
    }
    excess_vars = {
        (d, h): pulp.LpVariable(f"exc_{d}_{h}", 0, None)
        for d in range(D)
        for h in range(H)
    }

    # --- DYNAMIC WEIGHTS BY HOUR/DAY (from actual demand, not config) ---
    daily_tot = demand_matrix.sum(axis=1)
    hourly_tot = demand_matrix.sum(axis=0)
    # Identify critical day and peak hour from actual demand
    critical_day = int(np.argmax(daily_tot)) 
    peak_hour = int(np.argmax(hourly_tot))

    ex_pen = float(excess_penalty or 0.5)
    pk_bo = float(peak_bonus or 0.75)
    cr_bo = float(critical_bonus or 1.0)

    W_DEF_BASE = 1000.0
    W_EXC_BASE = 10.0 * ex_pen
    W_AGENTS = 0.01

    w_def, w_exc = {}, {}
    for d in range(D):
        for h in range(H):
            # Increase deficit weight on critical day and peak hour
            w_d = W_DEF_BASE
            if d == critical_day: 
                w_d *= (1.0 + cr_bo) 
            if h == peak_hour: 
                w_d *= (1.0 + pk_bo)

            if demand_matrix[d, h] <= 0:
                w_e = W_EXC_BASE * 200.0
            elif demand_matrix[d, h] <= 2:
                w_e = W_EXC_BASE * 50.0
            elif demand_matrix[d, h] <= 5:
                w_e = W_EXC_BASE * 10.0
            else:
                w_e = W_EXC_BASE * 4.0

            w_def[(d, h)] = w_d
            w_exc[(d, h)] = w_e

    total_agents = pulp.lpSum(shift_vars.values())
    prob += (
        pulp.lpSum(w_def[(d, h)] * deficit_vars[(d, h)] for d in range(D) for h in range(H))
        + pulp.lpSum(w_exc[(d, h)] * excess_vars[(d, h)] for d in range(D) for h in range(H))
        + W_AGENTS * total_agents
    )

    for d in range(D):
        for h in range(H):
            cov = pulp.lpSum(
                [
                    shift_vars[s]
                    * np.array(shifts_coverage[s]).reshape(D, len(shifts_coverage[s]) // D)[d, h]
                    for s in shifts_list
                ]
            )
            prob += cov + deficit_vars[(d, h)] - excess_vars[(d, h)] == int(demand_matrix[d, h])
    
    # Restricciones JEAN: sin exceso si está configurado
    if cfg.get("allow_excess") == False:
        for d in range(D):
            for h in range(H):
                cov = pulp.lpSum(
                    [
                        shift_vars[s]
                        * np.array(shifts_coverage[s]).reshape(D, len(shifts_coverage[s]) // D)[d, h]
                        for s in shifts_list
                    ]
                )
                prob += cov <= int(demand_matrix[d, h])  # Sin exceso
    
    # Control de exceso global más flexible – permitir hasta 15% de exceso
    profile_name = cfg.get("optimization_profile", "")
    if "JEAN" in profile_name:
        total_excess = pulp.lpSum(excess_vars.values())
        prob += total_excess <= demand_matrix.sum() * 0.15  # 15% exceso permitido
    
    # Límite de agentes total si está configurado
    if cfg.get("agent_cap"):
        prob += total_agents <= int(cfg["agent_cap"])

    time_limit = (
        TIME_SOLVER
        if TIME_SOLVER is not None
        else (
            iteration_time_limit if iteration_time_limit is not None else cfg.get("solver_time", 240)
        )
    )
    solver = _build_cbc_solver({**cfg, "solver_time": time_limit})
    prob.solve(solver)
    nodes = None
    solver_model = getattr(prob, "solverModel", None)
    if solver_model is not None:
        nodes = getattr(solver_model, "nodeCount", None)
        if nodes is None and hasattr(solver_model, "getNodeCount"):
            try:
                nodes = solver_model.getNodeCount()
            except Exception:
                nodes = None
    if nodes == 0:
        print("[DIAG] CBC devolvió 0 nodos. Posible objetivo plano/relajación trivial.")
        print(f"[DIAG] pesos: ex_pen={excess_penalty}, peak={peak_bonus}, critical={critical_bonus}")
        print(
            f"[DIAG] demanda total={int(demand_matrix.sum())}, pico={int(demand_matrix.max())}"
        )

    assignments = {}
    if prob.status == pulp.LpStatusOptimal:
        for s in shifts_list:
            v = int(shift_vars[s].varValue or 0)
            if v > 0:
                assignments[s] = v
        # Logs de métricas para precision targeting
        if assignments:
            pt_results = analyze_results(assignments, shifts_coverage, demand_matrix)
            if pt_results:
                print(f"[MÉTRICAS PT] Cobertura_real={pt_results.get('coverage_real', 0):.1f}%, Cobertura_pura={pt_results['coverage_percentage']:.1f}%, Exceso={pt_results['overstaffing']}, Déficit={pt_results['understaffing']}, Agentes={pt_results['total_agents']}")
        return assignments, "PRECISION_TARGETING"
    return {}, f"STATUS_{prob.status}"


def _evaluate(assigns, shifts_coverage, demand_matrix):
    import numpy as np
    if not assigns:
        return float("inf")
    D = demand_matrix.shape[0]
    cov = np.zeros_like(demand_matrix)
    for s, c in assigns.items():
        slots = len(shifts_coverage[s]) // D
        cov += np.array(shifts_coverage[s]).reshape(D, slots) * int(c)
    deficit = np.maximum(0, demand_matrix - cov).sum()
    excess = np.maximum(0, cov - demand_matrix).sum()
    return deficit * 1000 + excess * 10


def optimize_portfolio(shifts_coverage, demand_matrix, cfg=None):
    cfg = (cfg or {}).copy()
    candidates = []
    for seed in (1, 11, 21, 31):
        for factor in (
            cfg.get("agent_limit_factor", 30),
            max(10, cfg.get("agent_limit_factor", 30) // 2),
        ):
            local = cfg.copy()
            local["random_seed"] = seed
            local["agent_limit_factor"] = factor
            assigns, _ = optimize_with_precision_targeting(
                shifts_coverage,
                demand_matrix,
                cfg=local,
                iteration_time_limit=local.get("solver_time", 240),
                agent_limit_factor=factor,
                excess_penalty=local.get("excess_penalty", 0.5),
                peak_bonus=local.get("peak_bonus", 0.75),
                critical_bonus=local.get("critical_bonus", 1.0),
            )
            candidates.append(
                (_evaluate(assigns, shifts_coverage, demand_matrix), assigns, seed, factor)
            )
    candidates.sort(key=lambda x: x[0])
    if candidates:
        _, best, seed, factor = candidates[0]
        return best, f"seed={seed}_factor={factor}"
    return {}, "PORTFOLIO_EMPTY"


def optimize_ft_then_pt_strategy(
    shifts_coverage,
    demand_matrix,
    *,
    agent_limit_factor=30,
    excess_penalty=0,
    TIME_SOLVER=None,
):
    ft = {k: v for k, v in shifts_coverage.items() if k.startswith("FT")}
    pt = {k: v for k, v in shifts_coverage.items() if k.startswith("PT")}
    ft_ass, _ = optimize_with_precision_targeting(
        ft,
        demand_matrix,
        agent_limit_factor=agent_limit_factor,
        excess_penalty=excess_penalty,
        peak_bonus=0,
        critical_bonus=0,
        TIME_SOLVER=TIME_SOLVER,
    )
    cov_ft = np.zeros_like(demand_matrix)
    for s, c in ft_ass.items():
        slots = len(ft[s]) // 7
        cov_ft += ft[s].reshape(7, slots) * c
    remaining = np.maximum(0, demand_matrix - cov_ft)
    pt_ass, _ = optimize_with_precision_targeting(
        pt,
        remaining,
        agent_limit_factor=agent_limit_factor,
        excess_penalty=excess_penalty,
        peak_bonus=0,
        critical_bonus=0,
        TIME_SOLVER=TIME_SOLVER,
    )
    return {**ft_ass, **pt_ass}, "FT_PT"


def optimize_ft_no_excess(ft_shifts, demand_matrix, cfg=None):
    """Fase 1: Optimiza FT sin permitir exceso (cobertura <= demanda)."""
    if not ft_shifts:
        return {}
    cfg = cfg or {}
    prob = pulp.LpProblem("FT_No_Excess", pulp.LpMinimize)
    # Variables FT con límite dinámico
    total_demand = demand_matrix.sum()
    max_ft_per_shift = max(10, int(total_demand / cfg.get("agent_limit_factor", 30)))
    ft_vars = {shift: pulp.LpVariable(f"ft_{shift}", 0, max_ft_per_shift, pulp.LpInteger) for shift in ft_shifts}
    # Déficit (no exceso)
    hours = demand_matrix.shape[1]
    deficit = {(d, h): pulp.LpVariable(f"ft_def_{d}_{h}", 0, None) for d in range(7) for h in range(hours)}
    # Objetivo: minimizar déficit + número de agentes
    total_deficit = pulp.lpSum(deficit.values())
    total_agents = pulp.lpSum(ft_vars.values())
    prob += total_deficit * 1000 + total_agents * 1
    # Restricciones: cobertura + déficit >= demanda, y cobertura <= demanda (no exceso)
    for d in range(7):
        for h in range(hours):
            coverage = pulp.lpSum(ft_vars[s] * (ft_shifts[s][d * hours + h]) for s in ft_shifts)
            demand = int(demand_matrix[d, h])
            prob += coverage + deficit[(d, h)] >= demand
            prob += coverage <= demand
    # Resolver (mitad del tiempo asignado al solver)
    solver = _build_cbc_solver({**cfg, "solver_time": int(cfg.get("solver_time", 240)//2)})
    prob.solve(solver)
    ft_assignments = {}
    if prob.status == pulp.LpStatusOptimal:
        for s, var in ft_vars.items():
            val = int(var.value() or 0)
            if val > 0:
                ft_assignments[s] = val
    return ft_assignments

def optimize_pt_complete(pt_shifts, remaining_demand, cfg=None):
    """Fase 2: PT para completar el déficit restante (permite mínimo exceso controlado)."""
    if not pt_shifts or remaining_demand.sum() == 0:
        return {}
    cfg = cfg or {}
    prob = pulp.LpProblem("PT_Complete", pulp.LpMinimize)
    # Variables PT con límite dinámico
    total_rem = remaining_demand.sum()
    max_pt_per_shift = max(10, int(total_rem / max(1, cfg.get("agent_limit_factor", 30))))
    pt_vars = {shift: pulp.LpVariable(f"pt_{shift}", 0, max_pt_per_shift, pulp.LpInteger) for shift in pt_shifts}
    # Variables de déficit y exceso
    hours = remaining_demand.shape[1]
    deficit = {(d, h): pulp.LpVariable(f"pt_def_{d}_{h}", 0, None) for d in range(7) for h in range(hours)}
    excess = {(d, h): pulp.LpVariable(f"pt_exc_{d}_{h}", 0, None) for d in range(7) for h in range(hours)}
    # Objetivo: minimizar déficit, penalizar exceso, minimizar agentes
    total_deficit = pulp.lpSum(deficit.values())
    total_excess = pulp.lpSum(excess.values())
    total_agents = pulp.lpSum(pt_vars.values())
    excess_pen = cfg.get("excess_penalty", 5.0)
    prob += total_deficit * 1000 + total_excess * (excess_pen * 20) + total_agents * 1
    # Prohibir exceso para perfil JEAN
    profile = (cfg.get("optimization_profile") or "").upper()
    if profile.startswith("JEAN"):
        prob += total_excess == 0
    # Restricciones: cubrir demanda con posible déficit/exceso
    for d in range(7):
        for h in range(hours):
            cov = pulp.lpSum(pt_vars[s] * (pt_shifts[s][d * hours + h]) for s in pt_shifts)
            demand = int(remaining_demand[d, h])
            prob += cov + deficit[(d, h)] >= demand
            prob += cov - excess[(d, h)] <= demand
    # Resolver (segunda mitad del tiempo)
    solver = _build_cbc_solver({**cfg, "solver_time": int(cfg.get("solver_time", 240)//2)})
    prob.solve(solver)
    pt_assignments = {}
    if prob.status == pulp.LpStatusOptimal:
        for s, var in pt_vars.items():
            val = int(var.value() or 0)
            if val > 0:
                pt_assignments[s] = val
    return pt_assignments

def optimize_ft_then_pt_strategy(shifts_coverage, demand_matrix, cfg=None):
    """Estrategia en 2 fases: primero FT sin exceso, luego PT para déficit."""
    cfg = cfg or {}
    # Separar patrones FT y PT
    ft_shifts = {k: v for k, v in shifts_coverage.items() if k.startswith('FT')}
    pt_shifts = {k: v for k, v in shifts_coverage.items() if k.startswith('PT')}
    # Fase 1: asignación FT sin exceso
    ft_assignments = optimize_ft_no_excess(ft_shifts, demand_matrix, cfg)
    # Calcular cobertura FT obtenida
    ft_coverage = np.zeros_like(demand_matrix, dtype=int)
    for name, count in ft_assignments.items():
        pattern = np.array(ft_shifts[name]).reshape(7, -1)
        ft_coverage += pattern * count
    # Fase 2: asignación PT para cubrir déficit restante
    remaining = np.maximum(0, demand_matrix - ft_coverage)
    pt_assignments = optimize_pt_complete(pt_shifts, remaining, cfg)
    # Combinar resultados
    assignments = {**ft_assignments, **pt_assignments}
    return assignments, "FT_NO_EXCESS_THEN_PT"

def optimize_maximum_coverage(shifts_coverage, demand_matrix, cfg=None):
    """Optimizador de cobertura máxima que elimina el déficit ante todo."""
    if not PULP_AVAILABLE:
        return {}, "NO_PULP"
    
    cfg = cfg or {}
    shifts_list = list(shifts_coverage.keys())
    prob = pulp.LpProblem("MaximumCoverage", pulp.LpMinimize)
    
    D, H = demand_matrix.shape
    total_dem = int(demand_matrix.sum())
    
    # Variables de turnos con límites más conservadores
    shift_vars = {}
    for i, s in enumerate(shifts_list):
        pat = np.array(shifts_coverage[s])
        pat_2d = pat.reshape(D, len(pat) // D)
        pat_hours = int(pat_2d.sum())
        
        # Límite conservador para controlar exceso
        ub = max(1, int(np.ceil(total_dem / max(pat_hours, 1) / cfg.get("agent_limit_factor", 15))))
        shift_vars[s] = pulp.LpVariable(f"x_{i}", 0, ub, pulp.LpInteger)
    
    # Variables de déficit y exceso
    deficit_vars = {(d, h): pulp.LpVariable(f"def_{d}_{h}", 0, None) for d in range(D) for h in range(H)}
    excess_vars = {(d, h): pulp.LpVariable(f"exc_{d}_{h}", 0, None) for d in range(D) for h in range(H)}
    
    # Objetivo: eliminar déficit ante todo (peso enorme), minimizar exceso y agentes
    total_deficit = pulp.lpSum(deficit_vars.values())
    total_excess = pulp.lpSum(excess_vars.values())
    total_agents = pulp.lpSum(shift_vars.values())
    
    deficit_penalty = cfg.get("deficit_penalty", 1000.0)
    excess_penalty = cfg.get("excess_penalty", 1.0)
    
    prob += (total_deficit * deficit_penalty + 
             total_excess * excess_penalty + 
             total_agents * 0.01)
    
    # Restricciones de cobertura
    for d in range(D):
        for h in range(H):
            cov = pulp.lpSum([
                shift_vars[s] * np.array(shifts_coverage[s]).reshape(D, len(shifts_coverage[s]) // D)[d, h]
                for s in shifts_list
            ])
            prob += cov + deficit_vars[(d, h)] - excess_vars[(d, h)] == int(demand_matrix[d, h])
    
    # Resolver
    solver = _build_cbc_solver(cfg)
    prob.solve(solver)
    
    assignments = {}
    if prob.status == pulp.LpStatusOptimal:
        for s in shifts_list:
            v = int(shift_vars[s].varValue or 0)
            if v > 0:
                assignments[s] = v
        # Logs de métricas para maximum coverage
        if assignments:
            mc_results = analyze_results(assignments, shifts_coverage, demand_matrix)
            if mc_results:
                print(f"[MÉTRICAS MC] Cobertura_real={mc_results.get('coverage_real', 0):.1f}%, Cobertura_pura={mc_results['coverage_percentage']:.1f}%, Exceso={mc_results['overstaffing']}, Déficit={mc_results['understaffing']}, Agentes={mc_results['total_agents']}")
        return assignments, "MAXIMUM_COVERAGE"
    
    return {}, f"STATUS_{prob.status}"


def optimize_jean_search(
    shifts_coverage,
    demand_matrix,
    *,
    target_coverage=100.0,  # Cambio a 100% para priorizar cobertura real
    agent_limit_factor=15,
    excess_penalty=5.0,
    peak_bonus=2.0,
    critical_bonus=2.5,
    iteration_time_limit=None,
    max_iterations=10,
    verbose=False,
    cfg=None,
    job_id=None,
    **kwargs,
):
    """Búsqueda JEAN exacta del original: 2 fases con caps progresivos"""
    import math
    cfg = {**(cfg or {}), **kwargs}
    
    # Verificar que el perfil JEAN tenga agent_limit_factor correcto (15 por defecto)
    if cfg.get("agent_limit_factor") is None:
        cfg["agent_limit_factor"] = 15  # Valor por defecto JEAN
    
    print("[JEAN] Iniciando búsqueda JEAN de 2 fases (exacta del original)")
    
    # 1) FASE BASE: solución sin exceso para establecer baseline
    print("[JEAN] Fase 1: Solución base sin exceso")
    base_cfg = cfg.copy()
    base_cfg["allow_excess"] = False  # Sin exceso en fase 1
    base_cfg["allow_deficit"] = True
    base_cfg["optimization_profile"] = cfg.get("optimization_profile", "JEAN")
    
    if PULP_AVAILABLE:
        base_assignments, _ = optimize_with_precision_targeting(
            shifts_coverage, demand_matrix, 
            cfg=base_cfg,
            excess_penalty=cfg.get("excess_penalty", 5.0),
            peak_bonus=cfg.get("peak_bonus", 2.0),
            critical_bonus=cfg.get("critical_bonus", 2.5)
        )
    else:
        base_assignments = solve_in_chunks_optimized(
            shifts_coverage, demand_matrix, **base_cfg
        )
    
    # 2) FASE PRECISIÓN: caps progresivos con exceso controlado (<=10%)
    total_demand = float(demand_matrix.sum())
    peak_demand = float(demand_matrix.max())
    
    print(f"[JEAN] Fase 2: Búsqueda con exceso controlado (<=10%)")
    
    # Initialize tracking variables for complete iteration
    best_assignments = base_assignments
    best_method = ""
    best_score = float("inf")
    best_coverage = 0
    best_target_score = float("inf")
    got_target_solution = False
    
    if base_assignments:
        base_results = analyze_results(base_assignments, shifts_coverage, demand_matrix)
        if base_results:
            best_score = base_results["overstaffing"] + base_results["understaffing"]
            best_coverage = base_results["coverage_real"]
            if best_coverage >= target_coverage:
                best_target_score = best_score
                got_target_solution = True
            print(f"[JEAN] Base solution without excess: coverage_real {best_coverage:.1f}%, score {best_score:.1f}")
            # Logs de métricas clave
            print(f"[MÉTRICAS] Cobertura_real={base_results.get('coverage_real', best_coverage):.1f}%, Cobertura_pura={best_coverage:.1f}%, Exceso={base_results['overstaffing']}, Déficit={base_results['understaffing']}, Agentes={base_results['total_agents']}")
    else:
        print("[JEAN] No base solution found without excess")
    
    # Factores progresivos con más iteraciones para mejor cobertura
    start_time = time.time()
    max_time = 180  # Máximo 3 minutos para JEAN
    
    # Generar secuencia de factores más agresiva
    factors = []
    current_factor = agent_limit_factor
    for _ in range(max_iterations):
        factors.append(int(current_factor))
        current_factor *= 0.85  # Factor más agresivo (0.85 vs 0.9)
        if current_factor < 5:
            break
    
    print(f"[JEAN] Secuencia de factores: {factors}")
    
    for iteration, factor in enumerate(factors):
        # Verificar timeout
        if time.time() - start_time > max_time:
            print(f"[JEAN] Timeout alcanzado ({max_time}s)")
            break
            
        agent_cap = max(1, int(total_demand / factor), int(peak_demand * 1.1))
        print(f"[JEAN] Iteración: factor {factor}, cap agentes {agent_cap}")
        
        # Configuración con exceso controlado (<=10% de demanda)
        temp_cfg = cfg.copy()
        temp_cfg["allow_excess"] = True  # Permitir exceso controlado
        temp_cfg["allow_deficit"] = True
        temp_cfg["agent_cap"] = agent_cap
        temp_cfg["agent_limit_factor"] = max(5, agent_limit_factor // 2)
        temp_cfg["optimization_profile"] = cfg.get("optimization_profile", "JEAN")
        
        if PULP_AVAILABLE:
            trial_assignments, _ = optimize_with_precision_targeting(
                shifts_coverage, demand_matrix, 
                cfg=temp_cfg,
                agent_limit_factor=temp_cfg["agent_limit_factor"],
                excess_penalty=excess_penalty,
                peak_bonus=peak_bonus,
                critical_bonus=critical_bonus
            )
        else:
            # Fallback greedy con límite de agentes
            trial_assignments = solve_in_chunks_optimized(
                shifts_coverage, demand_matrix, 
                base_chunk_size=min(5000, agent_cap * 50),
                **temp_cfg
            )
        
        if trial_assignments:
            trial_results = analyze_results(trial_assignments, shifts_coverage, demand_matrix)
            if trial_results:
                # Score JEAN ajustado: |100 - cobertura_real| + overstaffing + understaffing
                cov_real = trial_results["coverage_real"]
                score = abs(100.0 - cov_real) + trial_results["overstaffing"] + trial_results["understaffing"]
                
                print(f"[JEAN] Factor {factor}: coverage_real {cov_real:.1f}%, score {score:.1f}, agents {trial_results['total_agents']}")
                
                # Usar score ajustado para todas las comparaciones
                if not got_target_solution or score < best_target_score:
                    best_assignments = trial_assignments
                    best_method = f"JEAN_F{factor}"
                    best_target_score = score
                    best_coverage = cov_real
                    print(f"[JEAN] New best solution: coverage_real {cov_real:.1f}%, score {score:.1f}, agents {trial_results['total_agents']}")
                    # Logs de métricas clave para cada iteración
                    print(f"[MÉTRICAS] Cobertura_real={cov_real:.1f}%, Cobertura_pura={trial_results['coverage_percentage']:.1f}%, Exceso={trial_results['overstaffing']}, Déficit={trial_results['understaffing']}, Agentes={trial_results['total_agents']}")
                    if cov_real >= target_coverage:
                        got_target_solution = True
    
    elapsed = time.time() - start_time
    final_score = best_target_score if got_target_solution else best_score
    final_coverage = best_coverage if not got_target_solution else (analyze_results(best_assignments, shifts_coverage, demand_matrix) or {}).get("coverage_real", best_coverage)
    print(f"[JEAN] Completed in {elapsed:.1f}s: final score {final_score:.1f}, coverage_real {final_coverage:.1f}%")
    # Logs finales de métricas
    if best_assignments:
        final_results = analyze_results(best_assignments, shifts_coverage, demand_matrix)
        if final_results:
            print(f"[MÉTRICAS FINALES] Cobertura_real={final_results.get('coverage_real', final_coverage):.1f}%, Cobertura_pura={final_coverage:.1f}%, Exceso={final_results['overstaffing']}, Déficit={final_results['understaffing']}, Agentes={final_results['total_agents']}")
    return best_assignments, best_method or "JEAN_NO_SOLUTION"


def analyze_results(assignments, shifts_coverage, demand_matrix):
    if not assignments:
        return None
    slots = len(next(iter(shifts_coverage.values()))) // 7
    total_cov = np.zeros((7, slots), dtype=int)
    total_agents, ft_agents, pt_agents = 0, 0, 0
    for name, cnt in assignments.items():
        pat = np.array(shifts_coverage[name]).reshape(7, slots)
        total_cov += pat * cnt
        total_agents += cnt
        if name.startswith("FT"):
            ft_agents += cnt
        else:
            pt_agents += cnt
    total_demand = demand_matrix.sum()
    total_covered = np.minimum(total_cov, demand_matrix).sum()
    coverage_percentage = (total_covered / total_demand * 100) if total_demand > 0 else 0
    # --- Nuevo cálculo de cobertura real (puede superar 100%) ---
    real_cov_units = total_cov.sum()
    coverage_real = (real_cov_units / total_demand * 100) if total_demand > 0 else 0
    diff = total_cov - demand_matrix
    over = int(np.sum(diff[diff > 0]))
    under = int(np.sum(np.abs(diff[diff < 0])))
    return {
        "total_coverage": total_cov,
        "coverage_percentage": coverage_percentage,
        "coverage_real": coverage_real,
        "overstaffing": over,
        "understaffing": under,
        "total_agents": total_agents,
        "ft_agents": ft_agents,
        "pt_agents": pt_agents,
    }


def _pattern_matrix(pattern):
    if pattern is None:
        return None
    if isinstance(pattern, dict) and "matrix" in pattern:
        return np.asarray(pattern["matrix"])
    return np.asarray(pattern)


def _assigned_matrix_from(assignments, patterns, D, H):
    cov = np.zeros((D, H), dtype=int)
    if not assignments:
        return cov
    for pid, count in assignments.items():
        p = patterns.get(pid)
        if p is None:
            continue
        mat = _pattern_matrix(p).reshape(D, H)
        cov += mat * int(count)
    return cov


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _count_contracts(assignments: dict):
    ft = sum(v for k, v in assignments.items() if str(k).upper().startswith("FT"))
    pt = sum(v for k, v in assignments.items() if str(k).upper().startswith("PT"))
    return int(ft), int(pt), int(ft + pt)


def _export_xlsx_b64(assignments, payload):
    """
    Crea un Excel con:
    - Hoja 'asignaciones' (turno, agentes)
    - Hoja 'kpis'
    - Hojas 'demanda', 'cobertura', 'diferencias'
    Devuelve base64 o None si no hay motor de escritura.
    """
    try:
        import xlsxwriter  # noqa: F401
    except Exception:
        return None, "Falta el paquete 'xlsxwriter' (pip install xlsxwriter)"

    counts = Counter()
    for v in (assignments or {}).values():
        pid = v.get("pattern") if isinstance(v, dict) else v
        counts[str(pid)] += 1

    df_asg = pd.DataFrame(
        [{"turno": k, "agentes": v} for k, v in sorted(counts.items())],
        columns=["turno", "agentes"]
    )

    m = payload.get("metrics", {})
    df_kpi = pd.DataFrame([{
        "agentes": m.get("agents", 0),
        "cobertura_pura_%": m.get("coverage_pure", 0.0),
        "cobertura_real_%": m.get("coverage_real", 0.0),
        "exceso": m.get("excess", 0),
        "déficit": m.get("deficit", 0),
        "perfil": (payload.get("config") or {}).get("optimization_profile", ""),
    }])

    mats = payload.get("matrices", {})
    df_dem  = pd.DataFrame(mats.get("demand", []))
    df_cov  = pd.DataFrame(mats.get("coverage", []))
    df_diff = pd.DataFrame(mats.get("diff", []))

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as xls:
        df_asg.to_excel(xls, sheet_name="asignaciones", index=False)
        df_kpi.to_excel(xls, sheet_name="kpis", index=False)
        df_dem.to_excel(xls, sheet_name="demanda", index=False, header=False)
        df_cov.to_excel(xls, sheet_name="cobertura", index=False, header=False)
        df_diff.to_excel(xls, sheet_name="diferencias", index=False, header=False)

    return base64.b64encode(bio.getvalue()).decode("ascii"), None


def _heatmap(matrix, title, day_labels=None, hour_labels=None,
             annotate=True, cmap="Reds"):
    import matplotlib.pyplot as plt
    import numpy as np

    M = np.asarray(matrix)
    D, H = M.shape

    # Tamaño “wide” cómodo en UI oscura
    fig, ax = plt.subplots(figsize=(18, 6), dpi=110)

    im = ax.imshow(M, aspect="auto", cmap=cmap, interpolation="nearest")

    # Ejes y título
    ax.set_title(title, fontsize=14, pad=12)
    if hour_labels is not None:
        ax.set_xticks(range(H))
        ax.set_xticklabels(hour_labels, fontsize=9)
    else:
        ax.set_xticks(range(H))
        ax.set_xticklabels(range(H), fontsize=9)

    if day_labels is not None:
        ax.set_yticks(range(D))
        ax.set_yticklabels(day_labels, fontsize=10)
    else:
        ax.set_yticks(range(D))
        ax.set_yticklabels([f"Día {i+1}" for i in range(D)], fontsize=10)

    ax.set_xlabel("Hora del día", labelpad=8)
    ax.set_ylabel("Día de la semana", labelpad=8)

    # Números legibles (blanco o negro según el fondo)
    if annotate:
        norm = im.norm
        for i in range(D):
            for j in range(H):
                v = int(M[i, j])
                if v:
                    color = "white" if norm(v) > 0.6 else "black"
                    ax.text(j, i, f"{v}", ha="center", va="center", fontsize=9, color=color)

    # Colorbar fina
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    return fig


def _insights_from_analysis(analysis, cfg):
    ins = {"demanda": [], "pt_habilitados": [], "realista": [], "criticos": []}
    if not analysis:
        return ins

    dias = analysis.get("active_days", 6)
    horas_ini = analysis.get("start_hour", 8)
    horas_fin = analysis.get("end_hour", 20)
    weekly = analysis.get("weekly_total", analysis.get("demand_total", 0))
    avg = analysis.get("avg_per_active_day", analysis.get("avg", 0))
    peak = analysis.get("peak", 0)
    ins["demanda"] += [
        f"Días activos: {analysis.get('days_names','Lunes, Martes, Miércoles, Jueves, Viernes, Sábado')} ({dias} días)",
        f"Horario operativo: {horas_ini:02d}:00 - {horas_fin:02d}:00 ({horas_fin - horas_ini} horas)",
        f"Demanda total semanal: {weekly} agentes-hora",
        f"Demanda promedio (días activos): {avg} agentes-hora/día",
        f"Pico de demanda: {peak} agentes simultáneos",
        f"Break configurado: {cfg.get('break_start_from',2.0)}h desde inicio, {cfg.get('break_before_end',2.0)}h antes del fin",
    ]

    pt = []
    if cfg.get("use_pt", True) and cfg.get("allow_pt_4h"):
        pt.append("4h×6días")
    if cfg.get("use_pt", True) and cfg.get("allow_pt_6h"):
        pt.append("6h×4días")
    if cfg.get("use_pt", True) and cfg.get("allow_pt_5h"):
        pt.append("5h×5días")
    ins["pt_habilitados"].append(", ".join(pt) if pt else "—")

    ins["realista"] += [
        f"Agentes simultáneos máximos: {analysis.get('peak',0)}",
        f"Total agentes estimados: ~{analysis.get('agents_estimate','?')} para {weekly} agentes-hora",
        f"Ratio eficiencia: {analysis.get('efficiency_ratio','?')} horas productivas por agente",
    ]

    ins["criticos"] += [
        f"Días críticos: {analysis.get('critical_days_label','Miércoles, Viernes')}",
        f"Horas pico: {analysis.get('critical_hours_label','11:00 - 16:00')}",
        f"Perfil seleccionado: {cfg.get('optimization_profile','JEAN')}",
    ]
    return ins


def _analyze_demand(matrix):
    days_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    daily = matrix.sum(axis=1)
    hourly = matrix.sum(axis=0)
    active_days_idx = [i for i, v in enumerate(daily) if v > 0]
    active_hours = np.where(hourly > 0)[0]
    start_hour = int(active_hours.min()) if active_hours.size else 8
    end_hour = int(active_hours.max()) + 1 if active_hours.size else 20
    weekly_total = int(matrix.sum())
    avg_day = int(weekly_total / len(active_days_idx)) if active_days_idx else 0
    peak = int(matrix.max()) if matrix.size else 0
    days_label = ", ".join([days_names[i] for i in active_days_idx]) if active_days_idx else ""
    crit_days_idx = (
        np.argsort(daily)[-2:] if daily.size > 1 else ([int(np.argmax(daily))] if daily.size else [])
    )
    critical_days_label = ", ".join([days_names[i] for i in crit_days_idx]) if crit_days_idx.size else ""
    if np.any(hourly > 0):
        thresh = np.percentile(hourly[hourly > 0], 75)
        peak_hours = np.where(hourly >= thresh)[0]
        if peak_hours.size:
            critical_hours_label = f"{int(peak_hours.min()):02d}:00 - {int(peak_hours.max()+1):02d}:00"
        else:
            critical_hours_label = ""
    else:
        critical_hours_label = ""
    return {
        "active_days": len(active_days_idx),
        "start_hour": start_hour,
        "end_hour": end_hour,
        "weekly_total": weekly_total,
        "avg_per_active_day": avg_day,
        "peak": peak,
        "days_names": days_label,
        "critical_days_label": critical_days_label,
        "critical_hours_label": critical_hours_label,
    }

def _build_sync_payload(assignments, patterns, demand_matrix, *, day_labels=None, hour_labels=None, meta=None, cfg=None):
    dem = np.asarray(demand_matrix, dtype=int)
    D, H = dem.shape

    # Normaliza asignaciones a {patron: count}
    counts = {}
    for k, v in (assignments or {}).items():
        if isinstance(v, int) and isinstance(k, str):
            counts[k] = counts.get(k, 0) + int(v)
        else:
            pid = v.get("pattern") if isinstance(v, dict) else v
            if pid is None:
                continue
            counts[str(pid)] = counts.get(str(pid), 0) + 1

    cov = _assigned_matrix_from(counts, patterns, D, H)
    diff = cov - dem

    total_dem = int(dem.sum())
    real_cov_units = int(cov.sum())  # total cobertura (puede >100%)
    pure_cov_units = int(np.minimum(cov, dem).sum())  # cobertura efectiva (<=100%)
    excess  = int(np.clip(cov - dem, 0, None).sum())
    deficit = int(np.clip(dem - cov, 0, None).sum())

    coverage_pure = (pure_cov_units / total_dem * 100.0) if total_dem > 0 else 0.0   # <=100
    coverage_real = (real_cov_units / total_dem * 100.0) if total_dem > 0 else 0.0   # puede >100

    ft = sum(c for p, c in counts.items() if str(p).upper().startswith("FT"))
    pt = sum(c for p, c in counts.items() if str(p).upper().startswith("PT"))
    total_agents = ft + pt if (ft or pt) else len(assignments or {})

    fig_dem  = _heatmap(dem,  "Demanda (agentes-hora)", day_labels, hour_labels, annotate=True, cmap="Reds")
    fig_cov  = _heatmap(cov,  "Cobertura (agentes-hora)", day_labels, hour_labels, annotate=True, cmap="Blues")
    fig_diff = _heatmap(diff, "Cobertura - Demanda (exceso/deficit)", day_labels, hour_labels, annotate=True, cmap="RdBu_r")

    return {
        "metrics": {
            "agents": total_agents,
            "ft": ft,
            "pt": pt,
            "coverage_pure": round(coverage_pure, 1),
            "coverage_real": round(coverage_real, 1),
            "excess": excess,
            "deficit": deficit,
        },
        "matrices": {
            "demand": dem.tolist(),
            "coverage": cov.tolist(),
            "diff": diff.tolist(),
            "day_labels": day_labels or [f"Día {i+1}" for i in range(D)],
            "hour_labels": hour_labels or list(range(H)),
        },
        "figures": {
            "demand_png": _fig_to_b64(fig_dem),
            "coverage_png": _fig_to_b64(fig_cov),
            "diff_png": _fig_to_b64(fig_diff),
        },
        "meta": meta or {},
        "config": cfg or {},
    }


def _empty_result_with_insights(demand_matrix, cfg, *, reason, analysis=None):
    analysis = analysis or _analyze_demand(demand_matrix)
    D, H = demand_matrix.shape
    day_labels = [f"Día {i+1}" for i in range(D)]
    hour_labels = list(range(H))
    payload = _build_sync_payload(
        {},
        {},
        demand_matrix,
        day_labels=day_labels,
        hour_labels=hour_labels,
        meta={"status": "NO_PATTERNS", "reason": reason},
    )
    payload["status"] = "NO_PATTERNS"
    payload["config"] = cfg
    payload["insights"] = _insights_from_analysis(analysis, cfg)
    payload["reason"] = reason
    return payload


def run_complete_optimization(
    file_stream,
    config=None,
    generate_charts=False,
    job_id=None,
    return_payload=False,
): 
    cfg = config or {}
    requested = (
        cfg.get("optimization_profile")
        or cfg.get("profile")
        or "Equilibrado (Recomendado)"
    )
    resolved = resolve_profile_name(requested)
    if resolved and resolved != requested:
        print(f"[PROFILE] Usando alias '{requested}' -> '{resolved}'")
    profile_params = PROFILES.get(resolved)
    if not profile_params:
        print(
            f"[PROFILE] ADVERTENCIA: Perfil '{requested}' no encontrado, usando configuración por defecto"
        )
    else:
        for k, v in profile_params.items():
            cfg.setdefault(k, v)
    cfg["optimization_profile"] = resolved or requested
    optimization_profile = cfg.get("optimization_profile", "")

    # Alias para iteraciones: UI usa 'iterations', perfil usa 'search_iterations'
    if "iterations" in cfg and "search_iterations" not in cfg:
        try:
            cfg["search_iterations"] = int(cfg["iterations"])
        except Exception:
            pass

    # Propagar configuraciones globales a current_app.config
    if current_app is not None:
        solver_time = cfg.get("TIME_SOLVER", cfg.get("solver_time"))
        if solver_time is not None:
            current_app.config["TIME_SOLVER"] = solver_time
        current_app.config["iterations"] = int(
            cfg.get("search_iterations", cfg.get("iterations", 30))
        )
        try:
            current_app.config.pop("JEAN_LAST_ITERS", None)
        except Exception:
            pass

    # JEAN Personalizado: manejar JSON de turnos si viene en config
    if optimization_profile == "JEAN Personalizado":
        json_text = cfg.get("custom_shifts_json")
        if json_text:
            import tempfile, os, json as _json
            fd, path = tempfile.mkstemp(prefix="jean_shifts_", suffix=".json")
            os.close(fd)
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_text if isinstance(json_text, str) else _json.dumps(json_text, ensure_ascii=False))
            cfg["custom_shifts"] = True
            cfg["shift_config_file"] = path
            print(f"[CONFIG] JEAN Personalizado: custom shifts enabled -> {path}")
        else:
            print("[CONFIG] JEAN Personalizado: using standard JEAN logic (no custom JSON)")

    df = pd.read_excel(file_stream)
    demand_matrix = load_demand_matrix_from_df(df)
    analysis = _analyze_demand(demand_matrix)
    patterns = {}
    use_ft = cfg.get("use_ft", True)
    use_pt = cfg.get("use_pt", True)
    
    # Verificar si usar plantillas personalizadas
    use_custom_templates = cfg.get("custom_shifts", False) and cfg.get("shift_config_file")
    
    if use_custom_templates:
        # Cargar plantillas personalizadas desde JSON
        try:
            import json
            with open(cfg["shift_config_file"], "r", encoding="utf-8") as f:
                template_cfg = json.load(f)
            
            patterns = load_shift_patterns(
                template_cfg,
                demand_matrix=demand_matrix,
                max_patterns=None,
                keep_percentage=1.0,
                slot_duration_minutes=cfg.get("slot_duration_minutes", 60),
                smart_start_hours=cfg.get("smart_start_hours", False),
                cfg=cfg
            )
            print(f"[CUSTOM] Loaded {len(patterns)} custom templates")
        except Exception as e:
            print(f"[CUSTOM] Error loading templates: {e} -> using standard patterns")
            use_custom_templates = False
    
    if not use_custom_templates:
        # Generar patrones estándar FT/PT
        for batch in generate_shifts_coverage_optimized(
            demand_matrix,
            allow_8h=use_ft and cfg.get("allow_8h", False),
            allow_10h8=use_ft and cfg.get("allow_10h8", False),
            allow_pt_4h=use_pt and cfg.get("allow_pt_4h", False),
            allow_pt_5h=use_pt and cfg.get("allow_pt_5h", False),
            allow_pt_6h=use_pt and cfg.get("allow_pt_6h", False),
            keep_percentage=1.0,
            max_patterns=None,
            peak_bonus=cfg.get("peak_bonus", 1.5),
            critical_bonus=cfg.get("critical_bonus", 2.0),
            efficiency_bonus=cfg.get("efficiency_bonus", 1.0),
        ):
            patterns.update(batch)

    ft_count = sum(1 for k in patterns if k.startswith("FT"))
    pt_count = sum(1 for k in patterns if k.startswith("PT"))
    patterns_count = len(patterns)
    print(f"[GEN] Patrones FT={ft_count} PT={pt_count} TOTAL={patterns_count}")
    if patterns_count == 0:
        if return_payload:
            return _empty_result_with_insights(
                demand_matrix,
                cfg,
                analysis=analysis,
                reason="No patterns: check enabled contracts/shifts.",
            )
        return {"assignments": {}}

    TARGET_COVERAGE = cfg.get("TARGET_COVERAGE", 100.0)  # Cambio a 100% para priorizar cobertura real
    agent_limit_factor = cfg.get("agent_limit_factor", 30)
    excess_penalty = cfg.get("excess_penalty", 0)
    peak_bonus = cfg.get("peak_bonus", 0)
    critical_bonus = cfg.get("critical_bonus", 0)

    if optimization_profile in ("JEAN", "JEAN Personalizado"):
        # Apply JEAN profile parameters before optimizing
        cfg = apply_profile(cfg)
        
        # Ensure JEAN uses correct default agent_limit_factor (15, not 30)
        if cfg.get("agent_limit_factor") is None or cfg.get("agent_limit_factor") == 30:
            cfg["agent_limit_factor"] = 15
        
        if optimization_profile == "JEAN Personalizado":
            try:
                from .profile_optimizers import optimize_jean_personalizado
                assignments, pulp_status = optimize_jean_personalizado(
                    patterns,
                    demand_matrix,
                    cfg=cfg,
                )
            except ImportError:
                # Fallback to standard JEAN if no profile_optimizers
                assignments, pulp_status = optimize_jean_search(
                    patterns,
                    demand_matrix,
                    target_coverage=TARGET_COVERAGE,
                    agent_limit_factor=cfg.get("agent_limit_factor", 15),
                    excess_penalty=cfg.get("excess_penalty", 5.0),
                    peak_bonus=cfg.get("peak_bonus", 2.0),
                    critical_bonus=cfg.get("critical_bonus", 2.5),
                    cfg=cfg
                )
        else:
            # Standard JEAN with profile parameters
            assignments, pulp_status = optimize_jean_search(
                patterns,
                demand_matrix,
                target_coverage=TARGET_COVERAGE,
                agent_limit_factor=cfg.get("agent_limit_factor", 15),
                excess_penalty=cfg.get("excess_penalty", 5.0),
                peak_bonus=cfg.get("peak_bonus", 2.0),
                critical_bonus=cfg.get("critical_bonus", 2.5),
                iteration_time_limit=cfg.get("solver_time", 240),
                max_iterations=cfg.get("search_iterations", cfg.get("iterations", 10)),
                cfg=cfg
            )
        assignments = {k: v for k, v in (assignments or {}).items() if _is_allowed_pid(k, cfg)}
    elif optimization_profile == "Aprendizaje Adaptativo":
        # Get adaptive parameters from history
        adaptive_params = get_adaptive_params(demand_matrix)
        
        # Update config with adaptive parameters
        temp_cfg = cfg.copy()
        temp_cfg.update(adaptive_params)
        
        start_time = time.time()
        
        try:
            from .profile_optimizers import optimize_adaptive_learning
            assignments, pulp_status = optimize_adaptive_learning(
                patterns,
                demand_matrix,
                cfg=temp_cfg,
            )
        except ImportError:
            # Fallback to JEAN search with adaptive parameters
            assignments, pulp_status = optimize_jean_search(
                patterns,
                demand_matrix,
                target_coverage=TARGET_COVERAGE,
                agent_limit_factor=adaptive_params["agent_limit_factor"],
                excess_penalty=adaptive_params["excess_penalty"],
                peak_bonus=adaptive_params["peak_bonus"],
                critical_bonus=adaptive_params["critical_bonus"],
                cfg=temp_cfg
            )
        
        assignments = {k: v for k, v in (assignments or {}).items() if _is_allowed_pid(k, cfg)}
        
        # Save execution result for future learning
        if assignments:
            results = analyze_results(assignments, patterns, demand_matrix)
            if results:
                execution_time = time.time() - start_time
                save_execution_result(
                    demand_matrix, 
                    temp_cfg, 
                    results["coverage_percentage"], 
                    results["total_agents"],
                    execution_time,
                    results["overstaffing"],
                    results["understaffing"]
                )
    elif use_ft and use_pt and optimization_profile not in ("JEAN", "JEAN Personalizado", "Aprendizaje Adaptativo"):
        # Unified profiles: apply same toggles and configuration as original Streamlit
        # Always enable two-phase strategy (ft_first_pt_last) for combined FT/PT profiles
        assignments, pulp_status = optimize_ft_then_pt_strategy(patterns, demand_matrix, cfg=cfg)
        assignments = {k: v for k, v in (assignments or {}).items() if _is_allowed_pid(k, cfg)}
    else:
        normalized = resolve_profile_name(optimization_profile or "") or optimization_profile
        print(
            f"[SCHEDULER] Executing standard profile with dedicated optimizer: {normalized}"
        )
        
        # Usar optimize_maximum_coverage para el nuevo perfil
        if normalized == "Cobertura Máxima (Completo)":
            assignments, pulp_status = optimize_maximum_coverage(patterns, demand_matrix, cfg=cfg)
        else:
            opt_fn = get_profile_optimizer(normalized)
            cfg["optimization_profile"] = normalized
            try:
                assignments, status = opt_fn(
                    patterns, demand_matrix, cfg=cfg, job_id=job_id
                )
            except Exception as e:
                print(
                    f"[SCHEDULER] Error in profile optimizer: {e} -> fallback to two-phase"
                )
                try:
                    assignments, status = optimize_ft_then_pt_strategy(patterns, demand_matrix, cfg=cfg)
                except Exception as e2:
                    print(f"[SCHEDULER] Error in two-phase: {e2} -> fallback to chunks")
                    assignments = solve_in_chunks_optimized(
                        patterns,
                        demand_matrix,
                        optimization_profile=normalized,
                        use_ft=use_ft,
                        use_pt=use_pt,
                        TARGET_COVERAGE=TARGET_COVERAGE,
                        agent_limit_factor=agent_limit_factor,
                        excess_penalty=excess_penalty,
                        peak_bonus=peak_bonus,
                        critical_bonus=critical_bonus,
                    )
                    status = "CHUNKS_FALLBACK"
            pulp_status = status
        assignments = {k: v for k, v in (assignments or {}).items() if _is_allowed_pid(k, cfg)}

    jean_iters = []
    if current_app is not None:
        try:
            jean_iters = current_app.config.get("JEAN_LAST_ITERS", [])
        except Exception:
            jean_iters = []

    if return_payload:
        # --- ANTES de armar payload, filtra patrones según familias seleccionadas ---
        patterns = _filter_patterns_by_cfg(patterns, cfg)

        # --- Payload con KPIs (pura vs real) ---
        D, H = demand_matrix.shape
        day_labels = [f"Día {i+1}" for i in range(D)]
        hour_labels = list(range(H))
        payload = _build_sync_payload(
            assignments=assignments or {},
            patterns=patterns,
            demand_matrix=demand_matrix,
            day_labels=day_labels,
            hour_labels=hour_labels,
            meta={"status": pulp_status},
            cfg=cfg,
        )
        payload["status"] = pulp_status
        payload["effective_profile"] = cfg.get("optimization_profile", "")
        payload["insights"] = _insights_from_analysis(analysis, cfg)
        payload["jean_iterations"] = jean_iters

        # --- Exportable ---
        b64, err = _export_xlsx_b64(assignments or {}, payload)
        if b64:
            payload["export_b64"] = b64
        if err:
            payload["export_error"] = err

        return payload

    return {"assignments": assignments, "jean_iterations": jean_iters}
