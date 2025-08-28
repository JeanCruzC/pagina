# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import math
from io import BytesIO
from pyworkforce.scheduling import MinAbsDifference
from itertools import combinations, permutations, product
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
import json
import hashlib
import os
import re
import gc
import psutil

from typing import Dict, List, Iterable, Union


def _build_pattern(
    days: Iterable[int],
    durations: Iterable[int],
    start_hour: float,
    break_len: float,
    break_from_start: float,
    break_from_end: float,
    slot_factor: int = 1,
) -> np.ndarray:
    """Return flattened weekly matrix with custom slot resolution."""
    slots_per_day = 24 * slot_factor
    pattern = np.zeros((7, slots_per_day), dtype=np.int8)
    for day, dur in zip(days, durations):
        for s in range(int(dur * slot_factor)):
            slot = int(start_hour * slot_factor) + s
            d_off, idx = divmod(slot, slots_per_day)
            pattern[(day + d_off) % 7, idx] = 1
        if break_len:
            b_start = int((start_hour + break_from_start) * slot_factor)
            b_end = int((start_hour + dur - break_from_end) * slot_factor)
            if b_start < b_end:
                b_slot = b_start + (b_end - b_start) // 2
            else:
                b_slot = b_start
            for b in range(int(break_len * slot_factor)):
                slot = b_slot + b
                d_off, idx = divmod(slot, slots_per_day)
                pattern[(day + d_off) % 7, idx] = 0
    return pattern.flatten()


def memory_limit_patterns(slots_per_day: int) -> int:
    """Return how many patterns fit in roughly 4GB of RAM."""
    if slots_per_day <= 0:
        return 0
    available = psutil.virtual_memory().available
    cap = min(available, 4 * 1024 ** 3)
    return int(cap // (7 * slots_per_day))


def monitor_memory_usage() -> float:
    """Return current memory usage percentage."""
    return psutil.virtual_memory().percent


def adaptive_chunk_size(base: int = 5000) -> int:
    """Return dynamic chunk size based on memory pressure."""
    usage = monitor_memory_usage()
    if usage > 80:
        return max(1000, base // 4)
    if usage > 60:
        return max(2000, base // 2)
    return base


def emergency_cleanup(threshold: float = 85.0) -> bool:
    """Trigger garbage collection if memory usage exceeds ``threshold``."""
    if monitor_memory_usage() >= threshold:
        gc.collect()
        return True
    return False


def get_smart_start_hours(demand_matrix: np.ndarray, max_hours: int = 12) -> List[float]:
    """Return a list of start hours around peak demand."""
    if demand_matrix is None or demand_matrix.size == 0:
        return [float(h) for h in range(24)]

    cols = demand_matrix.shape[1]
    hourly_totals = demand_matrix.sum(axis=0)
    top = np.argsort(hourly_totals)[-max_hours:]
    hours = sorted({round(h / cols * 24, 2) for h in top})
    return hours


def show_generation_progress(count: int, start_time: float) -> None:
    """Display generation rate and memory usage."""
    rate = count / max(1e-6, time.time() - start_time)
    mem = monitor_memory_usage()
    st.text(f"Patrones {count} | {rate:.1f}/s | Mem {mem:.1f}%")


def score_pattern(pattern: np.ndarray, demand_matrix: np.ndarray) -> int:
    """Quick heuristic score to sort patterns before solving."""
    dm = demand_matrix.flatten()
    pat = pattern.astype(int)
    lim = min(len(dm), len(pat))
    return int(np.minimum(pat[:lim], dm[:lim]).sum())


def _resize_matrix(matrix: np.ndarray, target_cols: int) -> np.ndarray:
    """Return ``matrix`` with the number of columns adjusted to ``target_cols``."""
    if matrix.shape[1] == target_cols:
        return matrix
    if matrix.shape[1] < target_cols:
        factor = target_cols // matrix.shape[1]
        return np.repeat(matrix, factor, axis=1)
    factor = matrix.shape[1] // target_cols
    return matrix.reshape(matrix.shape[0], target_cols, factor).max(axis=2)


def score_and_filter_patterns(
    patterns: Dict[str, np.ndarray],
    demand_matrix: np.ndarray | None,
    *,
    keep_percentage: float = 0.3,
    peak_bonus: float = 1.5,
    critical_bonus: float = 2.0,
    efficiency_bonus: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Score patterns and keep the best subset.

    Parameters
    ----------
    patterns:
        Mapping of pattern name to flattened coverage matrix.
    demand_matrix:
        Weekly demand matrix. If ``None`` no filtering is applied.
    keep_percentage:
        Fraction of patterns to keep. Defaults to ``0.3``.
    peak_bonus, critical_bonus:
        Multipliers for hours identified as peak or critical.
    """

    if demand_matrix is None or not patterns:
        return patterns

    dm = np.asarray(demand_matrix, dtype=float)
    days, hours = dm.shape
    daily_totals = dm.sum(axis=1)
    hourly_totals = dm.sum(axis=0)

    critical_days = (
        np.argsort(daily_totals)[-2:]
        if daily_totals.size > 1
        else [int(np.argmax(daily_totals))]
    )
    if np.any(hourly_totals > 0):
        thresh = np.percentile(hourly_totals[hourly_totals > 0], 75)
        peak_hours = np.where(hourly_totals >= thresh)[0]
    else:
        peak_hours = []

    scores = []
    for name, pat in patterns.items():
        cols = len(pat) // 7
        pat_mat = pat.reshape(7, cols)
        dm_resized = _resize_matrix(dm, cols)
        coverage = np.minimum(pat_mat, dm_resized)
        score = coverage.sum()
        total_hours = pat_mat.sum()
        if total_hours > 0:
            efficiency = coverage.sum() / total_hours
            score += efficiency * efficiency_bonus
        if len(critical_days) > 0:
            score += coverage[critical_days].sum() * critical_bonus
        if len(peak_hours) > 0:
            ph = [h for h in peak_hours if h < cols]
            if ph:
                score += coverage[:, ph].sum() * peak_bonus
        scores.append((name, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    keep_n = max(1, int(len(scores) * keep_percentage))
    top = {name for name, _ in scores[:keep_n]}
    return {k: patterns[k] for k in top}


def load_shift_patterns(
    cfg: Union[str, dict],
    *,
    start_hours: Iterable[float] | None = None,
    break_from_start: float = 2.0,
    break_from_end: float = 2.0,
    slot_duration_minutes: int | None = 30,
    max_patterns: int | None = None,
    demand_matrix: np.ndarray | None = None,
    keep_percentage: float = 0.3,
    peak_bonus: float = 1.5,
    critical_bonus: float = 2.0,
    efficiency_bonus: float = 1.0,
    max_patterns_per_shift: int | None = None,
    smart_start_hours: bool = False,
) -> Dict[str, np.ndarray]:
    """Parse JSON shift configuration and return pattern dictionary.

    If ``slot_duration_minutes`` is provided it overrides the value of
    ``slot_duration_minutes`` defined inside each shift.  Passing ``None`` keeps
    the per-shift resolution intact.  When ``max_patterns`` is provided the
    generator stops once that many unique patterns have been produced.  When a
    ``demand_matrix`` is supplied patterns are scored and only the top
    ``keep_percentage`` are returned. ``peak_bonus`` and ``critical_bonus``
    control the extra weight for covering peak hours or critical days.
    """
    if isinstance(cfg, str):
        with open(cfg, "r") as fh:
            data = json.load(fh)
    else:
        data = cfg

    if slot_duration_minutes is not None:
        if 60 % slot_duration_minutes != 0:
            raise ValueError("slot_duration_minutes must divide 60")

    base_slot_min = slot_duration_minutes
    if base_slot_min is None:
        mins = [shift.get("slot_duration_minutes", 60) for shift in data.get("shifts", [])]
        base_slot_min = mins and min(mins) or 60
    slots_per_day = 24 * (60 // base_slot_min)
    if max_patterns is None:
        max_patterns = memory_limit_patterns(slots_per_day)

    shifts_coverage: Dict[str, np.ndarray] = {}
    unique_patterns: Dict[bytes, str] = {}
    for shift in data.get("shifts", []):
        name = shift.get("name", "SHIFT")
        pat = shift.get("pattern", {})
        brk = shift.get("break", 0)

        slot_min = (
            slot_duration_minutes
            if slot_duration_minutes is not None
            else shift.get("slot_duration_minutes", 60)
        )
        if 60 % slot_min != 0:
            raise ValueError("slot_duration_minutes must divide 60")
        step = slot_min / 60
        slot_factor = 60 // slot_min
        base_hours = (
            list(start_hours)
            if start_hours is not None
            else list(np.arange(0, 24, step))
        )
        if smart_start_hours and demand_matrix is not None:
            smart = get_smart_start_hours(demand_matrix)
            sh_hours = [h for h in base_hours if any(abs(h - s) < step / 2 for s in smart)]
        else:
            sh_hours = base_hours

        work_days = pat.get("work_days", [])
        segments_spec = pat.get("segments", [])
        segments: List[int] = []
        for seg in segments_spec:
            if isinstance(seg, dict):
                hours = seg.get("hours")
                count = seg.get("count", 1)
                if hours is None:
                    continue
                segments.extend([int(hours)] * int(count))
            else:
                segments.append(int(seg))

        if isinstance(work_days, int):
            day_candidates = range(7)
            day_combos = combinations(day_candidates, work_days)
        else:
            day_combos = combinations(work_days, min(len(segments), len(work_days)))

        if isinstance(brk, dict):
            if brk.get("enabled", False):
                brk_len = brk.get("length_minutes", 0) / 60
                brk_start = brk.get("earliest_after_start", 0) / 60
                brk_end = brk.get("latest_before_end", 0) / 60
            else:
                brk_len = 0
                brk_start = break_from_start
                brk_end = break_from_end
        else:
            brk_len = float(brk)
            brk_start = break_from_start
            brk_end = break_from_end


        shift_patterns: Dict[str, np.ndarray] = {}
        for days_sel in day_combos:
            for perm in set(permutations(segments, len(days_sel))):
                for sh in sh_hours:
                    pattern = _build_pattern(
                        days_sel, perm, sh, brk_len, brk_start, brk_end, slot_factor
                    )
                    pat_key = pattern.tobytes()
                    if pat_key in unique_patterns:
                        continue
                    day_str = "".join(map(str, days_sel))
                    seg_str = "_".join(map(str, perm))
                    shift_name = f"{name}_{sh:04.1f}_{day_str}_{seg_str}"
                    shift_patterns[shift_name] = pattern
                    unique_patterns[pat_key] = shift_name
                    if max_patterns_per_shift and len(shift_patterns) >= max_patterns_per_shift:
                        break
                    if max_patterns is not None and len(shifts_coverage) + len(shift_patterns) >= max_patterns:
                        break
                if max_patterns_per_shift and len(shift_patterns) >= max_patterns_per_shift:
                    break
                if max_patterns is not None and len(shifts_coverage) + len(shift_patterns) >= max_patterns:
                    break
            if max_patterns_per_shift and len(shift_patterns) >= max_patterns_per_shift:
                break
            if max_patterns is not None and len(shifts_coverage) + len(shift_patterns) >= max_patterns:
                break

        if demand_matrix is not None:
            shift_patterns = score_and_filter_patterns(
                shift_patterns,
                demand_matrix,
                keep_percentage=keep_percentage,
                peak_bonus=peak_bonus,
                critical_bonus=critical_bonus,
                efficiency_bonus=efficiency_bonus,
            )

        shifts_coverage.update(shift_patterns)
        monitor_memory_usage()
        gc.collect()
        if max_patterns is not None and len(shifts_coverage) >= max_patterns:
            return shifts_coverage
    if demand_matrix is not None:
        shifts_coverage = score_and_filter_patterns(
            shifts_coverage,
            demand_matrix,
            keep_percentage=keep_percentage,
            peak_bonus=peak_bonus,
            critical_bonus=critical_bonus,
            efficiency_bonus=efficiency_bonus,
        )
    return shifts_coverage
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

# Store uploaded JSON configuration for custom shifts
template_cfg = {}

# ——————————————————————————————————————————————————————————————
# Sistema de Aprendizaje Adaptativo
# ——————————————————————————————————————————————————————————————

# Remover cache para permitir evolución en tiempo real
def load_learning_data():
    """Carga datos de aprendizaje con caché"""
    try:
        if os.path.exists("optimization_learning.json"):
            with open("optimization_learning.json", 'r') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Error al cargar el aprendizaje: {e}")
    return {"executions": [], "best_params": {}, "stats": {}}

def save_learning_data(data):
    """Guarda datos de aprendizaje sin cache"""
    try:
        with open("optimization_learning.json", 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.warning(f"No se pudo guardar el aprendizaje: {e}")

def get_adaptive_params(demand_matrix, target_coverage):
    """Sistema de aprendizaje evolutivo que mejora en cada ejecución"""
    learning_data = load_learning_data()
    input_hash = hashlib.md5(str(demand_matrix).encode()).hexdigest()[:12]
    
    # Análisis de patrón de demanda
    total_demand = demand_matrix.sum()
    peak_demand = demand_matrix.max()
    
    # Buscar ejecuciones similares
    similar_runs = []
    for execution in learning_data.get("executions", []):
        if execution.get("input_hash") == input_hash:
            similar_runs.append(execution)
    
    # Ordenar por timestamp para ver evolución
    similar_runs.sort(key=lambda x: x.get("timestamp", 0))
    
    if len(similar_runs) >= 1:
        # Obtener última ejecución y mejor histórica
        last_run = similar_runs[-1]
        best_run = max(similar_runs, key=lambda x: x.get("coverage", 0))
        
        last_coverage = last_run.get("coverage", 0)
        best_coverage = best_run.get("coverage", 0)
        coverage_gap = target_coverage - last_coverage
        
        # Parámetros base de la mejor ejecución
        base_params = best_run.get("params", {})
        
        # Factor de evolución basado en número de ejecuciones
        evolution_factor = min(0.3, len(similar_runs) * 0.05)
        
        # Si no mejoramos en las últimas 2 ejecuciones, ser más agresivo
        if len(similar_runs) >= 3:
            recent_coverages = [run.get("coverage", 0) for run in similar_runs[-3:]]
            if recent_coverages[-1] <= recent_coverages[-2]:
                evolution_factor *= 2  # Doble agresividad
        
        # Ajuste evolutivo basado en brecha
        if coverage_gap > 10:  # Brecha grande
            return {
                "agent_limit_factor": max(5, int(base_params.get("agent_limit_factor", 20) * (1 - evolution_factor))),
                "excess_penalty": base_params.get("excess_penalty", 0.1) * (1 - evolution_factor),
                "peak_bonus": base_params.get("peak_bonus", 2.0) * (1 + evolution_factor),
                "critical_bonus": base_params.get("critical_bonus", 2.5) * (1 + evolution_factor),
                "precision_mode": True,
                "learned": True,
                "runs_count": len(similar_runs),
                "evolution_step": "aggressive"
            }
        elif coverage_gap > 3:  # Brecha media
            return {
                "agent_limit_factor": max(8, int(base_params.get("agent_limit_factor", 20) * (1 - evolution_factor * 0.5))),
                "excess_penalty": base_params.get("excess_penalty", 0.2) * (1 - evolution_factor * 0.5),
                "peak_bonus": base_params.get("peak_bonus", 1.8) * (1 + evolution_factor * 0.5),
                "critical_bonus": base_params.get("critical_bonus", 2.0) * (1 + evolution_factor * 0.5),
                "precision_mode": True,
                "learned": True,
                "runs_count": len(similar_runs),
                "evolution_step": "moderate"
            }
        elif coverage_gap > 0:  # Ajuste fino
            return {
                "agent_limit_factor": max(12, int(base_params.get("agent_limit_factor", 22) * (1 - evolution_factor * 0.2))),
                "excess_penalty": base_params.get("excess_penalty", 0.3) * (1 - evolution_factor * 0.2),
                "peak_bonus": base_params.get("peak_bonus", 1.5) * (1 + evolution_factor * 0.2),
                "critical_bonus": base_params.get("critical_bonus", 1.8) * (1 + evolution_factor * 0.2),
                "precision_mode": False,
                "learned": True,
                "runs_count": len(similar_runs),
                "evolution_step": "fine_tune"
            }
        else:  # Objetivo alcanzado - explorar variaciones
            # Pequeñas variaciones para mantener diversidad
            noise = np.random.uniform(-0.1, 0.1)
            return {
                "agent_limit_factor": max(15, int(base_params.get("agent_limit_factor", 20) * (1 + noise))),
                "excess_penalty": max(0.01, base_params.get("excess_penalty", 0.5) * (1 + noise)),
                "peak_bonus": base_params.get("peak_bonus", 1.5) * (1 + noise * 0.5),
                "critical_bonus": base_params.get("critical_bonus", 2.0) * (1 + noise * 0.5),
                "precision_mode": False,
                "learned": True,
                "runs_count": len(similar_runs),
                "evolution_step": "explore"
            }
    
    # Primera ejecución - parámetros iniciales agresivos
    return {
        "agent_limit_factor": max(8, int(total_demand / max(1, peak_demand) * 3)),
        "excess_penalty": 0.05,
        "peak_bonus": 2.5,
        "critical_bonus": 3.0,
        "precision_mode": True,
        "learned": False,
        "runs_count": 0,
        "evolution_step": "initial"
    }

def save_execution_result(demand_matrix, params, coverage, total_agents, execution_time):
    """Guarda resultado con análisis de mejora evolutiva"""
    learning_data = load_learning_data()
    input_hash = hashlib.md5(str(demand_matrix).encode()).hexdigest()[:12]
    
    # Calcular métricas de calidad mejoradas
    efficiency_score = coverage / max(1, total_agents * 0.1)  # Cobertura por costo
    balance_score = coverage - abs(coverage - 100) * 0.5  # Penalizar exceso y déficit
    
    execution_result = {
        "timestamp": time.time(),
        "input_hash": input_hash,
        "params": {
            "agent_limit_factor": params.get("agent_limit_factor"),
            "excess_penalty": params.get("excess_penalty"),
            "peak_bonus": params.get("peak_bonus"),
            "critical_bonus": params.get("critical_bonus")
        },
        "coverage": coverage,
        "total_agents": total_agents,
        "efficiency_score": efficiency_score,
        "balance_score": balance_score,
        "execution_time": execution_time,
        "demand_total": float(demand_matrix.sum()),
        "evolution_step": params.get("evolution_step", "unknown")
    }
    
    learning_data["executions"].append(execution_result)
    
    # Mantener solo últimas 50 ejecuciones por patrón
    pattern_executions = [e for e in learning_data["executions"] if e.get("input_hash") == input_hash]
    if len(pattern_executions) > 50:
        # Remover las más antiguas de este patrón
        learning_data["executions"] = [e for e in learning_data["executions"] 
                                     if e.get("input_hash") != input_hash or 
                                     e.get("timestamp", 0) >= sorted([p.get("timestamp", 0) for p in pattern_executions])[-50]]
    
    # Actualizar mejores parámetros con múltiples criterios
    current_best = learning_data["best_params"].get(input_hash, {})
    
    # Score combinado: priorizar cobertura, luego eficiencia
    if coverage >= 98:  # Si cobertura es buena, optimizar eficiencia
        new_score = efficiency_score
    else:  # Si cobertura es baja, priorizarla
        new_score = coverage * 2
    
    if not current_best or new_score > current_best.get("score", 0):
        learning_data["best_params"][input_hash] = {
            "params": execution_result["params"],
            "coverage": coverage,
            "total_agents": total_agents,
            "score": new_score,
            "efficiency_score": efficiency_score,
            "timestamp": time.time()
        }
    
    # Estadísticas evolutivas
    pattern_runs = [e for e in learning_data["executions"] if e.get("input_hash") == input_hash]
    if len(pattern_runs) >= 2:
        recent_improvement = pattern_runs[-1]["coverage"] - pattern_runs[-2]["coverage"]
    else:
        recent_improvement = 0
    
    learning_data["stats"] = {
        "total_executions": len(learning_data["executions"]),
        "unique_patterns": len(set(e["input_hash"] for e in learning_data["executions"])),
        "avg_coverage": np.mean([e["coverage"] for e in learning_data["executions"][-10:]]),
        "recent_improvement": recent_improvement,
        "best_coverage": max([e["coverage"] for e in learning_data["executions"]], default=0),
        "last_updated": time.time()
    }
    
    save_learning_data(learning_data)
    return True

# ——————————————————————————————————————————————————————————————
# 0. Configuración
# ——————————————————————————————————————————————————————————————
st.set_page_config(page_title="Generador v6.2 - Optimización Corregida", layout="wide")
st.title("Generador de Turnos v6.2 - Sistema Corregido")

# ——————————————————————————————————————————————————————————————
# 1. Carga de demanda
# ——————————————————————————————————————————————————————————————
uploaded = st.file_uploader("Sube tu Excel de demanda (Requerido.xlsx)", type="xlsx")
if not uploaded:
    st.warning("Por favor, sube el archivo Requerido.xlsx para continuar.")
    st.stop()

df = pd.read_excel(uploaded)

# ——————————————————————————————————————————————————————————————
# 2. Parámetros de configuración
# ——————————————————————————————————————————————————————————————
st.sidebar.header("⚙️ Configuración")
MAX_ITER = int(st.sidebar.number_input("Iteraciones máximas", 10, 100, 30))
TIME_SOLVER = float(st.sidebar.number_input("Tiempo solver (s)", 60, 600, 240))
TARGET_COVERAGE = float(st.sidebar.slider("Cobertura objetivo (%)", 95, 100, 98))
VERBOSE = st.sidebar.checkbox("Modo verbose/debug", False)

# Configuración de contratos
st.sidebar.subheader("📋 Tipos de Contrato")
use_ft = st.sidebar.checkbox("Full Time (48h)", True, key="use_ft_main")
use_pt = st.sidebar.checkbox("Part Time (24h)", True, key="use_pt_main")

# Configuraciones FT
if use_ft:
    st.sidebar.subheader("⏰ Turnos FT Permitidos")
    allow_8h = st.sidebar.checkbox("8 horas (6 días)", True, key="allow_8h_main")
    allow_10h8 = st.sidebar.checkbox("10h + día de 8h (5 días)", False, key="allow_10h8_main")
else:
    allow_8h = allow_10h8 = False

# Configuración de breaks MEJORADA
st.sidebar.subheader("☕ Configuración de Breaks")
break_from_start = st.sidebar.slider(
    "Break desde inicio (horas)",
    min_value=1.0,
    max_value=5.0,
    value=2.5,
    step=0.5,
    help="Cuántas horas después del inicio puede ocurrir el break"
)

break_from_end = st.sidebar.slider(
    "Break antes del fin (horas)",
    min_value=1.0,
    max_value=5.0,
    value=2.5,
    step=0.5,
    help="Cuántas horas antes del fin puede ocurrir el break"
)

# Configuración PT
if use_pt:
    st.sidebar.subheader("⏰ Turnos PT Permitidos")
    allow_pt_4h = st.sidebar.checkbox("4 horas (6 días)", True, key="allow_pt_4h")
    allow_pt_6h = st.sidebar.checkbox("6 horas (4 días)", True, key="allow_pt_6h")
    allow_pt_5h = st.sidebar.checkbox("5 horas (5 días)", False, key="allow_pt_5h")
else:
    allow_pt_4h = allow_pt_6h = allow_pt_5h = False

# Configuración del Solver
st.sidebar.subheader("🎯 Perfil de Optimización")
optimization_profile = st.sidebar.selectbox(
    "Selecciona el perfil de optimización:",
    [
        "Equilibrado (Recomendado)",
        "Conservador", 
        "Agresivo",
        "Máxima Cobertura",
        "Mínimo Costo",
        "100% Cobertura Eficiente",
        "100% Cobertura Total",
        "Cobertura Perfecta",
        "100% Exacto",
        "JEAN",
        "JEAN Personalizado",
        "Personalizado",
        "Aprendizaje Adaptativo"
    ],
    index=0,
    help="Cada perfil ajusta automáticamente los parámetros del solver según el objetivo"
)

# Sistema de aprendizaje
use_learning = optimization_profile == "Aprendizaje Adaptativo"

# Mostrar tipo de solver disponible
if PULP_AVAILABLE:
    st.sidebar.success("🧠 **Solver Inteligente Disponible**\nProgramación Lineal (PuLP)")
else:
    st.sidebar.warning("🔄 **Solver Básico Activo**\nPara mejor rendimiento instala:\n`pip install pulp`")

# Inicializar session state para aprendizaje
if 'learning_initialized' not in st.session_state:
    st.session_state.learning_initialized = True
    st.session_state.learning_data = load_learning_data()

# Mostrar estadísticas de aprendizaje si existe historial
if use_learning:
    learning_stats = st.session_state.learning_data.get("stats", {})
    if learning_stats:
        improvement = learning_stats.get('recent_improvement', 0)
        best_coverage = learning_stats.get('best_coverage', 0)
        improvement_icon = "📈" if improvement > 0 else "📊" if improvement == 0 else "📉"
        st.sidebar.success(f"🧠 **Sistema Evolutivo Activo**")
        st.sidebar.info(f"{improvement_icon} **Progreso:**\n- Ejecuciones: {learning_stats.get('total_executions', 0)}\n- Mejor cobertura: {best_coverage:.1f}%\n- Última mejora: {improvement:+.1f}%\n- Patrones: {learning_stats.get('unique_patterns', 0)}")
    else:
        st.sidebar.info("🧠 **Modo Evolutivo**: El sistema mejora automáticamente en cada ejecución")

# Perfiles predefinidos
profiles = {
    "Equilibrado (Recomendado)": {"agent_limit_factor": 12, "excess_penalty": 2.0, "peak_bonus": 1.5, "critical_bonus": 2.0},
    "Conservador": {"agent_limit_factor": 30, "excess_penalty": 0.5, "peak_bonus": 1.0, "critical_bonus": 1.2},
    "Agresivo": {"agent_limit_factor": 15, "excess_penalty": 0.05, "peak_bonus": 1.5, "critical_bonus": 2.0},
    "Máxima Cobertura": {"agent_limit_factor": 7, "excess_penalty": 0.005, "peak_bonus": 3.0, "critical_bonus": 4.0},
    "Mínimo Costo": {"agent_limit_factor": 35, "excess_penalty": 0.8, "peak_bonus": 0.8, "critical_bonus": 1.0},
    "100% Cobertura Eficiente": {"agent_limit_factor": 6, "excess_penalty": 0.01, "peak_bonus": 3.5, "critical_bonus": 4.5},
    "100% Cobertura Total": {"agent_limit_factor": 5, "excess_penalty": 0.001, "peak_bonus": 4.0, "critical_bonus": 5.0},
    "Cobertura Perfecta": {"agent_limit_factor": 8, "excess_penalty": 0.01, "peak_bonus": 3.0, "critical_bonus": 4.0},
    "100% Exacto": {"agent_limit_factor": 6, "excess_penalty": 0.005, "peak_bonus": 4.0, "critical_bonus": 5.0},
    "JEAN": {"agent_limit_factor": 30, "excess_penalty": 5.0, "peak_bonus": 2.0, "critical_bonus": 2.5},
    "Aprendizaje Adaptativo": {"agent_limit_factor": 8, "excess_penalty": 0.01, "peak_bonus": 3.0, "critical_bonus": 4.0},
}

# Configuración inicial de parámetros
if optimization_profile == "Personalizado":
    st.sidebar.subheader("⚙️ Parámetros Personalizados")
    agent_limit_factor = st.sidebar.slider("Factor límite agentes", 15, 35, 25, help="Menor = más agentes")
    excess_penalty = st.sidebar.slider("Penalización exceso", 0.1, 2.0, 0.5, step=0.1)
    peak_bonus = st.sidebar.slider("Bonificación horas pico", 1.0, 3.0, 1.5, step=0.1)
    critical_bonus = st.sidebar.slider("Bonificación días críticos", 1.0, 3.0, 2.0, step=0.1)

elif optimization_profile == "JEAN Personalizado":
    st.sidebar.subheader("⚙️ JEAN Personalizado")
    template_file = st.sidebar.file_uploader("Plantilla JSON", type="json")
    if not template_file:
        st.sidebar.warning("Debe subir un archivo JSON de configuración")
        st.stop()
    try:
        template_cfg = json.load(template_file)
    except Exception as e:
        st.sidebar.error(f"Error al leer plantilla: {e}")
        st.stop()

    jean_cfg = profiles["JEAN"]
    agent_limit_factor = jean_cfg["agent_limit_factor"]
    excess_penalty = jean_cfg["excess_penalty"]
    peak_bonus = jean_cfg["peak_bonus"]
    critical_bonus = jean_cfg["critical_bonus"]

    use_ft = st.sidebar.checkbox(
        "Permitir FT", template_cfg.get("use_ft", True), key="jean_use_ft"
    )
    use_pt = st.sidebar.checkbox(
        "Permitir PT", template_cfg.get("use_pt", True), key="jean_use_pt"
    )

    ft_work_days = template_cfg.get("ft_work_days", 0)
    ft_shift_hours = template_cfg.get("ft_shift_hours", 0)
    ft_break_duration = template_cfg.get("ft_break_duration", 0)
    ft_break_from_start = template_cfg.get("ft_break_from_start", 0.0)
    ft_break_from_end = template_cfg.get("ft_break_from_end", 0.0)

    pt_work_days = template_cfg.get("pt_work_days", 0)
    pt_shift_hours = template_cfg.get("pt_shift_hours", 0)
    pt_break_duration = template_cfg.get("pt_break_duration", 0)
    pt_break_from_start = template_cfg.get("pt_break_from_start", 0.0)
    pt_break_from_end = template_cfg.get("pt_break_from_end", 0.0)

else:
    # Configuraciones predefinidas
    
    config = profiles[optimization_profile]
    agent_limit_factor = config["agent_limit_factor"]
    excess_penalty = config["excess_penalty"]
    peak_bonus = config["peak_bonus"]
    critical_bonus = config["critical_bonus"]

    ft_work_days = pt_work_days = 0
    ft_shift_hours = pt_shift_hours = 0
    ft_break_duration = pt_break_duration = 1
    # break_from_start and break_from_end already defined earlier



# ——————————————————————————————————————————————————————————————
# 3. Análisis de demanda
# ——————————————————————————————————————————————————————————————
dias_semana = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
required_resources = []
for i in range(7):
    required_resources.append([])

for _, row in df.iterrows():
    d = int(row['Día']) - 1
    required_resources[d].append(int(row['Suma de Agentes Requeridos Erlang']))

# Convertir a numpy
demand_matrix = np.array(required_resources, dtype=float)

# Aplicar configuración de aprendizaje adaptativo ahora que tenemos demand_matrix
if optimization_profile == "Aprendizaje Adaptativo":
    adaptive_config = get_adaptive_params(demand_matrix, TARGET_COVERAGE)
    
    # Aplicar parámetros automáticamente sin intervención del usuario
    agent_limit_factor = adaptive_config["agent_limit_factor"]
    excess_penalty = adaptive_config["excess_penalty"]
    peak_bonus = adaptive_config["peak_bonus"]
    critical_bonus = adaptive_config["critical_bonus"]
    
    if adaptive_config.get("learned", False):
        evolution_step = adaptive_config.get("evolution_step", "unknown")
        st.sidebar.success(f"🧠 **IA Evolutiva Activa**")
        st.sidebar.info(f"📊 **Evolución #{adaptive_config['runs_count']}:**\n- Estrategia: {evolution_step}\n- Parámetros evolucionando\n- Mejora continua activada")
    else:
        st.sidebar.info("🆕 **IA Iniciando Evolución**\nPrimera ejecución - estableciendo baseline")

# Actualizar configuración mostrada
st.sidebar.info(f"""
📋 **Configuración Actual:**
- Límite agentes: /{agent_limit_factor}
- Penalización exceso: {excess_penalty}x
- Bonus horas pico: {peak_bonus}x
- Bonus días críticos: {critical_bonus}x
""")
daily_demand = demand_matrix.sum(axis=1)
hourly_demand = demand_matrix.sum(axis=0)

# Identificar días activos/inactivos
ACTIVE_DAYS = [d for d in range(7) if daily_demand[d] > 0]
INACTIVE_DAYS = [d for d in range(7) if daily_demand[d] == 0]
WORKING_DAYS = len(ACTIVE_DAYS)

# Análisis de ventana horaria
active_hours = np.where(hourly_demand > 0)[0]
first_hour = int(active_hours.min()) if len(active_hours) > 0 else 8
last_hour = int(active_hours.max()) if len(active_hours) > 0 else 20
OPERATING_HOURS = last_hour - first_hour + 1

# Análisis de picos
peak_demand = demand_matrix.max()
avg_demand = demand_matrix[ACTIVE_DAYS].mean()

st.info(f"""
📊 **Análisis de demanda:**
- Días activos: {', '.join([dias_semana[d] for d in ACTIVE_DAYS])} ({WORKING_DAYS} días)
- Horario operativo: {first_hour}:00 - {last_hour}:00 ({OPERATING_HOURS} horas)
- Demanda total semanal: {daily_demand.sum():.0f} agentes-hora
- Demanda promedio (días activos): {daily_demand[ACTIVE_DAYS].mean():.0f} agentes-hora/día
- Pico de demanda: {peak_demand:.0f} agentes simultáneos
- **Break configurado**: {break_from_start}h desde inicio, {break_from_end}h antes del fin
""")

# Mostrar configuración de turnos PT si está habilitada
if use_pt:
    pt_config = []
    if allow_pt_4h:
        pt_config.append("4h×6días")
    if allow_pt_6h:
        pt_config.append("6h×4días") 
    if allow_pt_5h:
        pt_config.append("5h×5días")
    
    if pt_config:
        st.info(f"⏰ **Turnos PT habilitados**: {', '.join(pt_config)}")
    else:
        st.warning("⚠️ Part Time está habilitado pero no hay turnos PT seleccionados")

# Estimación realista de agentes necesarios
total_demand = daily_demand.sum()
avg_hours_ft = 42  # 48h - 6h breaks
avg_hours_pt = 20  # 24h - 4h breaks
estimated_agents = int(total_demand / ((avg_hours_ft + avg_hours_pt) / 2) * 1.05)  # 5% buffer
max_simultaneous = int(peak_demand)

# Análisis dinámico de patrones críticos
daily_totals = demand_matrix.sum(axis=1)
hourly_totals = demand_matrix.sum(axis=0)
critical_days_analysis = np.argsort(daily_totals)[-2:] if len(daily_totals) > 1 else [np.argmax(daily_totals)]
peak_threshold = np.percentile(hourly_totals[hourly_totals > 0], 75) if np.any(hourly_totals > 0) else 0
peak_hours_analysis = np.where(hourly_totals >= peak_threshold)[0]

st.info(f"📊 **Análisis realista**:")
st.info(f"- Agentes simultáneos máximos: {max_simultaneous}")
st.info(f"- Total agentes estimados: ~{estimated_agents} para {total_demand:.0f} agentes-hora")
st.info(f"- Ratio eficiencia: {total_demand/estimated_agents:.1f} horas productivas por agente")

st.info(f"🎯 **Análisis de patrones críticos**:")
st.info(f"- Días críticos: {', '.join([dias_semana[d] for d in critical_days_analysis if d < len(dias_semana)])}")
st.info(f"- Horas pico: {peak_hours_analysis[0]:02d}:00 - {peak_hours_analysis[-1]:02d}:00" if len(peak_hours_analysis) > 0 else "- Horas pico: No identificadas")
st.info(f"- Perfil seleccionado: {optimization_profile}")

# ——————————————————————————————————————————————————————————————
# Sistema de Aprendizaje Adaptativo
# ——————————————————————————————————————————————————————————————

def create_demand_signature(demand_matrix):
    """Crea una firma única para el patrón de demanda"""
    # Normalizar y crear hash del patrón de demanda
    normalized = demand_matrix / (demand_matrix.max() + 1e-8)
    signature = hashlib.md5(normalized.tobytes()).hexdigest()[:16]
    return signature

def load_learning_history():
    """Carga el historial de aprendizaje"""
    try:
        if os.path.exists('learning_history.json'):
            with open('learning_history.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"No se pudo cargar el historial de aprendizaje: {e}")
    return {}

def save_learning_history(history):
    """Guarda el historial de aprendizaje"""
    try:
        with open('learning_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        st.warning(f"No se pudo guardar el historial de aprendizaje: {e}")

def get_adaptive_parameters(demand_signature, learning_history):
    """Obtiene parámetros adaptativos basados en el historial"""
    if demand_signature in learning_history:
        # Usar parámetros aprendidos
        learned = learning_history[demand_signature]
        best_run = min(learned['runs'], key=lambda x: x['score'])
        
        return {
            'agent_limit_factor': best_run['params']['agent_limit_factor'],
            'excess_penalty': best_run['params']['excess_penalty'],
            'peak_bonus': best_run['params']['peak_bonus'],
            'critical_bonus': best_run['params']['critical_bonus']
        }
    else:
        # Parámetros iniciales equilibrados
        return {
            'agent_limit_factor': 22,
            'excess_penalty': 0.5,
            'peak_bonus': 1.5,
            'critical_bonus': 2.0
        }

def update_learning_history(demand_signature, params, results, learning_history):
    """Actualiza el historial con nuevos resultados"""
    if demand_signature not in learning_history:
        learning_history[demand_signature] = {'runs': []}
    
    # Calcular score de calidad (menor es mejor)
    score = results['understaffing'] + results['overstaffing'] * 0.3
    
    run_data = {
        'params': params,
        'score': score,
        'total_agents': results['total_agents'],
        'coverage': results['coverage_percentage'],
        'timestamp': time.time()
    }
    
    learning_history[demand_signature]['runs'].append(run_data)
    
    # Mantener solo los últimos 10 runs
    if len(learning_history[demand_signature]['runs']) > 10:
        learning_history[demand_signature]['runs'] = \
            learning_history[demand_signature]['runs'][-10:]
    
    return learning_history

# Aplicar sistema de aprendizaje si está habilitado
if use_learning:
    demand_signature = create_demand_signature(demand_matrix)
    learning_history = load_learning_history()
    
    adaptive_params = get_adaptive_parameters(demand_signature, learning_history)
    agent_limit_factor = adaptive_params['agent_limit_factor']
    excess_penalty = adaptive_params['excess_penalty']
    peak_bonus = adaptive_params['peak_bonus']
    critical_bonus = adaptive_params['critical_bonus']
    
    # Mostrar información de aprendizaje
    if demand_signature in learning_history:
        runs_count = len(learning_history[demand_signature]['runs'])
        best_score = min(run['score'] for run in learning_history[demand_signature]['runs'])
        st.info(f"🧠 **Aprendizaje activo**: {runs_count} ejecuciones previas, mejor score: {best_score:.1f}")
    else:
        st.info("🌱 **Primer análisis**: Iniciando aprendizaje para este patrón de demanda")

# ——————————————————————————————————————————————————————————————
# 4. Generación de patrones CORREGIDA (basada en el generador 2025)
# ——————————————————————————————————————————————————————————————

def get_optimal_break_time(start_hour, shift_duration, day, demand_day):
    """
    Selecciona el mejor horario de break para un día específico según la demanda
    """
    break_earliest = start_hour + break_from_start
    break_latest = start_hour + shift_duration - break_from_end
    
    if break_latest <= break_earliest:
        return break_earliest
    
    # Generar opciones de break cada 30 minutos
    break_options = []
    current_time = break_earliest
    while current_time <= break_latest:
        break_options.append(current_time)
        current_time += 0.5
    
    # Evaluar cada opción según la demanda del día
    best_break = break_earliest
    min_impact = float('inf')
    
    for break_time in break_options:
        break_hour = int(break_time) % 24
        if break_hour < len(demand_day):
            impact = demand_day[break_hour]  # Menor demanda = mejor momento para break
            if impact < min_impact:
                min_impact = impact
                best_break = break_time
    
    return best_break


def generate_shifts_coverage_corrected(*, max_patterns: int | None = None, batch_size: int | None = None):
    """
    Genera patrones semanales completos con breaks variables por día
    y permite limitar el número máximo generado.
    """
    # Crear barra de progreso para generación de patrones
    pattern_progress = st.progress(0)
    pattern_status = st.empty()
    
    shifts_coverage = {}
    seen_patterns = set()
    total_patterns = 0
    current_patterns = 0
    
    # Horarios de inicio optimizados
    step = 0.5
    slot_minutes = 60
    if optimization_profile == "JEAN Personalizado":
        slot_minutes = template_cfg.get("slot_duration_minutes", 30)
        step = slot_minutes / 60
    slots_per_day = 24 * (60 // slot_minutes)
    if max_patterns is None:
        max_patterns = memory_limit_patterns(slots_per_day)

    start_hours = [h for h in np.arange(0, 24, step) if h <= 23.5]

    # Perfil JEAN Personalizado: leer patrones desde JSON y retornar
    if optimization_profile == "JEAN Personalizado":
        shifts_coverage = load_shift_patterns(
            template_cfg,
            start_hours=start_hours,
            break_from_start=break_from_start,
            break_from_end=break_from_end,
            slot_duration_minutes=slot_minutes,
            max_patterns=max_patterns,
        )

        if not use_ft:
            shifts_coverage = {
                k: v for k, v in shifts_coverage.items() if not k.startswith("FT")
            }
        if not use_pt:
            shifts_coverage = {
                k: v for k, v in shifts_coverage.items() if not k.startswith("PT")
            }

        pattern_progress.progress(1.0)
        pattern_status.text(
            f"Generados {len(shifts_coverage)} patrones personalizados"
        )
        time.sleep(1)
        pattern_progress.empty()
        pattern_status.empty()
        if batch_size:
            if shifts_coverage:
                yield shifts_coverage
        else:
            yield shifts_coverage
        return
    
    # Calcular total de patrones expandido
    total_patterns = 0
    if use_ft:
        if allow_8h:
            total_patterns += len(start_hours) * len(ACTIVE_DAYS)
        if allow_10h8:
            total_patterns += len(start_hours[::2]) * len(ACTIVE_DAYS) * 5
    if use_pt:
        if allow_pt_4h:
            total_patterns += len(start_hours[::2]) * 35  # Múltiples combinaciones
        if allow_pt_6h:
            total_patterns += len(start_hours[::3]) * 35
        if allow_pt_5h:
            total_patterns += len(start_hours[::3]) * 9
    
    pattern_status.text(f"Iniciando generación de {total_patterns} patrones...")
    
    # ===== TURNOS FULL TIME =====
    if use_ft:
        # 8 horas - 6 días de trabajo
        if allow_8h:
            for start_hour in start_hours:
                for dso_day in ACTIVE_DAYS:
                    working_days = [d for d in ACTIVE_DAYS if d != dso_day][:6]
                    if len(working_days) >= 6 and 8 * len(working_days) <= 48:
                        weekly_pattern = generate_weekly_pattern(
                            start_hour, 8, working_days, dso_day
                        )
                        pat_key = weekly_pattern.tobytes()
                        if pat_key in seen_patterns:
                            continue
                        seen_patterns.add(pat_key)
                        shift_name = f"FT8_{start_hour:04.1f}_DSO{dso_day}"
                        shifts_coverage[shift_name] = weekly_pattern
                        if batch_size and len(shifts_coverage) >= batch_size:
                            yield shifts_coverage
                            shifts_coverage = {}

                        current_patterns += 1
                        if total_patterns > 0:
                            pattern_progress.progress(current_patterns / total_patterns)
                            pattern_status.text(f"Generando patrones FT8: {current_patterns}/{total_patterns}")
                        if max_patterns is not None and len(shifts_coverage) >= max_patterns:
                            truncated = total_patterns - len(shifts_coverage)
                            pattern_progress.progress(1.0)
                            ft_count = len([k for k in shifts_coverage if k.startswith('FT')])
                            pt_count = len([k for k in shifts_coverage if k.startswith('PT')])
                            pattern_status.text(
                                f"Generados {len(shifts_coverage)} patrones: {ft_count} FT, {pt_count} PT (truncados {truncated})"
                            )
                            time.sleep(1)
                            pattern_progress.empty()
                            pattern_status.empty()
                            if batch_size:
                                if shifts_coverage:
                                    yield shifts_coverage
                            else:
                                yield shifts_coverage
                            return
        


        # 10h + un día de 8h - 5 días de trabajo
        if allow_10h8:
            for start_hour in start_hours[::2]:
                for dso_day in ACTIVE_DAYS:
                    working_days = [d for d in ACTIVE_DAYS if d != dso_day][:5]
                    if len(working_days) >= 5:
                        for eight_day in working_days:
                            weekly_pattern = generate_weekly_pattern_10h8(
                                start_hour, working_days, eight_day
                            )
                            pat_key = weekly_pattern.tobytes()
                            if pat_key in seen_patterns:
                                continue
                            seen_patterns.add(pat_key)
                            shift_name = (
                                f"FT10p8_{start_hour:04.1f}_DSO{dso_day}_8{eight_day}"
                            )
                            shifts_coverage[shift_name] = weekly_pattern
                            if batch_size and len(shifts_coverage) >= batch_size:
                                yield shifts_coverage
                                shifts_coverage = {}
                            if max_patterns is not None and len(shifts_coverage) >= max_patterns:
                                truncated = total_patterns - len(shifts_coverage)
                                pattern_progress.progress(1.0)
                                ft_count = len([k for k in shifts_coverage if k.startswith('FT')])
                                pt_count = len([k for k in shifts_coverage if k.startswith('PT')])
                                pattern_status.text(
                                    f"Generados {len(shifts_coverage)} patrones: {ft_count} FT, {pt_count} PT (truncados {truncated})"
                                )
                                time.sleep(1)
                                pattern_progress.empty()
                                pattern_status.empty()
                                if batch_size:
                                    if shifts_coverage:
                                        yield shifts_coverage
                                else:
                                    yield shifts_coverage
                                return
    
    # ===== TURNOS PART TIME =====
    if use_pt:
        # 4 horas - múltiples combinaciones de días
        if allow_pt_4h:
            for start_hour in start_hours[::2]:  # Cada 1 hora
                for num_days in [4, 5, 6]:
                    if num_days <= len(ACTIVE_DAYS) and 4 * num_days <= 24:
                        for working_combo in combinations(ACTIVE_DAYS, num_days):
                            weekly_pattern = generate_weekly_pattern_simple(
                                start_hour, 4, list(working_combo)
                            )
                            pat_key = weekly_pattern.tobytes()
                            if pat_key in seen_patterns:
                                continue
                            seen_patterns.add(pat_key)
                            shift_name = f"PT4_{start_hour:04.1f}_DAYS{''.join(map(str,working_combo))}"
                            shifts_coverage[shift_name] = weekly_pattern
                            if batch_size and len(shifts_coverage) >= batch_size:
                                yield shifts_coverage
                                shifts_coverage = {}
                            if max_patterns is not None and len(shifts_coverage) >= max_patterns:
                                truncated = total_patterns - len(shifts_coverage)
                                pattern_progress.progress(1.0)
                                ft_count = len([k for k in shifts_coverage if k.startswith('FT')])
                                pt_count = len([k for k in shifts_coverage if k.startswith('PT')])
                                pattern_status.text(
                                    f"Generados {len(shifts_coverage)} patrones: {ft_count} FT, {pt_count} PT (truncados {truncated})"
                                )
                                time.sleep(1)
                                pattern_progress.empty()
                                pattern_status.empty()
                                if batch_size:
                                    if shifts_coverage:
                                        yield shifts_coverage
                                else:
                                    yield shifts_coverage
                                return
        
        # 6 horas - combinaciones de 4 días (24h/sem)
        if allow_pt_6h:
            for start_hour in start_hours[::3]:  # Cada 1.5 horas
                for num_days in [4]:
                    if num_days <= len(ACTIVE_DAYS) and 6 * num_days <= 24:
                        for working_combo in combinations(ACTIVE_DAYS, num_days):
                            weekly_pattern = generate_weekly_pattern_simple(
                                start_hour, 6, list(working_combo)
                            )
                            pat_key = weekly_pattern.tobytes()
                            if pat_key in seen_patterns:
                                continue
                            seen_patterns.add(pat_key)
                            shift_name = f"PT6_{start_hour:04.1f}_DAYS{''.join(map(str,working_combo))}"
                            shifts_coverage[shift_name] = weekly_pattern
                            if batch_size and len(shifts_coverage) >= batch_size:
                                yield shifts_coverage
                                shifts_coverage = {}
                            if max_patterns is not None and len(shifts_coverage) >= max_patterns:
                                truncated = total_patterns - len(shifts_coverage)
                                pattern_progress.progress(1.0)
                                ft_count = len([k for k in shifts_coverage if k.startswith('FT')])
                                pt_count = len([k for k in shifts_coverage if k.startswith('PT')])
                                pattern_status.text(
                                    f"Generados {len(shifts_coverage)} patrones: {ft_count} FT, {pt_count} PT (truncados {truncated})"
                                )
                                time.sleep(1)
                                pattern_progress.empty()
                                pattern_status.empty()
                                if batch_size:
                                    if shifts_coverage:
                                        yield shifts_coverage
                                else:
                                    yield shifts_coverage
                                return
        
        # 5 horas - combinaciones de 5 días (~25h/sem)
        if allow_pt_5h:
            for start_hour in start_hours[::3]:  # Cada 1.5 horas
                for num_days in [5]:
                    if num_days <= len(ACTIVE_DAYS) and 5 * num_days <= 25:
                        for working_combo in combinations(ACTIVE_DAYS, num_days):
                            weekly_pattern = generate_weekly_pattern_pt5(
                                start_hour, list(working_combo)
                            )
                            pat_key = weekly_pattern.tobytes()
                            if pat_key in seen_patterns:
                                continue
                            seen_patterns.add(pat_key)
                            shift_name = f"PT5_{start_hour:04.1f}_DAYS{''.join(map(str,working_combo))}"
                            shifts_coverage[shift_name] = weekly_pattern
                            if batch_size and len(shifts_coverage) >= batch_size:
                                yield shifts_coverage
                                shifts_coverage = {}
                            if max_patterns is not None and len(shifts_coverage) >= max_patterns:
                                truncated = total_patterns - len(shifts_coverage)
                                pattern_progress.progress(1.0)
                                ft_count = len([k for k in shifts_coverage if k.startswith('FT')])
                                pt_count = len([k for k in shifts_coverage if k.startswith('PT')])
                                pattern_status.text(
                                    f"Generados {len(shifts_coverage)} patrones: {ft_count} FT, {pt_count} PT (truncados {truncated})"
                                )
                                time.sleep(1)
                                pattern_progress.empty()
                                pattern_status.empty()
                                if batch_size:
                                    if shifts_coverage:
                                        yield shifts_coverage
                                else:
                                    yield shifts_coverage
                                return
    
    # Completar barra de progreso
    pattern_progress.progress(1.0)
    ft_count = len([k for k in shifts_coverage.keys() if k.startswith('FT')])
    pt_count = len([k for k in shifts_coverage.keys() if k.startswith('PT')])
    truncated = 0
    if max_patterns is not None and len(shifts_coverage) < total_patterns:
        truncated = total_patterns - len(shifts_coverage)
    if truncated:
        pattern_status.text(
            f"Generados {len(shifts_coverage)} patrones: {ft_count} FT, {pt_count} PT (truncados {truncated})"
        )
    else:
        pattern_status.text(
            f"Generados {len(shifts_coverage)} patrones: {ft_count} FT, {pt_count} PT"
        )
    
    # Limpiar elementos de progreso
    time.sleep(1)
    pattern_progress.empty()
    pattern_status.empty()
    
    if shifts_coverage:
        yield shifts_coverage
    return


def generate_shifts_coverage_optimized(
    demand_matrix: np.ndarray,
    *,
    max_patterns: int | None = None,
    batch_size: int = 2000,
    quality_threshold: int = 0,
) -> Iterable[dict[str, np.ndarray]]:
    """Generate and score patterns in batches.

    Patterns are produced using ``generate_shifts_coverage_corrected`` and
    filtered immediately based on ``score_pattern``.  Only patterns with a score
    at least ``quality_threshold`` are yielded.  Duplicates are removed using a
    set of hashes and generation stops once ``max_patterns`` patterns have been
    emitted.
    """

    progress = st.progress(0.0)
    status = st.empty()

    selected = 0
    start_time = time.time()
    seen: set[bytes] = set()

    inner = generate_shifts_coverage_corrected(batch_size=batch_size)
    for raw_batch in inner:
        batch: dict[str, np.ndarray] = {}
        for name, pat in raw_batch.items():
            key = hashlib.md5(pat).digest()
            if key in seen:
                continue
            seen.add(key)
            if score_pattern(pat, demand_matrix) >= quality_threshold:
                batch[name] = pat
                selected += 1
                if max_patterns is not None and selected >= max_patterns:
                    break
        if batch:
            if max_patterns:
                progress.progress(min(selected / max_patterns, 1.0))
                status.text(f"Patrones {selected}/{max_patterns}")
            else:
                status.text(f"Patrones {selected}")
            show_generation_progress(selected, start_time)
            yield batch
            gc.collect()
        if max_patterns is not None and selected >= max_patterns:
            break

    progress.progress(1.0)
    status.text(f"Total patrones válidos: {selected}")
    time.sleep(1)
    progress.empty()
    status.empty()


# ——————————————————————————————————————————————————————————————
# 5. Función de optimización
# ——————————————————————————————————————————————————————————————

def optimize_with_phased_strategy(shifts_coverage, demand_matrix):
    """Optimización en fases utilizando la estrategia FT→PT"""
    if not PULP_AVAILABLE:
        return optimize_schedule_greedy(shifts_coverage, demand_matrix)

    if use_ft and use_pt:
        return optimize_ft_then_pt_strategy(shifts_coverage, demand_matrix)
    elif use_ft and not use_pt:
        ft_shifts = {k: v for k, v in shifts_coverage.items() if k.startswith('FT')}
        return optimize_single_type(ft_shifts, demand_matrix, "FT")
    elif use_pt and not use_ft:
        pt_shifts = {k: v for k, v in shifts_coverage.items() if k.startswith('PT')}
        return optimize_single_type(pt_shifts, demand_matrix, "PT")
    else:
        return {}, "NO_CONTRACT_TYPE_SELECTED"


def optimize_single_type(shifts, demand_matrix, shift_type):
    """Optimiza un solo tipo usando parámetros del perfil"""
    if not shifts:
        return {}, f"NO_{shift_type}_SHIFTS"
    
    prob = pulp.LpProblem(f"{shift_type}_Only", pulp.LpMinimize)
    
    # Variables con límite basado en agent_limit_factor
    max_per_shift = max(5, int(demand_matrix.sum() / agent_limit_factor))
    shift_vars = {}
    for shift in shifts.keys():
        shift_vars[shift] = pulp.LpVariable(f"{shift_type.lower()}_{shift}", 0, max_per_shift, pulp.LpInteger)
    
    # Variables de déficit y exceso
    deficit_vars = {}
    excess_vars = {}
    hours = demand_matrix.shape[1]
    for day in range(7):
        for hour in range(hours):
            deficit_vars[(day, hour)] = pulp.LpVariable(f"{shift_type.lower()}_deficit_{day}_{hour}", 0, None)
            excess_vars[(day, hour)] = pulp.LpVariable(f"{shift_type.lower()}_excess_{day}_{hour}", 0, None)
    
    # Objetivo con parámetros del perfil
    total_deficit = pulp.lpSum([deficit_vars[(day, hour)] for day in range(7) for hour in range(hours)])
    total_excess = pulp.lpSum([excess_vars[(day, hour)] for day in range(7) for hour in range(hours)])
    total_agents = pulp.lpSum([shift_vars[shift] for shift in shifts.keys()])
    
    # Bonificaciones por días críticos y horas pico
    critical_bonus_value = 0
    peak_bonus_value = 0
    
    # Días críticos
    daily_demand = demand_matrix.sum(axis=1)
    if len(daily_demand) > 0 and daily_demand.max() > 0:
        critical_day = np.argmax(daily_demand)
        for hour in range(hours):
            if demand_matrix[critical_day, hour] > 0:
                critical_bonus_value -= deficit_vars[(critical_day, hour)] * critical_bonus
    
    # Horas pico
    hourly_demand = demand_matrix.sum(axis=0)
    if len(hourly_demand) > 0 and hourly_demand.max() > 0:
        peak_hour = np.argmax(hourly_demand)
        for day in range(7):
            if demand_matrix[day, peak_hour] > 0:
                peak_bonus_value -= deficit_vars[(day, peak_hour)] * peak_bonus
    
    # Función objetivo completa
    prob += (total_deficit * 1000 + 
             total_excess * excess_penalty + 
             total_agents * 0.1 + 
             critical_bonus_value + 
             peak_bonus_value)
    
    # Restricciones de cobertura
    for day in range(7):
        for hour in range(hours):
            coverage = pulp.lpSum([
                shift_vars[shift] * shifts[shift][day * hours + hour]
                for shift in shifts.keys()
            ])
            demand = demand_matrix[day, hour]
            
            prob += coverage + deficit_vars[(day, hour)] >= demand
            prob += coverage - excess_vars[(day, hour)] <= demand
    
    # Restricciones adicionales según perfil
    if excess_penalty > 5:  # Perfiles estrictos como "100% Exacto"
        prob += total_excess <= demand_matrix.sum() * 0.02
    elif excess_penalty > 2:
        prob += total_excess <= demand_matrix.sum() * 0.05
    
    # Resolver
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=TIME_SOLVER))
    
    # Extraer resultados
    assignments = {}
    if prob.status == pulp.LpStatusOptimal:
        for shift in shifts.keys():
            value = int(shift_vars[shift].varValue or 0)
            if value > 0:
                assignments[shift] = value
    
    return assignments, f"{shift_type}_ONLY_OPTIMAL"

def optimize_with_precision_targeting(shifts_coverage, demand_matrix):
    """Optimización ultra-precisa para cobertura exacta"""
    if not PULP_AVAILABLE:
        return optimize_schedule_greedy(shifts_coverage, demand_matrix)
    
    try:
        shifts_list = list(shifts_coverage.keys())
        if not shifts_list:
            return {}, "NO_SHIFTS"
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("🎯 Configurando optimización de precisión...")
        
        prob = pulp.LpProblem("Precision_Scheduling", pulp.LpMinimize)
        
        # Variables con límites dinámicos basados en demanda
        total_demand = demand_matrix.sum()
        peak_demand = demand_matrix.max()
        max_per_shift = max(15, int(total_demand / max(1, len(shifts_list) / 10)))
        
        shift_vars = {}
        for shift in shifts_list:
            shift_vars[shift] = pulp.LpVariable(f"shift_{shift}", 0, max_per_shift, pulp.LpInteger)
        
        progress_bar.progress(0.2)
        status_text.text("📊 Analizando patrones de demanda...")
        
        # Variables de déficit y exceso con pesos dinámicos
        deficit_vars = {}
        excess_vars = {}
        hours = demand_matrix.shape[1]
        for day in range(7):
            for hour in range(hours):
                deficit_vars[(day, hour)] = pulp.LpVariable(f"deficit_{day}_{hour}", 0, None)
                excess_vars[(day, hour)] = pulp.LpVariable(f"excess_{day}_{hour}", 0, None)
        
        # Análisis de patrones críticos
        daily_totals = demand_matrix.sum(axis=1)
        hourly_totals = demand_matrix.sum(axis=0)
        critical_days = np.argsort(daily_totals)[-2:] if len(daily_totals) > 1 else [np.argmax(daily_totals)]
        peak_hours = np.where(hourly_totals >= np.percentile(hourly_totals[hourly_totals > 0], 80))[0]
        
        progress_bar.progress(0.4)
        status_text.text("⚙️ Construyendo función objetivo ultra-precisa...")
        
        # Función objetivo ultra-precisa
        total_deficit = pulp.lpSum([deficit_vars[(day, hour)] for day in range(7) for hour in range(hours)])
        total_agents = pulp.lpSum([shift_vars[shift] for shift in shifts_list])
        
        # Penalización de exceso ultra-inteligente
        smart_excess_penalty = 0
        for day in range(7):
            for hour in range(hours):
                demand_val = demand_matrix[day, hour]
                if demand_val == 0:
                    # Prohibición total de exceso en horas sin demanda
                    smart_excess_penalty += excess_vars[(day, hour)] * 50000
                elif demand_val <= 2:
                    # Penalización muy alta para demanda baja
                    smart_excess_penalty += excess_vars[(day, hour)] * (excess_penalty * 100)
                elif demand_val <= 5:
                    # Penalización moderada
                    smart_excess_penalty += excess_vars[(day, hour)] * (excess_penalty * 20)
                else:
                    # Penalización mínima para alta demanda
                    smart_excess_penalty += excess_vars[(day, hour)] * (excess_penalty * 5)
        
        # Bonificaciones ultra-precisas para patrones críticos
        precision_bonus = 0
        
        # Bonificar cobertura en días críticos
        for critical_day in critical_days:
            if critical_day < 7:
                day_multiplier = min(5.0, daily_totals[critical_day] / max(1, daily_totals.mean()))
                for hour in range(hours):
                    if demand_matrix[critical_day, hour] > 0:
                        precision_bonus -= deficit_vars[(critical_day, hour)] * (critical_bonus * 100 * day_multiplier)
        
        # Bonificar cobertura en horas pico
        for hour in peak_hours:
            if hour < hours:
                hour_multiplier = min(3.0, hourly_totals[hour] / max(1, hourly_totals.mean()))
                for day in range(7):
                    if demand_matrix[day, hour] > 0:
                        precision_bonus -= deficit_vars[(day, hour)] * (peak_bonus * 50 * hour_multiplier)
        
        # Objetivo final ultra-preciso
        prob += (total_deficit * 100000 +      # Prioridad máxima: eliminar déficit
                smart_excess_penalty +         # Control inteligente de exceso
                total_agents * 0.01 +          # Minimizar agentes
                precision_bonus)               # Bonificaciones precisas
        
        progress_bar.progress(0.6)
        status_text.text("🔗 Aplicando restricciones de precisión...")
        
        # Restricciones de cobertura exacta
        for day in range(7):
            for hour in range(hours):
                coverage = pulp.lpSum([
                    shift_vars[shift] * shifts_coverage[shift][day * hours + hour]
                    for shift in shifts_list
                ])
                demand = demand_matrix[day, hour]
                
                # Restricciones básicas
                prob += coverage + deficit_vars[(day, hour)] >= demand
                prob += coverage - excess_vars[(day, hour)] <= demand
                
                # Restricción más suave: limitar exceso donde no hay demanda
                if demand == 0:
                    prob += coverage <= 1  # Permitir máximo 1 agente en horas sin demanda
        
        # Límite dinámico de agentes ajustado según perfil
        if optimization_profile in ("JEAN", "JEAN Personalizado"):
            dynamic_agent_limit = max(
                int(total_demand / max(1, agent_limit_factor)),
                int(peak_demand * 1.1),
            )
        else:
            dynamic_agent_limit = max(
                int(total_demand / max(1, agent_limit_factor - 2)),
                int(peak_demand * 2),
            )
        prob += total_agents <= dynamic_agent_limit
        
        # Control de exceso global más flexible
        total_excess = pulp.lpSum([excess_vars[(day, hour)] for day in range(7) for hour in range(hours)])
        
        # Restricciones más flexibles para encontrar soluciones
        prob += total_excess <= total_demand * 0.10  # 10% exceso permitido
        
        # Equilibrio por día más flexible
        for day in range(7):
            day_demand = demand_matrix[day].sum()
            if day_demand > 0:
                day_coverage = pulp.lpSum([
                    shift_vars[shift]
                    * np.sum(
                        np.array(shifts_coverage[shift]).reshape(
                            7, len(shifts_coverage[shift]) // 7
                        )[day]
                    )
                    for shift in shifts_list
                ])
                # Control más flexible por día
                prob += day_coverage <= day_demand * 1.15  # Máximo 15% exceso por día
                prob += day_coverage >= day_demand * 0.85  # Mínimo 85% cobertura por día
        
        progress_bar.progress(0.8)
        status_text.text("⚡ Ejecutando solver de precisión...")
        
        # Solver con configuración más flexible
        solver = pulp.PULP_CBC_CMD(
            msg=0, 
            timeLimit=TIME_SOLVER,
            gapRel=0.02,   # 2% gap de optimalidad (más flexible)
            threads=4,
            presolve=1,
            cuts=1
        )
        prob.solve(solver)
        
        # Extraer solución
        assignments = {}
        if prob.status == pulp.LpStatusOptimal:
            for shift in shifts_list:
                value = int(shift_vars[shift].varValue or 0)
                if value > 0:
                    assignments[shift] = value
            method = "PRECISION_TARGETING"
        elif prob.status == pulp.LpStatusInfeasible:
            st.warning("⚠️ Problema infactible, relajando restricciones...")
            # Intentar con restricciones más relajadas
            return optimize_with_relaxed_constraints(shifts_coverage, demand_matrix)
        else:
            st.warning(f"⚠️ Solver status: {prob.status}, usando fallback inteligente")
            return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Optimización de precisión completada")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return assignments, method
        
    except Exception as e:
        st.error(f"Error en optimización de precisión: {str(e)}")
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix)

def optimize_ft_then_pt_strategy(shifts_coverage, demand_matrix):
    """Estrategia 2 fases: FT sin exceso, luego PT para completar"""
    try:
        # Separar turnos por tipo
        ft_shifts = {k: v for k, v in shifts_coverage.items() if k.startswith('FT')}
        pt_shifts = {k: v for k, v in shifts_coverage.items() if k.startswith('PT')}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # FASE 1: Optimizar FT sin exceso
        status_text.text("🏢 Fase 1: Full Time (SIN exceso)...")
        progress_bar.progress(0.3)
        
        ft_assignments = optimize_ft_no_excess(ft_shifts, demand_matrix)
        
        # Calcular cobertura FT
        ft_coverage = np.zeros_like(demand_matrix)
        for shift_name, count in ft_assignments.items():
            slots_per_day = len(ft_shifts[shift_name]) // 7
            pattern = np.array(ft_shifts[shift_name]).reshape(7, slots_per_day)
            ft_coverage += pattern * count
        
        # FASE 2: PT para completar déficit
        status_text.text("⏰ Fase 2: Part Time (completar déficit)...")
        progress_bar.progress(0.7)
        
        remaining_demand = np.maximum(0, demand_matrix - ft_coverage)
        pt_assignments = optimize_pt_complete(pt_shifts, remaining_demand)
        
        # Combinar resultados
        final_assignments = {**ft_assignments, **pt_assignments}
        
        progress_bar.progress(1.0)
        status_text.text("✅ Estrategia 2 fases completada")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return final_assignments, "FT_NO_EXCESS_THEN_PT"
        
    except Exception as e:
        st.error(f"Error en estrategia 2 fases: {str(e)}")
        return optimize_with_precision_targeting(shifts_coverage, demand_matrix)

def optimize_ft_no_excess(ft_shifts, demand_matrix):
    """Fase 1: FT con CERO exceso permitido"""
    if not ft_shifts:
        return {}
    
    prob = pulp.LpProblem("FT_No_Excess", pulp.LpMinimize)
    
    # Variables FT
    max_ft_per_shift = max(10, int(demand_matrix.sum() / agent_limit_factor))
    ft_vars = {}
    for shift in ft_shifts.keys():
        ft_vars[shift] = pulp.LpVariable(f"ft_{shift}", 0, max_ft_per_shift, pulp.LpInteger)
    
    # Solo variables de déficit (NO exceso)
    deficit_vars = {}
    hours = demand_matrix.shape[1]
    for day in range(7):
        for hour in range(hours):
            deficit_vars[(day, hour)] = pulp.LpVariable(f"ft_deficit_{day}_{hour}", 0, None)
    
    # Objetivo: minimizar déficit + agentes
    total_deficit = pulp.lpSum([deficit_vars[(day, hour)] for day in range(7) for hour in range(hours)])
    total_ft_agents = pulp.lpSum([ft_vars[shift] for shift in ft_shifts.keys()])
    
    prob += total_deficit * 1000 + total_ft_agents * 1
    
    # Restricciones: cobertura <= demanda (SIN exceso)
    for day in range(7):
        for hour in range(hours):
            coverage = pulp.lpSum([
                ft_vars[shift] * ft_shifts[shift][day * hours + hour]
                for shift in ft_shifts.keys()
            ])
            demand = demand_matrix[day, hour]
            
            # Cobertura + déficit >= demanda
            prob += coverage + deficit_vars[(day, hour)] >= demand
            # Cobertura <= demanda (SIN exceso)
            prob += coverage <= demand
    
    # Resolver
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=TIME_SOLVER//2))
    
    ft_assignments = {}
    if prob.status == pulp.LpStatusOptimal:
        for shift in ft_shifts.keys():
            value = int(ft_vars[shift].varValue or 0)
            if value > 0:
                ft_assignments[shift] = value
    
    return ft_assignments

def optimize_pt_complete(pt_shifts, remaining_demand):
    """Fase 2: PT para completar el déficit restante"""
    if not pt_shifts or remaining_demand.sum() == 0:
        return {}
    
    prob = pulp.LpProblem("PT_Complete", pulp.LpMinimize)
    
    # Variables PT
    max_pt_per_shift = max(10, int(remaining_demand.sum() / max(1, agent_limit_factor)))
    pt_vars = {}
    for shift in pt_shifts.keys():
        pt_vars[shift] = pulp.LpVariable(f"pt_{shift}", 0, max_pt_per_shift, pulp.LpInteger)
    
    # Variables de déficit y exceso
    deficit_vars = {}
    excess_vars = {}
    hours = remaining_demand.shape[1]
    for day in range(7):
        for hour in range(hours):
            deficit_vars[(day, hour)] = pulp.LpVariable(f"pt_deficit_{day}_{hour}", 0, None)
            excess_vars[(day, hour)] = pulp.LpVariable(f"pt_excess_{day}_{hour}", 0, None)
    
    # Objetivo: minimizar déficit, controlar exceso
    total_deficit = pulp.lpSum([deficit_vars[(day, hour)] for day in range(7) for hour in range(hours)])
    total_excess = pulp.lpSum([excess_vars[(day, hour)] for day in range(7) for hour in range(hours)])
    total_pt_agents = pulp.lpSum([pt_vars[shift] for shift in pt_shifts.keys()])
    
    prob += total_deficit * 1000 + total_excess * (excess_penalty * 20) + total_pt_agents * 1

    # Para el perfil JEAN no se permite ningún exceso
    if optimization_profile in ("JEAN", "JEAN Personalizado"):
        prob += total_excess == 0
    
    # Restricciones de cobertura
    for day in range(7):
        for hour in range(hours):
            coverage = pulp.lpSum([
                pt_vars[shift] * pt_shifts[shift][day * hours + hour]
                for shift in pt_shifts.keys()
            ])
            demand = remaining_demand[day, hour]
            
            prob += coverage + deficit_vars[(day, hour)] >= demand
            prob += coverage - excess_vars[(day, hour)] <= demand
    
    # Resolver
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=TIME_SOLVER//2))
    
    pt_assignments = {}
    if prob.status == pulp.LpStatusOptimal:
        for shift in pt_shifts.keys():
            value = int(pt_vars[shift].varValue or 0)
            if value > 0:
                pt_assignments[shift] = value
    
    return pt_assignments

def optimize_with_relaxed_constraints(shifts_coverage, demand_matrix):
    """Optimización con restricciones muy relajadas para problemas difíciles"""
    if not PULP_AVAILABLE:
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix)
    
    try:
        shifts_list = list(shifts_coverage.keys())
        if not shifts_list:
            return {}, "NO_SHIFTS"
        
        prob = pulp.LpProblem("Relaxed_Scheduling", pulp.LpMinimize)
        
        # Variables con límites muy generosos
        total_demand = demand_matrix.sum()
        max_per_shift = max(20, int(total_demand / 5))
        
        shift_vars = {}
        for shift in shifts_list:
            shift_vars[shift] = pulp.LpVariable(f"shift_{shift}", 0, max_per_shift, pulp.LpInteger)
        
        # Solo variables de déficit (sin restricciones de exceso)
        deficit_vars = {}
        for day in range(7):
            for hour in range(24):
                deficit_vars[(day, hour)] = pulp.LpVariable(f"deficit_{day}_{hour}", 0, None)
        
        # Objetivo simple: minimizar déficit
        total_deficit = pulp.lpSum([deficit_vars[(day, hour)] for day in range(7) for hour in range(24)])
        total_agents = pulp.lpSum([shift_vars[shift] for shift in shifts_list])
        
        prob += total_deficit * 1000 + total_agents * 0.1
        
        # Solo restricciones básicas de cobertura
        for day in range(7):
            for hour in range(24):
                coverage = pulp.lpSum([
                    shift_vars[shift] * shifts_coverage[shift][day * 24 + hour]
                    for shift in shifts_list
                ])
                demand = demand_matrix[day, hour]
                prob += coverage + deficit_vars[(day, hour)] >= demand
        
        # Límite muy generoso de agentes
        prob += total_agents <= int(total_demand / 3)
        
        # Resolver con configuración básica
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=TIME_SOLVER//2))
        
        assignments = {}
        if prob.status == pulp.LpStatusOptimal:
            for shift in shifts_list:
                value = int(shift_vars[shift].varValue or 0)
                if value > 0:
                    assignments[shift] = value
            return assignments, "RELAXED_CONSTRAINTS"
        else:
            return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix)
            
    except Exception as e:
        st.error(f"Error en optimización relajada: {str(e)}")
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix)

def optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix):
    """Solver greedy mejorado con lógica de precisión"""
    try:
        shifts_list = list(shifts_coverage.keys())
        assignments = {}
        current_coverage = np.zeros_like(demand_matrix)
        max_agents = max(50, int(demand_matrix.sum() / agent_limit_factor))
        
        # Análisis de patrones críticos
        daily_totals = demand_matrix.sum(axis=1)
        hourly_totals = demand_matrix.sum(axis=0)
        critical_days = np.argsort(daily_totals)[-2:] if len(daily_totals) > 1 else [np.argmax(daily_totals)]
        peak_hours = np.where(hourly_totals >= np.percentile(hourly_totals[hourly_totals > 0], 75))[0]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for iteration in range(max_agents):
            progress_bar.progress(iteration / max_agents)
            status_text.text(f"Greedy Mejorado {iteration + 1}/{max_agents}")
            
            best_shift = None
            best_score = -float('inf')
            
            for shift_name in shifts_list:
                try:
                    slots_per_day = len(shifts_coverage[shift_name]) // 7
                    base_pattern = np.array(shifts_coverage[shift_name]).reshape(7, slots_per_day)
                    new_coverage = current_coverage + base_pattern
                    
                    # Cálculo de score mejorado
                    current_deficit = np.maximum(0, demand_matrix - current_coverage)
                    new_deficit = np.maximum(0, demand_matrix - new_coverage)
                    deficit_reduction = np.sum(current_deficit - new_deficit)
                    
                    # Penalización inteligente de exceso
                    current_excess = np.maximum(0, current_coverage - demand_matrix)
                    new_excess = np.maximum(0, new_coverage - demand_matrix)
                    excess_increase = np.sum(new_excess - current_excess)
                    
                    # Penalización progresiva de exceso
                    smart_excess_penalty = 0
                    for day in range(7):
                        for hour in range(24):
                            if demand_matrix[day, hour] == 0 and new_excess[day, hour] > current_excess[day, hour]:
                                smart_excess_penalty += 1000  # Penalización extrema
                            elif demand_matrix[day, hour] <= 2:
                                smart_excess_penalty += (new_excess[day, hour] - current_excess[day, hour]) * excess_penalty * 10
                            else:
                                smart_excess_penalty += (new_excess[day, hour] - current_excess[day, hour]) * excess_penalty
                    
                    # Bonificaciones por patrones críticos
                    critical_bonus_score = 0
                    for critical_day in critical_days:
                        if critical_day < 7:
                            day_improvement = np.sum(current_deficit[critical_day] - new_deficit[critical_day])
                            critical_bonus_score += day_improvement * critical_bonus * 2
                    
                    peak_bonus_score = 0
                    for hour in peak_hours:
                        if hour < 24:
                            hour_improvement = np.sum(current_deficit[:, hour] - new_deficit[:, hour])
                            peak_bonus_score += hour_improvement * peak_bonus * 2
                    
                    # Score final mejorado
                    score = (deficit_reduction * 100 + 
                            critical_bonus_score + 
                            peak_bonus_score - 
                            smart_excess_penalty)
                    
                    if score > best_score:
                        best_score = score
                        best_shift = shift_name
                        best_pattern = base_pattern
                        
                except Exception:
                    continue
            
            # Criterio de parada mejorado
            if best_score <= 1.0 or np.sum(np.maximum(0, demand_matrix - current_coverage)) == 0:
                break
            
            if best_shift:
                if best_shift not in assignments:
                    assignments[best_shift] = 0
                assignments[best_shift] += 1
                current_coverage += best_pattern
        
        progress_bar.progress(1.0)
        status_text.text("Greedy mejorado completado")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return assignments, f"GREEDY_ENHANCED_{optimization_profile.upper()}"
        
    except Exception as e:
        st.error(f"Error en greedy mejorado: {str(e)}")
        return {}, "ERROR"


def optimize_direct_improved(shifts_coverage, demand_matrix):
    """Optimización directa mejorada para mejor cobertura"""
    try:
        shifts_list = list(shifts_coverage.keys())
        if not shifts_list:
            return {}, "NO_SHIFTS"
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("🎯 Configurando optimización directa...")
        
        # Crear problema
        prob = pulp.LpProblem("Direct_Optimization", pulp.LpMinimize)
        
        # Variables con límites más generosos
        max_per_shift = max(20, int(demand_matrix.sum() / 8))
        shift_vars = {}
        for shift in shifts_list:
            shift_vars[shift] = pulp.LpVariable(f"shift_{shift}", 0, max_per_shift, pulp.LpInteger)
        
        progress_bar.progress(0.3)
        status_text.text("📊 Definiendo variables de cobertura...")
        
        # Variables de déficit y exceso
        deficit_vars = {}
        excess_vars = {}
        for day in range(7):
            for hour in range(24):
                deficit_vars[(day, hour)] = pulp.LpVariable(f"deficit_{day}_{hour}", 0, None)
                excess_vars[(day, hour)] = pulp.LpVariable(f"excess_{day}_{hour}", 0, None)
        
        # Función objetivo mejorada - priorizar cobertura uniforme
        total_deficit = pulp.lpSum([deficit_vars[(day, hour)] for day in range(7) for hour in range(24)])
        total_excess = pulp.lpSum([excess_vars[(day, hour)] for day in range(7) for hour in range(24)])
        total_agents = pulp.lpSum([shift_vars[shift] for shift in shifts_list])
        
        # Función objetivo simplificada pero efectiva
        prob += (total_deficit * 50000 +     # Eliminar déficit es prioridad máxima
                total_excess * 500 +         # Controlar exceso fuertemente
                total_agents * 1)            # Minimizar agentes
        
        progress_bar.progress(0.6)
        status_text.text("🔗 Agregando restricciones de cobertura...")
        
        # Restricciones de cobertura
        for day in range(7):
            for hour in range(24):
                coverage = pulp.lpSum([
                    shift_vars[shift] * shifts_coverage[shift][day * 24 + hour]
                    for shift in shifts_list
                ])
                demand = demand_matrix[day, hour]
                
                # Restricciones de balance
                prob += coverage + deficit_vars[(day, hour)] >= demand
                prob += coverage - excess_vars[(day, hour)] <= demand
        
        # Restricciones mejoradas
        # Limitar exceso total
        prob += total_excess <= demand_matrix.sum() * 0.10  # Máximo 10% exceso
        
        # Limitar agentes totales para evitar sobreoptimización
        max_total_agents = int(demand_matrix.sum() / 8)  # Más generoso
        prob += total_agents <= max_total_agents
        
        progress_bar.progress(0.9)
        status_text.text("⚡ Resolviendo optimización...")
        
        # Resolver con configuración optimizada
        solver = pulp.PULP_CBC_CMD(
            msg=0, 
            timeLimit=TIME_SOLVER,
            gapRel=0.02,  # 2% gap de optimalidad
            threads=4
        )
        prob.solve(solver)
        
        # Extraer solución
        assignments = {}
        if prob.status == pulp.LpStatusOptimal:
            for shift in shifts_list:
                value = int(shift_vars[shift].varValue or 0)
                if value > 0:
                    assignments[shift] = value
            method = "DIRECT_IMPROVED"
        else:
            st.warning("⚠️ Solución no óptima, usando greedy")
            return optimize_schedule_greedy(shifts_coverage, demand_matrix)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Optimización completada")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return assignments, method
        
    except Exception as e:
        st.error(f"Error en optimización directa: {str(e)}")
        return optimize_schedule_greedy(shifts_coverage, demand_matrix)



def optimize_single_type_improved(shifts_coverage, demand_matrix, shift_type):
    """Optimización mejorada para un solo tipo de turno"""
    shifts = {k: v for k, v in shifts_coverage.items() if k.startswith(shift_type)}
    if not shifts:
        return {}, f"NO_{shift_type}_SHIFTS"
    
    prob = pulp.LpProblem(f"{shift_type}_Improved", pulp.LpMinimize)
    
    # Variables con límites generosos
    max_per_shift = max(15, int(demand_matrix.sum() / 10))
    shift_vars = {}
    for shift in shifts.keys():
        shift_vars[shift] = pulp.LpVariable(f"{shift_type.lower()}_{shift}", 0, max_per_shift, pulp.LpInteger)
    
    # Variables de déficit y exceso
    deficit_vars = {}
    excess_vars = {}
    for day in range(7):
        for hour in range(24):
            deficit_vars[(day, hour)] = pulp.LpVariable(f"{shift_type.lower()}_deficit_{day}_{hour}", 0, None)
            excess_vars[(day, hour)] = pulp.LpVariable(f"{shift_type.lower()}_excess_{day}_{hour}", 0, None)
    
    # Objetivo mejorado
    total_deficit = pulp.lpSum([deficit_vars[(day, hour)] for day in range(7) for hour in range(24)])
    total_excess = pulp.lpSum([excess_vars[(day, hour)] for day in range(7) for hour in range(24)])
    total_agents = pulp.lpSum([shift_vars[shift] for shift in shifts.keys()])
    
    prob += total_deficit * 10000 + total_excess * 200 + total_agents * 1
    
    # Restricciones de cobertura
    for day in range(7):
        for hour in range(24):
            coverage = pulp.lpSum([
                shift_vars[shift] * shifts[shift][day * 24 + hour]
                for shift in shifts.keys()
            ])
            demand = demand_matrix[day, hour]
            
            prob += coverage + deficit_vars[(day, hour)] >= demand
            prob += coverage - excess_vars[(day, hour)] <= demand
    
    # Límite de exceso
    prob += total_excess <= demand_matrix.sum() * 0.15
    
    # Resolver
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=TIME_SOLVER))
    
    assignments = {}
    if prob.status == pulp.LpStatusOptimal:
        for shift in shifts.keys():
            value = int(shift_vars[shift].varValue or 0)
            if value > 0:
                assignments[shift] = value
    
    return assignments, f"{shift_type}_IMPROVED"


def optimize_jean_search(shifts_coverage, demand_matrix, target_coverage=98.0, max_iterations=5, verbose=False):
    """Búsqueda iterativa para el perfil JEAN minimizando exceso y déficit."""
    global agent_limit_factor
    original_factor = agent_limit_factor

    best_assignments = {}
    best_method = ""
    best_score = float("inf")
    best_coverage = 0

    factor = agent_limit_factor
    iteration = 0
    while iteration < max_iterations and factor >= 1:
        agent_limit_factor = factor
        assignments, method = optimize_with_precision_targeting(shifts_coverage, demand_matrix)
        results = analyze_results(assignments, shifts_coverage, demand_matrix)
        if results:
            cov = results["coverage_percentage"]
            score = results["overstaffing"] + results["understaffing"]
            if verbose:
                st.info(f"Iteración {iteration + 1}: factor {factor}, cobertura {cov:.1f}%, score {score:.1f}")

            if cov >= target_coverage:
                if score < best_score or not best_assignments:
                    best_assignments, best_method = assignments, method
                    best_score = score
                    best_coverage = cov
                else:
                    break
            elif cov > best_coverage and not best_assignments:
                best_assignments, best_method, best_coverage = assignments, method, cov

        factor = max(1, int(factor * 0.9))
        iteration += 1

    agent_limit_factor = original_factor
    return best_assignments, best_method


def optimize_schedule_iterative(shifts_coverage, demand_matrix):
    """Función principal con estrategia FT primero + PT después"""
    if PULP_AVAILABLE:
        if optimization_profile == "JEAN":
            st.info("🔍 **Búsqueda JEAN**: cobertura sin exceso")
            return optimize_jean_search(shifts_coverage, demand_matrix, verbose=VERBOSE)

        if optimization_profile == "JEAN Personalizado":
            if use_ft and use_pt:
                st.info("🏢⏰ **Estrategia 2 Fases**: FT sin exceso → PT para completar")
                assignments, method = optimize_ft_then_pt_strategy(shifts_coverage, demand_matrix)

                results = analyze_results(assignments, shifts_coverage, demand_matrix)
                if results:
                    cov = results["coverage_percentage"]
                    score = results["overstaffing"] + results["understaffing"]
                    if cov < TARGET_COVERAGE or score > 0:
                        st.info("♻️ **Refinando con búsqueda JEAN**")
                        assignments, method = optimize_jean_search(shifts_coverage, demand_matrix, verbose=VERBOSE)

                return assignments, method
            else:
                st.info("🔍 **Búsqueda JEAN**: cobertura sin exceso")
                return optimize_jean_search(shifts_coverage, demand_matrix, verbose=VERBOSE)

        if use_ft and use_pt:
            st.info("🏢⏰ **Estrategia 2 Fases**: FT sin exceso → PT para completar")
            return optimize_ft_then_pt_strategy(shifts_coverage, demand_matrix)
        else:
            st.info("🎯 **Modo Precisión**: Optimización directa")
            return optimize_with_precision_targeting(shifts_coverage, demand_matrix)
    else:
        st.info("🔄 **Solver Básico**: Greedy mejorado")
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix)

def select_candidate_shifts(shifts_list, ft_shifts, pt_shifts, strategy_name, 
                          current_coverage, demand_matrix, iteration, max_agents):
    """Selección inteligente de turnos candidatos basada en la situación actual"""
    if not shifts_list:
        return []
    
    current_deficit = np.maximum(0, demand_matrix - current_coverage)
    total_deficit = np.sum(current_deficit)
    
    if strategy_name == 'greedy_best':
        return shifts_list
    
    elif strategy_name == 'balanced_mix':
        phase = iteration / max(1, max_agents)
        if phase < 0.3:
            pt_subset = pt_shifts[:max(1, len(pt_shifts)//2)] if pt_shifts else []
            return ft_shifts + pt_subset
        elif phase < 0.7:
            return shifts_list
        else:
            ft_subset = ft_shifts[:max(1, len(ft_shifts)//3)] if ft_shifts else []
            return pt_shifts + ft_subset
    
    elif strategy_name == 'deficit_priority':
        if total_deficit > np.sum(demand_matrix) * 0.3:
            return ft_shifts + pt_shifts
        else:
            ft_subset = ft_shifts[:max(1, len(ft_shifts)//2)] if ft_shifts else []
            return pt_shifts + ft_subset
    
    else:  # efficiency_focus
        deficit_cells = np.sum(current_deficit > 0)
        avg_deficit_per_cell = total_deficit / max(1, deficit_cells)
        
        if avg_deficit_per_cell > 2:
            pt_subset = pt_shifts[:max(1, len(pt_shifts)//3)] if pt_shifts else []
            return ft_shifts + pt_subset
        else:
            ft_subset = ft_shifts[:max(1, len(ft_shifts)//3)] if ft_shifts else []
            return pt_shifts + ft_subset

def adjust_score_by_shift_type(score, shift_name, current_coverage, demand_matrix, strategy_name):
    """Ajusta el score basado en la eficiencia del tipo de turno para la situación actual"""
    current_deficit = np.maximum(0, demand_matrix - current_coverage)
    total_deficit = np.sum(current_deficit)
    
    # Calcular eficiencia potencial del turno
    is_pt = shift_name.startswith('PT')
    is_ft = shift_name.startswith('FT')
    
    # Bonificar PT cuando hay muchos gaps pequeños
    small_gaps = np.sum((current_deficit > 0) & (current_deficit <= 2))
    large_gaps = np.sum(current_deficit > 2)
    
    if is_pt and small_gaps > large_gaps:
        score *= 1.2  # PT es mejor para gaps pequeños
    elif is_ft and large_gaps > small_gaps:
        score *= 1.2  # FT es mejor para gaps grandes
    
    return score

def generate_break_variants(shift_name, base_pattern, demand_matrix, current_coverage):
    """
    Genera variantes de break dinámicas basadas en la demanda actual
    """
    variants = [base_pattern]  # Incluir patrón original
    
    # Extraer información del turno
    parts = shift_name.split('_')
    start_hour = _extract_start_hour(shift_name)
    shift_duration = int(parts[0][2:])
    
    # Calcular ventana de break válida dinámicamente
    break_start = max(1.0, break_from_start)
    break_end = max(1.0, break_from_end)
    
    # Generar variantes inteligentes basadas en demanda
    for day in range(len(base_pattern)):
        if np.sum(base_pattern[day]) > 0:  # Si hay trabajo este día
            # Identificar horas con exceso actual para evitarlas
            day_excess = np.maximum(0, current_coverage[day] - demand_matrix[day])
            excess_hours = np.where(day_excess > 0)[0]
            
            # Generar opciones de break cada 30 minutos
            break_options = []
            current_time = start_hour + break_start
            while current_time <= start_hour + shift_duration - break_end:
                break_options.append(current_time)
                current_time += 0.5
            
            # Evaluar cada opción de break
            for break_time in break_options[:8]:  # Limitar opciones
                break_hour = int(break_time) % 24
                
                if break_hour < len(base_pattern[day]):
                    # Crear variante
                    variant = base_pattern.copy()
                    
                    # Reconstruir patrón con nuevo break
                    variant[day] = 0
                    for h in range(shift_duration):
                        hour_idx = int(start_hour + h) % 24
                        if hour_idx < len(variant[day]):
                            variant[day, hour_idx] = 1
                    
              
                    # Evitar duplicados
                    if not np.array_equal(variant, base_pattern):
                        variants.append(variant)
    
    return variants[:10]  # Limitar variantes

def calculate_comprehensive_score(current_coverage, new_coverage, demand_matrix, critical_days, peak_hours, strategy):
    """Calcula score integral considerando múltiples factores"""
    improvement = new_coverage - current_coverage
    
    # Score base por cobertura mejorada
    coverage_score = 0
    for day in range(len(demand_matrix)):
        for hour in range(len(demand_matrix[day])):
            demand = demand_matrix[day, hour]
            if demand > 0:
                old_coverage = min(current_coverage[day, hour], demand)
                new_coverage_val = min(new_coverage[day, hour], demand)
                coverage_score += (new_coverage_val - old_coverage) / demand
    
    # Bonificaciones por patrones críticos
    critical_bonus_score = 0
    if any(day in critical_days for day in range(len(improvement))):
        critical_bonus_score = np.sum(improvement[critical_days]) * critical_bonus
    
    peak_bonus_score = 0
    if len(peak_hours) > 0:
        peak_bonus_score = np.sum(improvement[:, peak_hours]) * peak_bonus
    
    # Penalización por exceso
    excess_penalty_score = 0
    for day in range(len(new_coverage)):
        for hour in range(len(new_coverage[day])):
            excess = max(0, new_coverage[day, hour] - demand_matrix[day, hour])
            excess_penalty_score -= excess * excess_penalty
    
    # Ajuste por estrategia
    strategy_multiplier = [1.0, 1.2, 0.8][strategy]
    
    total_score = (coverage_score + critical_bonus_score + peak_bonus_score + excess_penalty_score) * strategy_multiplier
    return total_score

def evaluate_solution_quality(coverage_matrix, demand_matrix):
    """Evalúa calidad general de la solución"""
    total_demand = demand_matrix.sum()
    total_coverage = np.minimum(coverage_matrix, demand_matrix).sum()
    coverage_percentage = (total_coverage / total_demand) * 100 if total_demand > 0 else 0
    
    excess = np.maximum(0, coverage_matrix - demand_matrix).sum()
    efficiency = total_coverage / (total_coverage + excess) if (total_coverage + excess) > 0 else 0
    
    # Score combinado (menor es mejor para minimización)
    quality_score = (100 - coverage_percentage) + (excess * 0.1) + ((1 - efficiency) * 50)
    return quality_score

def generate_weekly_pattern(start_hour, duration, working_days, dso_day=None, break_len=1):
    """Genera patrón semanal con breaks inteligentes"""
    pattern = np.zeros((7, 24), dtype=np.int8)

    for day in working_days:
        if day != dso_day:  # Excluir día de descanso
            for h in range(duration):
                t = start_hour + h
                d_off, idx = divmod(int(t), 24)
                pattern[(day + d_off) % 7, idx] = 1

            break_start_idx = start_hour + break_from_start
            break_end_idx = start_hour + duration - break_from_end

            if int(break_start_idx) < int(break_end_idx):
                break_hour = int(break_start_idx) + (
                    int(break_end_idx) - int(break_start_idx)
                ) // 2
            else:
                break_hour = int(break_start_idx)

            for b in range(int(break_len)):
                t = break_hour + b
                d_off, idx = divmod(int(t), 24)
                pattern[(day + d_off) % 7, idx] = 0

    return pattern.flatten()

def generate_shift_patterns(
    demand_matrix: np.ndarray | None = None,
    *,
    keep_percentage: float = 0.3,
    peak_bonus: float = 1.5,
    critical_bonus: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Genera patrones exhaustivos con múltiples franjas de break."""
    shifts_coverage = {}
    
    # Horas de inicio cada 30 minutos
    start_hours = np.arange(max(6, first_hour), min(last_hour - 2, 20), 0.5)
    
    # TURNOS FULL TIME con múltiples opciones de break
    if use_ft:
        if allow_8h:
            for start_hour in start_hours:
                for working_combo in combinations(ACTIVE_DAYS, min(6, len(ACTIVE_DAYS))):
                    non_working = [d for d in ACTIVE_DAYS if d not in working_combo]
                    for dso_day in non_working + [None]:
                        # Generar múltiples patrones con diferentes franjas de break
                        break_options = get_valid_break_times(start_hour, 8)
                        for break_idx, break_start in enumerate(break_options):
                            pattern = generate_weekly_pattern_with_break(start_hour, 8, list(working_combo), dso_day, break_start)
                            dso_suffix = f"_DSO{dso_day}" if dso_day is not None else ""
                            shifts_coverage[f"FT8_{start_hour:04.1f}_DAYS{''.join(map(str,working_combo))}_BRK{break_start:04.1f}{dso_suffix}"] = pattern
        

    
    # TURNOS PART TIME (sin break por ser cortos)
    if use_pt:
        if allow_pt_4h:
            for start_hour in start_hours:
                for num_days in [4, 5, 6]:
                    if num_days <= len(ACTIVE_DAYS):
                        for working_combo in combinations(ACTIVE_DAYS, num_days):
                            pattern = generate_weekly_pattern_simple(start_hour, 4, list(working_combo))
                            shifts_coverage[f"PT4_{start_hour:04.1f}_DAYS{''.join(map(str,working_combo))}"] = pattern
        
        if allow_pt_6h:
            for start_hour in start_hours[::2]:
                for num_days in [4]:
                    if num_days <= len(ACTIVE_DAYS):
                        for working_combo in combinations(ACTIVE_DAYS, num_days):
                            pattern = generate_weekly_pattern_simple(start_hour, 6, list(working_combo))
                            shifts_coverage[f"PT6_{start_hour:04.1f}_DAYS{''.join(map(str,working_combo))}"] = pattern
        
        if allow_pt_5h:
            for start_hour in start_hours[::2]:
                for num_days in [5]:
                    if num_days <= len(ACTIVE_DAYS):
                        for working_combo in combinations(ACTIVE_DAYS, num_days):
                            pattern = generate_weekly_pattern_simple(start_hour, 5, list(working_combo))
                            shifts_coverage[f"PT5_{start_hour:04.1f}_DAYS{''.join(map(str,working_combo))}"] = pattern
    
    return shifts_coverage

def get_valid_break_times(start_hour, duration):
    """Obtiene todas las franjas válidas de break para un turno"""
    valid_breaks = []
    
    # Calcular ventana válida para el break
    earliest_break = start_hour + break_from_start  # 2 horas después del inicio
    latest_break = start_hour + duration - break_from_end - 1  # 2 horas antes del fin, -1 por duración del break
    
    # Generar opciones cada 30 minutos
    current_time = earliest_break
    while current_time <= latest_break:
        # Solo permitir breaks en horas exactas o medias horas
        if current_time % 0.5 == 0:  # Múltiplo de 0.5
            valid_breaks.append(current_time)
        current_time += 0.5
    
    return valid_breaks[:7]  # Máximo 7 opciones para no saturar

def generate_weekly_pattern_with_break(start_hour, duration, working_days, dso_day, break_start, break_len=1):
    """Genera patrón semanal con break específico - CORREGIDO para turnos que cruzan medianoche"""
    pattern = np.zeros((7, 24), dtype=np.int8)

    for day in working_days:
        if day == dso_day:
            continue

        for h in range(duration):
            t = start_hour + h
            d_off, idx = divmod(int(t), 24)
            pattern[(day + d_off) % 7, idx] = 1

        for b in range(int(break_len)):
            t = break_start + b
            d_off, idx = divmod(int(t), 24)
            pattern[(day + d_off) % 7, idx] = 0

    return pattern.flatten()

def generate_weekly_pattern_simple(start_hour, duration, working_days):
    """Genera patrón semanal simple sin break (para PT)"""
    pattern = np.zeros((7, 24), dtype=np.int8)

    for day in working_days:
        for h in range(duration):
            t = start_hour + h
            d_off, idx = divmod(int(t), 24)
            pattern[(day + d_off) % 7, idx] = 1

    return pattern.flatten()

def generate_weekly_pattern_pt5(start_hour, working_days):
    """Genera patrón de 24h para PT5 (5h en cuatro días y 4h en uno)"""
    pattern = np.zeros((7, 24), dtype=np.int8)

    if not working_days:
        return pattern.flatten()

    four_hour_day = working_days[-1]
    for day in working_days:
        hours = 4 if day == four_hour_day else 5
        for h in range(hours):
            t = start_hour + h
            d_off, idx = divmod(int(t), 24)
            pattern[(day + d_off) % 7, idx] = 1

    return pattern.flatten()

def generate_weekly_pattern_10h8(start_hour, working_days, eight_hour_day, break_len=1):
    """Genera patrón con cuatro días de 10h y uno de 8h"""
    pattern = np.zeros((7, 24), dtype=np.int8)

    for day in working_days:
        duration = 8 if day == eight_hour_day else 10
        for h in range(duration):
            t = start_hour + h
            d_off, idx = divmod(int(t), 24)
            pattern[(day + d_off) % 7, idx] = 1

        break_start_idx = start_hour + break_from_start
        break_end_idx = start_hour + duration - break_from_end
        if int(break_start_idx) < int(break_end_idx):
            break_hour = int(break_start_idx) + (
                int(break_end_idx) - int(break_start_idx)
            ) // 2
        else:
            break_hour = int(break_start_idx)
        for b in range(int(break_len)):
            t = break_hour + b
            d_off, idx = divmod(int(t), 24)
            pattern[(day + d_off) % 7, idx] = 0

    return pattern.flatten()




def generate_weekly_pattern_advanced(start_hour, duration, working_days, break_position):
    """Genera patrón semanal avanzado con break posicionado dinámicamente"""
    pattern = np.zeros((7, 24), dtype=np.int8)
    
    for day in working_days:
        # Marcar horas de trabajo
        for h in range(duration):
            hour_idx = int(start_hour + h) % 24
            if hour_idx < 24:
                pattern[day, hour_idx] = 1
        
        # Calcular posición del break dinámicamente
        break_hour_offset = int(duration * break_position)
        break_hour = int(start_hour + break_hour_offset) % 24
        
        # Aplicar break respetando restricciones
        if (break_hour >= int(start_hour + break_from_start) and 
            break_hour <= int(start_hour + duration - break_from_end) and
            break_hour < 24):
            pattern[day, break_hour] = 0
    
    return pattern.flatten()

def analyze_results(assignments, shifts_coverage, demand_matrix):
    """Analiza los resultados de la optimización.

    Calcula métricas como :data:`coverage_percentage`, que ahora
    representa el porcentaje de demanda cubierta (ponderado por horas).
    """
    if not assignments:
        return None
    
    # Calcular cobertura total
    slots_per_day = len(next(iter(shifts_coverage.values()))) // 7 if shifts_coverage else 24
    total_coverage = np.zeros((7, slots_per_day), dtype=np.int16)
    total_agents = 0
    ft_agents = 0
    pt_agents = 0
    
    for shift_name, count in assignments.items():
        weekly_pattern = shifts_coverage[shift_name]
        slots_per_day = len(weekly_pattern) // 7
        pattern_matrix = np.array(weekly_pattern).reshape(7, slots_per_day)

        weekly_hours = pattern_matrix.sum()
        max_allowed = 48 if shift_name.startswith('FT') else 24
        if weekly_hours > max_allowed:
            st.warning(f"⚠️ {shift_name} excede el máximo de {max_allowed}h (tiene {weekly_hours}h)")
        total_coverage += pattern_matrix * count
        total_agents += count
        
        if shift_name.startswith('FT'):
            ft_agents += count
        else:
            pt_agents += count
    
    # Calcular métricas
    total_demand = demand_matrix.sum()
    total_covered = np.minimum(total_coverage, demand_matrix).sum()
    coverage_percentage = (total_covered / total_demand) * 100 if total_demand > 0 else 0
    
    # Calcular over/under staffing
    diff_matrix = total_coverage - demand_matrix
    overstaffing = np.sum(diff_matrix[diff_matrix > 0])
    understaffing = np.sum(np.abs(diff_matrix[diff_matrix < 0]))
    
    return {
        'total_coverage': total_coverage,
        'total_agents': total_agents,
        'ft_agents': ft_agents,
        'pt_agents': pt_agents,
        'coverage_percentage': coverage_percentage,
        'overstaffing': overstaffing,
        'understaffing': understaffing,
        'diff_matrix': diff_matrix
    }

def create_heatmap(matrix, title, cmap='RdYlBu_r'):
    """Crea un heatmap de la matriz"""
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_yticks(range(7))
    ax.set_yticklabels(['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo'])
    
    for i in range(7):
        for j in range(24):
            text = ax.text(j, i, f'{matrix[i, j]:.0f}',
                         ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title(title)
    ax.set_xlabel('Hora del día')
    ax.set_ylabel('Día de la semana')
    plt.colorbar(im, ax=ax)
    return fig



# ——————————————————————————————————————————————————————————————
# Exportación detallada de horarios
# ——————————————————————————————————————————————————————————————

def _extract_start_hour(name: str) -> float:
    """Return the first decimal start hour found in ``name``.

    The shift naming convention encodes the start hour separated by
    underscores.  Older patterns only looked for the first numeric
    segment after an underscore which failed when additional metadata
    preceded the hour.  The updated logic scans all underscore
    separated tokens and returns the first value containing a decimal
    point, e.g. ``08.0``.
    """
    for part in name.split('_'):
        if '.' in part and part.replace('.', '').isdigit():
            try:
                return float(part)
            except ValueError:
                continue
    return 0.0

def export_detailed_schedule(assignments, shifts_coverage):
    """Exporta horarios semanales detallados - ROBUSTO"""
    if not assignments:
        return None

    detailed_data = []
    agent_id = 1

    for shift_name, count in assignments.items():
        weekly_pattern = shifts_coverage[shift_name]
        slots_per_day = len(weekly_pattern) // 7
        pattern_matrix = np.array(weekly_pattern).reshape(7, slots_per_day)

        # Parsing robusto del nombre del turno
        parts = shift_name.split('_')
        start_hour = _extract_start_hour(shift_name)

        # Determinar tipo y duración del turno
        if shift_name.startswith('FT10p8'):
            shift_type = 'FT'
            shift_duration = 10
            total_hours = shift_duration + 1
        elif shift_name.startswith('FT'):
            shift_type = 'FT'
            if '_Variado_' in shift_name or len(parts) > 4:
                shift_duration = int(pattern_matrix.sum(axis=1).max())
            else:
                try:
                    if len(parts[0]) > 2 and parts[0][2:].isdigit():
                        shift_duration = int(parts[0][2:])
                    else:
                        shift_duration = int(pattern_matrix.sum(axis=1).max())
                except (ValueError, IndexError):
                    shift_duration = int(pattern_matrix.sum(axis=1).max())
            total_hours = shift_duration + 1
        elif shift_name.startswith('PT'):
            shift_type = 'PT'
            if '_Variado_' in shift_name or len(parts) > 4:
                shift_duration = int(pattern_matrix.sum(axis=1).max())
            else:
                try:
                    if len(parts[0]) > 2 and parts[0][2:].isdigit():
                        shift_duration = int(parts[0][2:])
                    else:
                        shift_duration = int(pattern_matrix.sum(axis=1).max())
                except (ValueError, IndexError):
                    shift_duration = int(pattern_matrix.sum(axis=1).max())
            total_hours = shift_duration
        else:
            shift_type = 'FT'
            shift_duration = 8
            total_hours = 9

        for agent_num in range(count):
            for day in range(7):
                day_pattern = pattern_matrix[day]
                work_hours = np.where(day_pattern == 1)[0]

                if len(work_hours) > 0:
                    start_idx = int(start_hour)
                    # Calcular horario específico para cada tipo
                    if shift_name.startswith('PT'):
                        # Para PT usar las horas reales trabajadas según el patrón
                        end_idx = (int(work_hours[-1]) + 1) % 24
                        next_day = end_idx <= start_idx
                        horario = f"{start_idx:02d}:00-{end_idx:02d}:00" + ("+1" if next_day else "")
                    elif shift_name.startswith('FT10p8'):
                        end_idx = (int(work_hours[-1]) + 1) % 24
                        next_day = end_idx <= start_idx
                        horario = f"{start_idx:02d}:00-{end_idx:02d}:00" + ("+1" if next_day else "")
                    else:
                        # Otros turnos normales
                        end_idx = int(work_hours[-1]) + 1
                        next_day = end_idx <= start_idx
                        horario = (
                            f"{start_idx:02d}:00-{end_idx % 24:02d}:00" + ("+1" if next_day else "")
                        )

                    # Calcular break específico
                    if shift_name.startswith('PT'):
                        break_time = ""
                    elif shift_name.startswith('FT10p8'):
                        all_expected = set(range(int(start_hour), int(start_hour + total_hours)))
                        actual_hours = set(work_hours)
                        break_hours = all_expected - actual_hours

                        if break_hours:
                            break_hour = min(break_hours) % 24
                            break_end = (break_hour + 1) % 24
                            if break_end == 0:
                                break_end = 24
                            break_time = f"{break_hour:02d}:00-{break_end:02d}:00"
                        else:
                            break_time = ""
                    else:
                        # Otros turnos con break de 1 hora
                        expected = list(range(start_idx, end_idx))
                        if next_day:
                            expected = list(range(start_idx, 24)) + list(range(0, end_idx % 24))
                        break_hours = set(expected) - set(work_hours)

                        if break_hours:
                            break_hour = min(break_hours)
                            break_time = f"{break_hour % 24:02d}:00-{((break_hour + 1) % 24) or 24:02d}:00"
                        else:
                            break_time = ""

                    detailed_data.append({
                        'Agente': f"AGT_{agent_id:03d}",
                        'Dia': ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo'][day],
                        'Horario': horario,
                        'Break': break_time,
                        'Turno': shift_name,
                        'Tipo': shift_type
                    })
                else:
                    detailed_data.append({
                        'Agente': f"AGT_{agent_id:03d}",
                        'Dia': ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo'][day],
                        'Horario': "DSO",
                        'Break': "",
                        'Turno': shift_name,
                        'Tipo': 'DSO'
                    })
            agent_id += 1

    df_detailed = pd.DataFrame(detailed_data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_detailed.to_excel(writer, sheet_name='Horarios_Semanales', index=False)

        df_summary = df_detailed.groupby(['Agente', 'Turno']).size().reset_index(name='Dias_Trabajo')
        df_summary.to_excel(writer, sheet_name='Resumen_Agentes', index=False)

        df_shifts = pd.DataFrame([
            {'Turno': shift, 'Agentes': count}
            for shift, count in assignments.items()
        ])
        df_shifts.to_excel(writer, sheet_name='Turnos_Asignados', index=False)

    return output.getvalue()
def solve_in_chunks_optimized(
    shifts_coverage: Dict[str, np.ndarray],
    demand_matrix: np.ndarray,
    base_chunk_size: int = 10000,
) -> Dict[str, int]:
    """Solve using adaptive chunks sorted by score."""
    scored = []
    seen: set[bytes] = set()
    for name, pat in shifts_coverage.items():
        key = hashlib.md5(pat).digest()
        if key in seen:
            continue
        seen.add(key)
        scored.append((name, pat, score_pattern(pat, demand_matrix)))

    scored.sort(key=lambda x: x[2], reverse=True)

    assignments_total: Dict[str, int] = {}
    coverage = np.zeros_like(demand_matrix)
    idx = 0
    while idx < len(scored):
        chunk_size = adaptive_chunk_size(base_chunk_size)
        chunk_dict = {name: pat for name, pat, _ in scored[idx: idx + chunk_size]}
        remaining = np.maximum(0, demand_matrix - coverage)
        if not np.any(remaining):
            break
        assigns, _ = optimize_schedule_iterative(chunk_dict, remaining)
        for name, val in assigns.items():
            assignments_total[name] = assignments_total.get(name, 0) + val
            slots = len(chunk_dict[name]) // 7
            coverage += chunk_dict[name].reshape(7, slots) * val
        idx += chunk_size
        gc.collect()
        emergency_cleanup()
        if not np.any(np.maximum(0, demand_matrix - coverage)):
            break
    return assignments_total



# ——————————————————————————————————————————————————————————————
# Botón de ejecución con aprendizaje integrado
# ——————————————————————————————————————————————————————————————

if st.button("🚀 Ejecutar Optimización", type="primary", use_container_width=True):
    start_time = time.time()
    
    # Generar patrones de turnos
    st.info("🔄 Generando patrones de turnos...")
    shifts_coverage = {}
    for batch in generate_shifts_coverage_optimized(demand_matrix):
        shifts_coverage.update(batch)
    
    if not shifts_coverage:
        st.error("⚠️ No se pudieron generar patrones válidos con la configuración actual")
        st.stop()
    
    # Ejecutar optimización
    st.info(f"🎯 Optimizando con {len(shifts_coverage)} patrones...")
    if PULP_AVAILABLE:
        st.success("🧠 **Solver Inteligente Activado** - Programación Lineal")
    assignments = solve_in_chunks_optimized(shifts_coverage, demand_matrix)
    method = "CHUNKED_OPT"
    
    if not assignments:
        st.error("⚠️ No se pudo encontrar una solución válida")
        st.stop()
    
    # Calcular métricas finales
    total_coverage = np.zeros_like(demand_matrix)
    total_agents = sum(assignments.values())
    
    for shift_name, count in assignments.items():
        if shift_name in shifts_coverage:
            slots_per_day = len(shifts_coverage[shift_name]) // 7
            shift_pattern = np.array(shifts_coverage[shift_name]).reshape(7, slots_per_day)
            total_coverage += shift_pattern * count
    
    # Calcular cobertura real (puede ser >100% si hay exceso)
    total_demand = demand_matrix.sum()
    if total_demand > 0:
        # Cobertura real: total cubierto / total demandado
        final_coverage = (total_coverage.sum() / total_demand) * 100
    else:
        final_coverage = 0
    
    execution_time = time.time() - start_time
    
    # Guardar resultado para aprendizaje
    if use_learning:
        current_params = {
            "agent_limit_factor": agent_limit_factor,
            "excess_penalty": excess_penalty,
            "peak_bonus": peak_bonus,
            "critical_bonus": critical_bonus
        }
        
        learning_success = save_execution_result(
            demand_matrix, current_params, final_coverage, 
            total_agents, execution_time
        )
        
        if learning_success:
            # Mostrar progreso evolutivo
            updated_stats = load_learning_data().get("stats", {})
            recent_improvement = updated_stats.get("recent_improvement", 0)
            if recent_improvement > 0:
                st.success(f"📈 **¡Evolución exitosa!** Mejora de {recent_improvement:+.1f}% vs ejecución anterior")
            elif recent_improvement == 0:
                st.info("📊 **Resultado estable** - El sistema explorará nuevas variaciones")
            else:
                st.warning(f"📉 **Retroceso detectado** ({recent_improvement:+.1f}%) - El sistema se adaptará más agresivamente")
            
            # Actualizar session state
            st.session_state.learning_data = load_learning_data()
        else:
            st.warning("⚠️ No se pudo guardar el resultado para evolución futura")
    
    # Mostrar resultados
    st.success(f"✅ **Optimización completada en {execution_time:.1f}s**")
    
    # Análisis detallado de cobertura
    coverage_analysis = None
    try:
        coverage_analysis = analyze_coverage_precision(assignments, shifts_coverage, demand_matrix)
    except NameError:
        # Función no definida, usar análisis básico
        pass
    
    # Mostrar análisis de precisión
    if coverage_analysis and 'exact_coverage' in coverage_analysis:
        st.info(f"🎯 **Análisis de Precisión:**")
        st.info(f"- Cobertura exacta: {coverage_analysis['exact_coverage']:.1f}%")
        st.info(f"- Horas con déficit: {coverage_analysis['deficit_hours']}")
        st.info(f"- Horas con exceso: {coverage_analysis['excess_hours']}")
        st.info(f"- Eficiencia: {coverage_analysis['efficiency']:.1f}%")
        
        if coverage_analysis['deficit_hours'] > 0:
            st.warning(f"⚠️ **Déficit detectado en {coverage_analysis['deficit_hours']} horas**")
            st.info("💡 **Sugerencia**: Prueba con 'Máxima Cobertura' o '100% Cobertura Total'")
            
            # Mostrar problemas específicos
            if coverage_analysis['problem_areas']:
                st.subheader("🔍 Problemas Detectados")
                problem_df = pd.DataFrame(coverage_analysis['problem_areas'])
                st.dataframe(problem_df, use_container_width=True)
    
    # Analizar resultados completos
    results = analyze_results(assignments, shifts_coverage, demand_matrix)
    
    if results:
        # Métricas principales mejoradas
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("👥 Total Agentes", results['total_agents'])
        with col2:
            # Cobertura real (puede ser >100%)
            coverage_color = "normal" if final_coverage <= 105 else "inverse"
            st.metric("📊 Cobertura Real", f"{final_coverage:.1f}%", delta=f"{final_coverage-100:.1f}%" if final_coverage > 100 else None)
        with col3:
            # Eficiencia (cobertura sin exceso)
            pure_coverage = min(100, (np.minimum(total_coverage, demand_matrix).sum() / demand_matrix.sum()) * 100)
            st.metric("✅ Cobertura Pura", f"{pure_coverage:.1f}%")
        with col4:
            st.metric("⬆️ Exceso", f"{results['overstaffing']:.0f}")
        with col5:
            st.metric("⬇️ Déficit", f"{results['understaffing']:.0f}")
        
        # Distribución de contratos
        if results['ft_agents'] > 0 or results['pt_agents'] > 0:
            st.subheader("📋 Distribución de Contratos")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Full Time", results['ft_agents'])
            with col2:
                st.metric("Part Time", results['pt_agents'])
        
        # Visualizaciones
        st.subheader("📈 Análisis Visual")
        tab1, tab2, tab3, tab4 = st.tabs(["Demanda vs Cobertura", "Diferencias", "Turnos Asignados", "Tabla Comparativa"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Demanda Requerida")
                fig_demand = create_heatmap(demand_matrix, "Demanda por Hora y Día", 'Reds')
                st.pyplot(fig_demand)
            with col2:
                st.subheader("Cobertura Asignada")
                fig_coverage = create_heatmap(results['total_coverage'], "Cobertura por Hora y Día", 'Blues')
                st.pyplot(fig_coverage)
        
        with tab2:
            st.subheader("Diferencias (Cobertura - Demanda)")
            fig_diff = create_heatmap(results['diff_matrix'], "Diferencias por Hora y Día", 'RdBu')
            st.pyplot(fig_diff)
        
        with tab3:
            st.subheader("Turnos Asignados")
            turnos_data = []
            for shift_name, count in assignments.items():
                turnos_data.append({'Turno': shift_name, 'Agentes': count})
            df_turnos = pd.DataFrame(turnos_data)
            st.dataframe(df_turnos, use_container_width=True)
        
        with tab4:
            st.subheader("Tabla Comparativa Detallada")
            # Crear tabla comparativa hora por hora
            comparison_data = []
            dias_semana = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
            
            for day in range(7):
                for hour in range(24):
                    if demand_matrix[day, hour] > 0 or results['total_coverage'][day, hour] > 0:
                        comparison_data.append({
                            'Día': dias_semana[day],
                            'Hora': f"{hour:02d}:00",
                            'Demanda': int(demand_matrix[day, hour]),
                            'Cobertura': int(results['total_coverage'][day, hour]),
                            'Diferencia': int(results['total_coverage'][day, hour] - demand_matrix[day, hour]),
                            'Estado': 'OK' if results['total_coverage'][day, hour] == demand_matrix[day, hour] else 
                                     ('Déficit' if results['total_coverage'][day, hour] < demand_matrix[day, hour] else 'Exceso')
                        })
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                # Filtrar solo problemas si hay muchas filas
                if len(df_comparison) > 50:
                    df_problems = df_comparison[df_comparison['Estado'] != 'OK']
                    if len(df_problems) > 0:
                        st.write("**Solo mostrando horas con problemas:**")
                        st.dataframe(df_problems, use_container_width=True)
                    else:
                        st.success("✅ **¡Cobertura perfecta!** No hay problemas detectados")
                else:
                    st.dataframe(df_comparison, use_container_width=True)
        
        # Exportación
        st.subheader("📥 Exportar Resultados")
        excel_data = export_detailed_schedule(assignments, shifts_coverage)
        if excel_data:
            st.download_button(
                label="📊 Descargar Horarios Detallados",
                data=excel_data,
                file_name="horarios_semanales_detallados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.error("❌ Error al analizar los resultados")


def calculate_comprehensive_score(current_coverage, new_coverage, demand_matrix, critical_days, peak_hours, strategy):
    """Scoring que balancea déficit vs exceso y promueve eficiencia"""
    # Déficit base
    current_deficit = np.maximum(0, demand_matrix - current_coverage)
    new_deficit = np.maximum(0, demand_matrix - new_coverage)
    deficit_reduction = np.sum(current_deficit - new_deficit)
    
    # Penalización de exceso progresiva
    current_excess = np.maximum(0, current_coverage - demand_matrix)
    new_excess = np.maximum(0, new_coverage - demand_matrix)
    excess_increase = np.sum(new_excess - current_excess)
    
    # Penalización progresiva: más exceso = mayor penalización
    total_current_excess = np.sum(current_excess)
    if total_current_excess > 100:  # Si ya hay mucho exceso
        excess_penalty_value = excess_increase * excess_penalty * 3  # Triple penalización
    elif total_current_excess > 50:
        excess_penalty_value = excess_increase * excess_penalty * 2  # Doble penalización
    else:
        excess_penalty_value = excess_increase * excess_penalty
    
    # Bonificación por eficiencia (más déficit cubierto por hora trabajada)
    pattern_diff = new_coverage - current_coverage
    total_hours_added = np.sum(pattern_diff)
    efficiency_bonus = 0
    if total_hours_added > 0:
        efficiency_ratio = deficit_reduction / total_hours_added
        efficiency_bonus = efficiency_ratio * 20  # Bonificar eficiencia
    
    # Bonificaciones para patrones críticos
    critical_bonus_value = 0
    for critical_day in critical_days:
        if critical_day < len(new_coverage):
            day_improvement = np.sum(np.maximum(0, current_deficit[critical_day] - new_deficit[critical_day]))
            critical_bonus_value += day_improvement * critical_bonus
    
    peak_bonus_value = 0
    for day in range(len(new_coverage)):
        for hour in peak_hours:
            if hour < len(new_coverage[day]):
                hour_improvement = max(0, current_deficit[day, hour] - new_deficit[day, hour])
                peak_bonus_value += hour_improvement * peak_bonus
    
    return deficit_reduction + efficiency_bonus + critical_bonus_value + peak_bonus_value - excess_penalty_value

def evaluate_solution_quality(coverage, demand_matrix):
    """
    Evalúa la calidad general de una solución
    """
    deficit = np.sum(np.maximum(0, demand_matrix - coverage))
    excess = np.sum(np.maximum(0, coverage - demand_matrix))
    return deficit + excess * 0.5  # Penalizar exceso menos que déficit

def optimize_schedule(shifts_coverage, demand_matrix):
    """
    Usa optimización iterativa avanzada
    """
    return optimize_schedule_iterative(shifts_coverage, demand_matrix)



# ——————————————————————————————————————————————————————————————
# 6. Análisis de resultados
# ——————————————————————————————————————————————————————————————

def analyze_coverage_precision(assignments, shifts_coverage, demand_matrix):
    """Analiza la precisión de cobertura con métricas detalladas"""
    if not assignments:
        return None
    
    # Calcular cobertura total
    slots_per_day = len(next(iter(shifts_coverage.values()))) // 7 if shifts_coverage else 24
    total_coverage = np.zeros((7, slots_per_day), dtype=np.int16)
    for shift_name, count in assignments.items():
        weekly_pattern = shifts_coverage[shift_name]
        slots_per_day = len(weekly_pattern) // 7
        pattern_matrix = np.array(weekly_pattern).reshape(7, slots_per_day)
        total_coverage += pattern_matrix * count
    
    # Métricas de precisión
    total_demand = demand_matrix.sum()
    exact_coverage_sum = np.minimum(total_coverage, demand_matrix).sum()
    exact_coverage_pct = (exact_coverage_sum / total_demand * 100) if total_demand > 0 else 0
    
    # Contar horas con problemas
    deficit_mask = total_coverage < demand_matrix
    excess_mask = total_coverage > demand_matrix
    deficit_hours = np.sum(deficit_mask & (demand_matrix > 0))
    excess_hours = np.sum(excess_mask)
    
    # Calcular eficiencia (cobertura útil vs total)
    total_coverage_sum = total_coverage.sum()
    efficiency = (exact_coverage_sum / total_coverage_sum * 100) if total_coverage_sum > 0 else 0
    
    # Identificar patrones problemáticos
    problem_areas = []
    dias_semana = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
    
    for day in range(7):
        for hour in range(24):
            if demand_matrix[day, hour] > 0:
                if total_coverage[day, hour] < demand_matrix[day, hour]:
                    deficit = demand_matrix[day, hour] - total_coverage[day, hour]
                    problem_areas.append({
                        'day': dias_semana[day],
                        'hour': f"{hour:02d}:00",
                        'type': 'Déficit',
                        'amount': deficit,
                        'demand': demand_matrix[day, hour],
                        'coverage': total_coverage[day, hour]
                    })
                elif total_coverage[day, hour] > demand_matrix[day, hour]:
                    excess = total_coverage[day, hour] - demand_matrix[day, hour]
                    problem_areas.append({
                        'day': dias_semana[day],
                        'hour': f"{hour:02d}:00",
                        'type': 'Exceso',
                        'amount': excess,
                        'demand': demand_matrix[day, hour],
                        'coverage': total_coverage[day, hour]
                    })
    
    return {
        'exact_coverage': exact_coverage_pct,
        'deficit_hours': deficit_hours,
        'excess_hours': excess_hours,
        'efficiency': efficiency,
        'problem_areas': problem_areas[:10],  # Top 10 problemas
        'total_coverage': total_coverage
    }




# ——————————————————————————————————————————————————————————————
# 9. Interfaz principal
# ——————————————————————————————————————————————————————————————





# ——————————————————————————————————————————————————————————————
# 10. Footer
# ——————————————————————————————————————————————————————————————
# ——————————————————————————————————————————————————————————————
# 10. Footer
# ——————————————————————————————————————————————————————————————
st.markdown("---")
st.markdown("**Generador de Turnos v6.2** - Sistema de optimización con aprendizaje adaptativo")

