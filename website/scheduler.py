# -*- coding: utf-8 -*-
from flask import current_app
import json
import time
import os
import gc
import hashlib
import io
from io import BytesIO, StringIO
from itertools import combinations, permutations, product
import heapq

import tempfile
import csv

import numpy as np
import pandas as pd
import math
from collections import defaultdict
import re
import base64

try:
    import matplotlib
    matplotlib.use("Agg")  # Use a non-GUI backend for server-side generation
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover - optional dependency
    matplotlib = None
    plt = None
    sns = None
try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

from typing import Dict, List, Iterable, Union

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

print(f"[OPTIMIZER] PuLP disponible: {PULP_AVAILABLE}")

from threading import RLock, current_thread
from functools import wraps

_MODEL_LOCK = RLock()

# Bridge helpers to the shared store in website.extensions
try:
    from .extensions import scheduler as _ext_store
except Exception:
    _ext_store = None

try:
    from .parallel_optimizer import save_partial_results
    from .optimizer_greedy import optimize_with_greedy
except Exception:  # pragma: no cover - allow direct execution
    def save_partial_results(*args, **kwargs):
        pass

    def optimize_with_greedy(*args, **kwargs):
        return {}, "NOT_EXECUTED"

def _store(app=None):
    """Return backing store dict from extensions (jobs/results)."""
    if _ext_store is None:
        return {"jobs": {}, "results": {}}
    try:
        return _ext_store._s(app)
    except Exception:
        # Last resort, create a minimal structure
        return {"jobs": {}, "results": {}, "active_jobs": {}}

def mark_running(job_id, app=None):
    if _ext_store is not None:
        try:
            _ext_store.mark_running(job_id, app=app)
            return
        except Exception:
            pass
    _store(app).setdefault("jobs", {})[job_id] = {"status": "running"}

def update_progress(job_id, info, app=None):
    if _ext_store is not None:
        try:
            _ext_store.update_progress(job_id, info, app=app)
            return
        except Exception:
            pass
    s = _store(app)
    job = s.setdefault("jobs", {}).setdefault(job_id, {"status": "running"})
    prog = job.get("progress", {})
    if not isinstance(prog, dict):
        prog = {}
    if not isinstance(info, dict):
        info = {"msg": str(info)}
    prog.update(info)
    job["progress"] = prog
    s["jobs"][job_id] = job

def mark_finished(job_id, result, excel_path, csv_path, app=None):
    if _ext_store is not None:
        try:
            _ext_store.mark_finished(job_id, result, excel_path, csv_path, app=app)
            return
        except Exception:
            pass
    s = _store(app)
    s.setdefault("jobs", {})[job_id] = {"status": "finished"}
    s.setdefault("results", {})[job_id] = {
        "result": result,
        "excel_path": excel_path,
        "csv_path": csv_path,
        "timestamp": time.time(),
    }

def mark_error(job_id, msg, app=None):
    if _ext_store is not None:
        try:
            _ext_store.mark_error(job_id, msg, app=app)
            return
        except Exception:
            pass
    _store(app).setdefault("jobs", {})[job_id] = {"status": "error", "error": msg}

def mark_cancelled(job_id, app=None):
    if _ext_store is not None:
        try:
            _ext_store.mark_cancelled(job_id, app=app)
            return
        except Exception:
            pass
    _store(app).setdefault("jobs", {})[job_id] = {"status": "cancelled"}

def get_status(job_id, app=None):
    if _ext_store is not None:
        try:
            return _ext_store.get_status(job_id, app=app)
        except Exception:
            pass
    return _store(app).setdefault("jobs", {}).get(job_id, {"status": "unknown"})

def get_payload(job_id, app=None):
    if _ext_store is not None:
        try:
            return _ext_store.get_payload(job_id, app=app)
        except Exception:
            pass
    return _store(app).setdefault("results", {}).get(job_id)

# Store uploaded JSON configuration for custom shifts
template_cfg = {}


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
    print(f"Patrones {count} | {rate:.1f}/s | Mem {mem:.1f}%")


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

# ——————————————————————————————————————————————————————————————
# Sistema de Aprendizaje Adaptativo
# ——————————————————————————————————————————————————————————————

def load_learning_data():
    """Carga datos de aprendizaje con caché"""
    try:
        if os.path.exists("optimization_learning.json"):
            with open("optimization_learning.json", 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error al cargar el aprendizaje: {e}")
    return {"executions": [], "best_params": {}, "stats": {}}

def save_learning_data(data):
    """Guarda datos de aprendizaje sin cache"""
    try:
        with open("optimization_learning.json", 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"No se pudo guardar el aprendizaje: {e}")

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
    """Guarda resultado con análisis de mejora evolutiva EXACTO del legacy"""
    learning_data = load_learning_data()
    input_hash = hashlib.md5(str(demand_matrix).encode()).hexdigest()[:12]
    
    # Calcular métricas de calidad mejoradas EXACTAS del legacy
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
    
    # Mantener solo últimas 50 ejecuciones por patrón EXACTO del legacy
    pattern_executions = [e for e in learning_data["executions"] if e.get("input_hash") == input_hash]
    if len(pattern_executions) > 50:
        # Remover las más antiguas de este patrón
        learning_data["executions"] = [e for e in learning_data["executions"] 
                                     if e.get("input_hash") != input_hash or 
                                     e.get("timestamp", 0) >= sorted([p.get("timestamp", 0) for p in pattern_executions])[-50]]
    
    # Actualizar mejores parámetros con múltiples criterios EXACTO del legacy
    current_best = learning_data["best_params"].get(input_hash, {})
    
    # Score combinado EXACTO del legacy: priorizar cobertura, luego eficiencia
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
    
    # Estadísticas evolutivas EXACTAS del legacy
    pattern_runs = [e for e in learning_data["executions"] if e.get("input_hash") == input_hash]
    if len(pattern_runs) >= 2:
        recent_improvement = pattern_runs[-1]["coverage"] - pattern_runs[-2]["coverage"]
    else:
        recent_improvement = 0
    
    learning_data["stats"] = {
        "total_executions": len(learning_data["executions"]),
        "unique_patterns": len(set(e["input_hash"] for e in learning_data["executions"])),
        "avg_coverage": np.mean([e["coverage"] for e in learning_data["executions"][-10:]]) if learning_data["executions"] else 0,
        "recent_improvement": recent_improvement,
        "best_coverage": max([e["coverage"] for e in learning_data["executions"]], default=0),
        "last_updated": time.time()
    }
    
    save_learning_data(learning_data)
    return True

def create_demand_signature(demand_matrix):
    """Crea una firma única para el patrón de demanda EXACTO del legacy"""
    # Normalizar y crear hash del patrón de demanda
    normalized = demand_matrix / (demand_matrix.max() + 1e-8)
    signature = hashlib.md5(normalized.tobytes()).hexdigest()[:16]
    return signature

def load_learning_history():
    """Carga el historial de aprendizaje EXACTO del legacy"""
    try:
        if os.path.exists('learning_history.json'):
            with open('learning_history.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"No se pudo cargar el historial de aprendizaje: {e}")
    return {}

def save_learning_history(history):
    """Guarda el historial de aprendizaje EXACTO del legacy"""
    try:
        with open('learning_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"No se pudo guardar el historial de aprendizaje: {e}")

def get_adaptive_parameters(demand_signature, learning_history):
    """Obtiene parámetros adaptativos basados en el historial EXACTO del legacy"""
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
        # Parámetros iniciales equilibrados EXACTOS del legacy
        return {
            'agent_limit_factor': 22,
            'excess_penalty': 0.5,
            'peak_bonus': 1.5,
            'critical_bonus': 2.0
        }

def update_learning_history(demand_signature, params, results, learning_history):
    """Actualiza el historial con nuevos resultados EXACTO del legacy"""
    if demand_signature not in learning_history:
        learning_history[demand_signature] = {'runs': []}
    
    # Calcular score de calidad (menor es mejor) EXACTO del legacy
    score = results['understaffing'] + results['overstaffing'] * 0.3
    
    run_data = {
        'params': params,
        'score': score,
        'total_agents': results['total_agents'],
        'coverage': results['coverage_percentage'],
        'timestamp': time.time()
    }
    
    learning_history[demand_signature]['runs'].append(run_data)
    
    # Mantener solo los últimos 10 runs EXACTO del legacy
    if len(learning_history[demand_signature]['runs']) > 10:
        learning_history[demand_signature]['runs'] = \
            learning_history[demand_signature]['runs'][-10:]
    
    return learning_history

# ——————————————————————————————————————————————————————————————
# Funciones de generación de patrones EXACTAS del legacy
# ——————————————————————————————————————————————————————————————

def get_optimal_break_time(start_hour, shift_duration, day, demand_day, break_from_start=2.5, break_from_end=2.5):
    """
    Selecciona el mejor horario de break para un día específico según la demanda EXACTO del legacy
    """
    break_earliest = start_hour + break_from_start
    break_latest = start_hour + shift_duration - break_from_end
    
    if break_latest <= break_earliest:
        return break_earliest
    
    # Generar opciones de break cada 30 minutos EXACTO del legacy
    break_options = []
    current_time = break_earliest
    while current_time <= break_latest:
        break_options.append(current_time)
        current_time += 0.5
    
    # Evaluar cada opción según la demanda del día EXACTO del legacy
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

def get_valid_break_times(start_hour, duration, break_from_start=2.5, break_from_end=2.5):
    """Obtiene todas las franjas válidas de break para un turno EXACTO del legacy"""
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
    """Genera patrón semanal con break específico EXACTO del legacy - CORREGIDO para turnos que cruzan medianoche"""
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

def generate_shift_patterns(
    demand_matrix: np.ndarray | None = None,
    *,
    keep_percentage: float = 0.3,
    peak_bonus: float = 1.5,
    critical_bonus: float = 2.0,
    ACTIVE_DAYS=None,
    use_ft=True,
    use_pt=True,
    allow_8h=True,
    allow_10h8=False,
    allow_pt_4h=True,
    allow_pt_6h=True,
    allow_pt_5h=False,
    break_from_start=2.5,
    break_from_end=2.5,
    first_hour=8,
    last_hour=20
) -> Dict[str, np.ndarray]:
    """Genera patrones exhaustivos con múltiples franjas de break EXACTO del legacy."""
    if ACTIVE_DAYS is None:
        ACTIVE_DAYS = list(range(7))
    
    shifts_coverage = {}
    
    # Horas de inicio cada 30 minutos EXACTO del legacy
    start_hours = np.arange(max(6, first_hour), min(last_hour - 2, 20), 0.5)
    
    # TURNOS FULL TIME con múltiples opciones de break EXACTO del legacy
    if use_ft:
        if allow_8h:
            for start_hour in start_hours:
                for working_combo in combinations(ACTIVE_DAYS, min(6, len(ACTIVE_DAYS))):
                    non_working = [d for d in ACTIVE_DAYS if d not in working_combo]
                    for dso_day in non_working + [None]:
                        # Generar múltiples patrones con diferentes franjas de break
                        break_options = get_valid_break_times(start_hour, 8, break_from_start, break_from_end)
                        for break_idx, break_start in enumerate(break_options):
                            pattern = generate_weekly_pattern_with_break(start_hour, 8, list(working_combo), dso_day, break_start)
                            dso_suffix = f"_DSO{dso_day}" if dso_day is not None else ""
                            shifts_coverage[f"FT8_{start_hour:04.1f}_DAYS{''.join(map(str,working_combo))}_BRK{break_start:04.1f}{dso_suffix}"] = pattern
        
        if allow_10h8:
            for start_hour in start_hours[::2]:
                for working_combo in combinations(ACTIVE_DAYS, min(5, len(ACTIVE_DAYS))):
                    non_working = [d for d in ACTIVE_DAYS if d not in working_combo]
                    for dso_day in non_working + [None]:
                        for eight_day in working_combo:
                            break_options = get_valid_break_times(start_hour, 10, break_from_start, break_from_end)
                            for break_start in break_options[:3]:  # Limitar opciones
                                pattern = generate_weekly_pattern_10h8(start_hour, list(working_combo), eight_day, break_len=1)
                                dso_suffix = f"_DSO{dso_day}" if dso_day is not None else ""
                                shifts_coverage[f"FT10p8_{start_hour:04.1f}_DAYS{''.join(map(str,working_combo))}_8{eight_day}_BRK{break_start:04.1f}{dso_suffix}"] = pattern
    
    # TURNOS PART TIME (sin break por ser cortos) EXACTO del legacy
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
                            pattern = generate_weekly_pattern_pt5(start_hour, list(working_combo))
                            shifts_coverage[f"PT5_{start_hour:04.1f}_DAYS{''.join(map(str,working_combo))}"] = pattern
    
    return shifts_coverage


def generate_shifts_coverage_corrected(*, max_patterns: int | None = None, batch_size: int | None = None, 
                                     use_ft=True, use_pt=True, allow_8h=True, allow_10h8=False,
                                     allow_pt_4h=True, allow_pt_6h=True, allow_pt_5h=False,
                                     ACTIVE_DAYS=None, break_from_start=2.5, break_from_end=2.5,
                                     optimization_profile="Equilibrado", template_cfg=None):
    """
    Genera patrones semanales completos con breaks variables por día
    y permite limitar el número máximo generado.
    """
    if ACTIVE_DAYS is None:
        ACTIVE_DAYS = list(range(7))
    
    shifts_coverage = {}
    seen_patterns = set()
    total_patterns = 0
    current_patterns = 0
    
    # Horarios de inicio optimizados
    step = 0.5
    slot_minutes = 60
    if optimization_profile == "JEAN Personalizado":
        slot_minutes = template_cfg.get("slot_duration_minutes", 30) if template_cfg else 30
        step = slot_minutes / 60
    slots_per_day = 24 * (60 // slot_minutes)
    if max_patterns is None:
        max_patterns = memory_limit_patterns(slots_per_day)

    start_hours = [h for h in np.arange(0, 24, step) if h <= 23.5]

    # Perfil JEAN Personalizado: leer patrones desde JSON y retornar
    if optimization_profile == "JEAN Personalizado" and template_cfg:
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
    
    # ===== TURNOS FULL TIME =====
    if use_ft:
        # 8 horas - 6 días de trabajo
        if allow_8h:
            for start_hour in start_hours:
                for dso_day in ACTIVE_DAYS:
                    working_days = [d for d in ACTIVE_DAYS if d != dso_day][:6]
                    if len(working_days) >= 6 and 8 * len(working_days) <= 48:
                        weekly_pattern = generate_weekly_pattern(
                            start_hour, 8, working_days, dso_day, break_from_start=break_from_start, break_from_end=break_from_end
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
                        if max_patterns is not None and len(shifts_coverage) >= max_patterns:
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
                                start_hour, working_days, eight_day, break_from_start=break_from_start, break_from_end=break_from_end
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
                                if batch_size:
                                    if shifts_coverage:
                                        yield shifts_coverage
                                else:
                                    yield shifts_coverage
                                return
    
    if shifts_coverage:
        yield shifts_coverage
    return


def generate_shifts_coverage_optimized(
    demand_matrix: np.ndarray,
    *,
    max_patterns: int | None = None,
    batch_size: int = 2000,
    quality_threshold: int = 0,
    use_ft=True, use_pt=True, allow_8h=True, allow_10h8=False,
    allow_pt_4h=True, allow_pt_6h=True, allow_pt_5h=False,
    ACTIVE_DAYS=None, break_from_start=2.5, break_from_end=2.5,
    optimization_profile="Equilibrado", template_cfg=None
) -> Iterable[dict[str, np.ndarray]]:
    """Generate and score patterns in batches.

    Patterns are produced using ``generate_shifts_coverage_corrected`` and
    filtered immediately based on ``score_pattern``.  Only patterns with a score
    at least ``quality_threshold`` are yielded.  Duplicates are removed using a
    set of hashes and generation stops once ``max_patterns`` patterns have been
    emitted.
    """

    selected = 0
    start_time = time.time()
    seen: set[bytes] = set()

    inner = generate_shifts_coverage_corrected(
        batch_size=batch_size, 
        use_ft=use_ft, use_pt=use_pt, allow_8h=allow_8h, allow_10h8=allow_10h8,
        allow_pt_4h=allow_pt_4h, allow_pt_6h=allow_pt_6h, allow_pt_5h=allow_pt_5h,
        ACTIVE_DAYS=ACTIVE_DAYS, break_from_start=break_from_start, break_from_end=break_from_end,
        optimization_profile=optimization_profile, template_cfg=template_cfg
    )
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
            show_generation_progress(selected, start_time)
            yield batch
            gc.collect()
        if max_patterns is not None and selected >= max_patterns:
            break


def generate_weekly_pattern(start_hour, duration, working_days, dso_day=None, break_len=1, break_from_start=2.5, break_from_end=2.5):
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

def generate_weekly_pattern_10h8(start_hour, working_days, eight_hour_day, break_len=1, break_from_start=2.5, break_from_end=2.5):
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

def generate_weekly_pattern_advanced(start_hour, duration, working_days, break_position, break_from_start=2.5, break_from_end=2.5):
    """Genera patrón semanal avanzado con break posicionado dinámicamente EXACTO del legacy"""
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

def calculate_comprehensive_score(current_coverage, new_coverage, demand_matrix, critical_days, peak_hours, strategy):
    """Scoring que balancea déficit vs exceso y promueve eficiencia EXACTO del legacy"""
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
        excess_penalty_value = excess_increase * 3  # Triple penalización
    elif total_current_excess > 50:
        excess_penalty_value = excess_increase * 2  # Doble penalización
    else:
        excess_penalty_value = excess_increase
    
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
            critical_bonus_value += day_improvement * 2.0  # critical_bonus
    
    peak_bonus_value = 0
    for day in range(len(new_coverage)):
        for hour in peak_hours:
            if hour < len(new_coverage[day]):
                hour_improvement = max(0, current_deficit[day, hour] - new_deficit[day, hour])
                peak_bonus_value += hour_improvement * 1.5  # peak_bonus
    
    return deficit_reduction + efficiency_bonus + critical_bonus_value + peak_bonus_value - excess_penalty_value

def evaluate_solution_quality(coverage, demand_matrix):
    """
    Evalúa la calidad general de una solución EXACTO del legacy
    """
    deficit = np.sum(np.maximum(0, demand_matrix - coverage))
    excess = np.sum(np.maximum(0, coverage - demand_matrix))
    return deficit + excess * 0.5  # Penalizar exceso menos que déficit
# ——————————————————————————————————————————————————————————————
# Funciones de optimización EXACTAS del legacy
# ——————————————————————————————————————————————————————————————

def single_model(func):
    """Ensure that only one optimization model is built/solved at a time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with _MODEL_LOCK:
            return func(*args, **kwargs)
    return wrapper

def optimize_with_phased_strategy(shifts_coverage, demand_matrix, use_ft=True, use_pt=True):
    """Optimización en fases utilizando la estrategia FT→PT"""
    if not PULP_AVAILABLE:
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix)

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


def optimize_single_type(shifts, demand_matrix, shift_type, agent_limit_factor=12, excess_penalty=2.0, 
                        peak_bonus=1.5, critical_bonus=2.0, TIME_SOLVER=240):
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

@single_model
def optimize_with_precision_targeting(shifts_coverage, demand_matrix, agent_limit_factor=12, excess_penalty=2.0,
                                    peak_bonus=1.5, critical_bonus=2.0, TIME_SOLVER=240):
    """Optimización ultra-precisa para cobertura exacta"""
    if not PULP_AVAILABLE:
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix)
    
    try:
        shifts_list = list(shifts_coverage.keys())
        if not shifts_list:
            return {}, "NO_SHIFTS"
        
        prob = pulp.LpProblem("Precision_Scheduling", pulp.LpMinimize)
        
        # Variables con límites dinámicos basados en demanda
        total_demand = demand_matrix.sum()
        peak_demand = demand_matrix.max()
        max_per_shift = max(15, int(total_demand / max(1, len(shifts_list) / 10)))
        
        shift_vars = {}
        for shift in shifts_list:
            shift_vars[shift] = pulp.LpVariable(f"shift_{shift}", 0, max_per_shift, pulp.LpInteger)
        
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
        dynamic_agent_limit = max(
            int(total_demand / max(1, agent_limit_factor)),
            int(peak_demand * 1.1),
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

        if prob.status == pulp.LpStatusOptimal:
            assignments = {}
            for shift in shifts_list:
                try:
                    value = int(shift_vars[shift].varValue or 0)
                except Exception:
                    value = 0
                if value > 0:
                    assignments[shift] = value
            return assignments, "PRECISION_TARGETING"

        print(f"⚠️ Solver status: {prob.status}, usando fallback inteligente")
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix)
        
    except Exception as e:
        print(f"Error en optimización de precisión: {str(e)}")
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix)

def optimize_ft_then_pt_strategy(shifts_coverage, demand_matrix, agent_limit_factor=12, excess_penalty=2.0, TIME_SOLVER=240):
    """Estrategia 2 fases: FT sin exceso, luego PT para completar"""
    try:
        # Separar turnos por tipo
        ft_shifts = {k: v for k, v in shifts_coverage.items() if k.startswith('FT')}
        pt_shifts = {k: v for k, v in shifts_coverage.items() if k.startswith('PT')}
        
        # FASE 1: Optimizar FT sin exceso
        print("🏢 Fase 1: Full Time (SIN exceso)...")
        
        ft_assignments = optimize_ft_no_excess(ft_shifts, demand_matrix, agent_limit_factor, TIME_SOLVER)
        
        # Calcular cobertura FT
        ft_coverage = np.zeros_like(demand_matrix)
        for shift_name, count in ft_assignments.items():
            slots_per_day = len(ft_shifts[shift_name]) // 7
            pattern = np.array(ft_shifts[shift_name]).reshape(7, slots_per_day)
            ft_coverage += pattern * count
        
        # FASE 2: PT para completar déficit
        print("⏰ Fase 2: Part Time (completar déficit)...")
        
        remaining_demand = np.maximum(0, demand_matrix - ft_coverage)
        pt_assignments = optimize_pt_complete(pt_shifts, remaining_demand, agent_limit_factor, excess_penalty, TIME_SOLVER)
        
        # Combinar resultados
        final_assignments = {**ft_assignments, **pt_assignments}
        
        return final_assignments, "FT_NO_EXCESS_THEN_PT"
        
    except Exception as e:
        print(f"Error en estrategia 2 fases: {str(e)}")
        return optimize_with_precision_targeting(shifts_coverage, demand_matrix)

@single_model
def optimize_ft_no_excess(ft_shifts, demand_matrix, agent_limit_factor=12, TIME_SOLVER=240):
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
    for shift in ft_shifts.keys():
        try:
            value = int(ft_vars[shift].varValue or 0)
        except Exception:
            value = 0
        if value > 0:
            ft_assignments[shift] = value
    
    return ft_assignments

@single_model
def optimize_pt_complete(pt_shifts, remaining_demand, agent_limit_factor=12, excess_penalty=2.0, TIME_SOLVER=240, optimization_profile="Equilibrado"):
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
    for shift in pt_shifts.keys():
        try:
            value = int(pt_vars[shift].varValue or 0)
        except Exception:
            value = 0
        if value > 0:
            pt_assignments[shift] = value
    
    return pt_assignments

@single_model
def optimize_with_relaxed_constraints(shifts_coverage, demand_matrix, TIME_SOLVER=240):
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
        print(f"Error en optimización relajada: {str(e)}")
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix)

def optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix, agent_limit_factor=12, excess_penalty=2.0,
                                    peak_bonus=1.5, critical_bonus=2.0):
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
        
        for iteration in range(max_agents):
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
        
        return assignments, f"GREEDY_ENHANCED"
        
    except Exception as e:
        print(f"Error en greedy mejorado: {str(e)}")
        return {}, "ERROR"

def optimize_jean_search(
    shifts_coverage,
    demand_matrix,
    target_coverage=98.0,
    max_iterations=5,
    verbose=False,
    agent_limit_factor=30,
    excess_penalty=5.0,
    peak_bonus=2.0,
    critical_bonus=2.5,
    iteration_time_limit=None,
    job_id=None,
):
    """Búsqueda iterativa EXACTA del legacy para el perfil JEAN minimizando exceso y déficit."""
    try:
        cfg = current_app.config
    except Exception:
        cfg = {}
    if iteration_time_limit is None:
        iteration_time_limit = cfg.get("TIME_SOLVER", 45)
    # Limitar el tiempo de cada iteración para evitar que la interfaz quede congelada
    iteration_time_limit = min(
        iteration_time_limit, cfg.get("JEAN_ITER_TIME_LIMIT", 25)
    )

    best_assignments = {}
    best_method = ""
    best_score = float("inf")
    best_coverage = 0

    factor = agent_limit_factor
    iteration = 0
    while iteration < max_iterations and factor >= 1:

        print(f"[JEAN] iter={iteration + 1}, factor={factor}, job={job_id}")
        try:
            if job_id is not None:
                update_progress(
                    job_id,
                    {
                        "job_id": job_id,
                        "jean_iter": iteration + 1,
                        "jean_total_iters": max_iterations,
                        "running": True,
                    },
                )
                # Snapshot ligero que no toca assignments
                D, H = demand_matrix.shape
                pat_small = _compact_patterns_for({}, shifts_coverage, D, H)
                _write_partial_result(
                    job_id,
                    None,
                    pat_small,
                    demand_matrix,
                    meta={
                        "iteration": iteration + 1,
                        "factor": factor,
                        "day_labels": [f"Día {i+1}" for i in range(D)],
                        "hour_labels": list(range(H)),
                    },
                )
        except Exception:
            pass
        if verbose:
            print(f"🔍 JEAN Iteración {iteration + 1}/{max_iterations}: factor {factor}")

        iter_start = time.time()
        assignments, method = optimize_with_precision_targeting(
            shifts_coverage,
            demand_matrix,
            agent_limit_factor=factor,
            excess_penalty=excess_penalty,
            peak_bonus=peak_bonus,
            critical_bonus=critical_bonus,
            TIME_SOLVER=iteration_time_limit,
        )
        results = analyze_results(assignments, shifts_coverage, demand_matrix)
        iter_elapsed = time.time() - iter_start
        # Publicar snapshot parcial de esta iteración para que la UI siempre muestre progreso
        try:
            if job_id is not None:
                D, H = demand_matrix.shape
                pat_small = _compact_patterns_for(assignments or {}, shifts_coverage, D, H)
                _write_partial_result(
                    job_id,
                    assignments or {},
                    pat_small,
                    demand_matrix,
                    meta={
                        "iteration": iteration + 1,
                        "factor": factor,
                        "day_labels": [f"Día {i+1}" for i in range(D)],
                        "hour_labels": list(range(H)),
                    },
                )
                update_progress(
                    job_id,
                    {
                        "jean_iter": iteration + 1,
                        "jean_factor": factor,
                        "jean_status": "done",
                        "jean_time": round(iter_elapsed, 2),
                    },
                )
        except Exception:
            pass
        if results:
            cov = results["coverage_percentage"]
            score = results["overstaffing"] + results["understaffing"]
            if verbose:
                print(
                    f"Iteración {iteration + 1}: factor {factor}, cobertura {cov:.1f}%, score {score:.1f}"
                )

            # Priorizar cobertura objetivo; terminar si se alcanza
            if cov >= target_coverage:
                best_assignments, best_method = assignments, method
                best_score = score
                best_coverage = cov
                if verbose:
                    print(
                        f"✅ Cobertura objetivo alcanzada: {cov:.1f}% cobertura, score {score:.1f}"
                    )
                break
            elif cov > best_coverage and not best_assignments:
                # Solo guardar si no tenemos nada mejor aún
                best_assignments, best_method, best_coverage = assignments, method, cov
                if verbose:
                    print(f"📊 Solución parcial guardada: {cov:.1f}% cobertura")

        factor = max(1, int(factor * 0.9))
        iteration += 1

    if verbose:
        print(
            f"🏁 JEAN completado: mejor cobertura {best_coverage:.1f}%, score {best_score:.1f}"
        )

    return best_assignments, best_method


def optimize_schedule_iterative(shifts_coverage, demand_matrix, optimization_profile="Equilibrado", 
                              use_ft=True, use_pt=True, TARGET_COVERAGE=98.0, VERBOSE=False,
                              agent_limit_factor=12, excess_penalty=2.0, peak_bonus=1.5, critical_bonus=2.0):
    """Función principal con estrategia FT primero + PT después"""
    if PULP_AVAILABLE:
        if optimization_profile == "JEAN":
            print("🔍 **Búsqueda JEAN**: cobertura sin exceso")
            return optimize_jean_search(
                shifts_coverage,
                demand_matrix,
                verbose=VERBOSE,
                target_coverage=TARGET_COVERAGE,
                agent_limit_factor=agent_limit_factor,
                excess_penalty=excess_penalty,
                peak_bonus=peak_bonus,
                critical_bonus=critical_bonus,
                iteration_time_limit=current_app.config.get("TIME_SOLVER"),
            )

        if optimization_profile == "JEAN Personalizado":
            if use_ft and use_pt:
                print("🏢⏰ **Estrategia 2 Fases**: FT sin exceso → PT para completar")
                assignments, method = optimize_ft_then_pt_strategy(shifts_coverage, demand_matrix,
                                                                 agent_limit_factor=agent_limit_factor, excess_penalty=excess_penalty)

                results = analyze_results(assignments, shifts_coverage, demand_matrix)
                if results:
                    cov = results["coverage_percentage"]
                    score = results["overstaffing"] + results["understaffing"]
                    if cov < TARGET_COVERAGE or score > 0:
                        print("♻️ **Refinando con búsqueda JEAN**")
                        assignments, method = optimize_jean_search(
                            shifts_coverage,
                            demand_matrix,
                            verbose=VERBOSE,
                            target_coverage=TARGET_COVERAGE,
                            agent_limit_factor=agent_limit_factor,
                            excess_penalty=excess_penalty,
                            peak_bonus=peak_bonus,
                            critical_bonus=critical_bonus,
                            iteration_time_limit=current_app.config.get("TIME_SOLVER"),
                        )

                return assignments, method
            else:
                print("🔍 **Búsqueda JEAN**: cobertura sin exceso")
                return optimize_jean_search(
                    shifts_coverage,
                    demand_matrix,
                    verbose=VERBOSE,
                    target_coverage=TARGET_COVERAGE,
                    agent_limit_factor=agent_limit_factor,
                    excess_penalty=excess_penalty,
                    peak_bonus=peak_bonus,
                    critical_bonus=critical_bonus,
                    iteration_time_limit=current_app.config.get("TIME_SOLVER"),
                )

        if use_ft and use_pt:
            print("🏢⏰ **Estrategia 2 Fases**: FT sin exceso → PT para completar")
            return optimize_ft_then_pt_strategy(shifts_coverage, demand_matrix, agent_limit_factor=agent_limit_factor, excess_penalty=excess_penalty)
        else:
            print("🎯 **Modo Precisión**: Optimización directa")
            return optimize_with_precision_targeting(shifts_coverage, demand_matrix, agent_limit_factor=agent_limit_factor,
                                                   excess_penalty=excess_penalty, peak_bonus=peak_bonus, critical_bonus=critical_bonus)
    else:
        print("🔄 **Solver Básico**: Greedy mejorado")
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix, agent_limit_factor=agent_limit_factor,
                                               excess_penalty=excess_penalty, peak_bonus=peak_bonus, critical_bonus=critical_bonus)

def solve_in_chunks_optimized(
    shifts_coverage: Dict[str, np.ndarray],
    demand_matrix: np.ndarray,
    base_chunk_size: int = 10000,
    optimization_profile="Equilibrado", use_ft=True, use_pt=True, TARGET_COVERAGE=98.0,
    agent_limit_factor=12, excess_penalty=2.0, peak_bonus=1.5, critical_bonus=2.0
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
        assigns, _ = optimize_schedule_iterative(chunk_dict, remaining, optimization_profile=optimization_profile,
                                               use_ft=use_ft, use_pt=use_pt, TARGET_COVERAGE=TARGET_COVERAGE,
                                               agent_limit_factor=agent_limit_factor, excess_penalty=excess_penalty,
                                               peak_bonus=peak_bonus, critical_bonus=critical_bonus)
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
# Análisis de resultados y exportación EXACTOS del legacy
# ——————————————————————————————————————————————————————————————

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
            print(f"⚠️ {shift_name} excede el máximo de {max_allowed}h (tiene {weekly_hours}h)")
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


def _write_partial_result(job_id, assignments, patterns, demand_matrix, *, meta=None):
    """
    Graba un snapshot ligero en: %TEMP%/scheduler_result_{job_id}.json

    - No sobreescribe assignments válidos con None.
    - Calcula y persiste 'charts' (demand, coverage, deficit, excess) si tiene
      assignments + patterns + demand_matrix.
    """
    import os, json, time, tempfile
    from collections import defaultdict
    try:
        import numpy as np
    except Exception:
        np = None  # evitamos fallo por import

    path = os.path.join(tempfile.gettempdir(), f"scheduler_result_{job_id}.json")

    # lee previo
    try:
        with open(path, "r", encoding="utf-8") as fh:
            existing = json.load(fh)
    except Exception:
        existing = {}

    pr = existing.setdefault("pulp_results", {})
    meta = (meta or {})

    # ---------- anti-borrado ----------
    wrote_new = False
    if assignments is None:
        # no tocar assignments previos
        pass
    else:
        # contamos "cuánto" nuevo hay. Soportamos dos formas:
        # a) assignments = {agente: patron}
        # b) assignments = {patron: cantidad}
        def _safe_sum(d):
            try:
                return sum(int(v) for v in d.values())
            except Exception:
                return 0

        prev_cnt = int(pr.get("assignments_count", 0))
        # intentamos deducir el tipo
        new_cnt = 0
        try:
            vals = list(assignments.values())
            if vals and isinstance(vals[0], (str, int, tuple, dict)):
                # tipo a) probablemente {agente: patron} o similar
                new_cnt = len(assignments)
            else:
                new_cnt = _safe_sum(assignments)
        except Exception:
            new_cnt = _safe_sum(assignments)

        # solo aceptamos algo si trae contenido y no es "menos" que lo previo
        if new_cnt > 0 and new_cnt >= prev_cnt:
            pr["assignments"] = assignments
            pr["assignments_count"] = int(new_cnt)
            pr["status"] = "PARTIAL"
            wrote_new = True

    # ---------- métricas mínimas y heatmaps ----------
    if (
        np is not None
        and demand_matrix is not None
        and patterns is not None
        and pr.get("assignments")
    ):
        try:
            D, H = demand_matrix.shape
            # reconstruir cobertura desde patrones compactos
            total_cov = np.zeros((D, H), dtype=int)
            for a in pr["assignments"].values():
                pid = a.get("pattern") if isinstance(a, dict) else a
                if pid in patterns:
                    mat = np.asarray(patterns[pid]["matrix"], dtype=int).reshape(D, H)
                    total_cov += mat

            demand = np.asarray(demand_matrix, dtype=int)
            diff = total_cov - demand

            cov_curve = total_cov.sum(axis=0).tolist()
            dem_curve = demand.sum(axis=0).tolist()
            diff_curve = diff.sum(axis=0)

            deficit_curve = np.clip(-diff_curve, 0, None).tolist()
            excess_curve = np.clip(diff_curve, 0, None).tolist()

            demand_total = int(demand.sum())
            covered_total = int(np.minimum(total_cov, demand).sum())
            coverage_pct = (
                covered_total / demand_total * 100.0
            ) if demand_total > 0 else 0.0

            pr["metrics"] = {
                "total_coverage": total_cov.tolist(),
                "diff_matrix": diff.tolist(),
                "coverage_curve": cov_curve,
                "demand_curve": dem_curve,
                "deficit_curve": deficit_curve,
                "excess_curve": excess_curve,
                "coverage_percentage": coverage_pct,
            }

            # Etiquetas livianas para frontend
            pr.setdefault("day_labels", [f"Día {i+1}" for i in range(D)])
            pr.setdefault("hour_labels", list(range(H)))

        except Exception as e:
            pr.setdefault("errors", []).append(f"metrics_failed: {e}")

    pr.setdefault("heatmaps", {})
    try:
        # DEMAND
        fig = create_heatmap(np.asarray(demand_matrix), "Demanda", cmap="RdYlBu_r")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp.name, format="png", bbox_inches="tight")
        plt.close(fig)
        pr["heatmaps"]["demand"] = f"/heatmap/{os.path.basename(tmp.name)}"
    except Exception as e:
        pr.setdefault("errors", []).append(f"heatmap_demand_failed: {e}")

    try:
        if pr.get("metrics"):
            cov = np.asarray(pr["metrics"]["total_coverage"])
            fig = create_heatmap(cov, "Cobertura (PuLP)", cmap="RdYlBu_r")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(tmp.name, format="png", bbox_inches="tight")
            plt.close(fig)
            pr["heatmaps"]["coverage"] = f"/heatmap/{os.path.basename(tmp.name)}"

            diff = np.asarray(pr["metrics"]["diff_matrix"])
            fig = create_heatmap(diff, "Cobertura - Demanda (PuLP)", cmap="RdYlBu_r")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(tmp.name, format="png", bbox_inches="tight")
            plt.close(fig)
            pr["heatmaps"]["difference"] = f"/heatmap/{os.path.basename(tmp.name)}"
    except Exception as e:
        pr.setdefault("errors", []).append(f"heatmap_pulp_failed: {e}")

    # meta liviana (sin campos pesados)
    if meta:
        for k, v in meta.items():
            if k not in ("patterns", "demand_matrix"):
                pr[k] = v

    # persistimos
    existing["pulp_results"] = pr
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, ensure_ascii=False)
            fh.flush(); os.fsync(fh.fileno())
    except Exception:
        pass

    # log corto opcional
    try:
        print(f"[PARTIAL] {path} escrito (PuLP) — assignments={pr.get('assignments_count', 0)}, wrote_new={wrote_new}")
    except Exception:
        pass


def _compact_patterns_for(assignments, patterns, D, H):
    used = set()
    for v in (assignments or {}).values():
        pid = v.get("pattern") if isinstance(v, dict) else v
        if pid is not None:
            used.add(pid)
    small = {}
    for pid in used:
        p = patterns.get(pid)
        if p is None:
            continue
        mat = p.get("matrix", p)
        arr = np.asarray(mat, dtype=int).reshape(D, H).tolist()
        small[pid] = {"matrix": arr}
    return small


# ------------------------------------------------------------
# Utilidades para "modo Streamlit": todo en RAM, sin snapshots
# ------------------------------------------------------------
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
    for v in assignments.values():
        pid = v.get("pattern") if isinstance(v, dict) else v
        p = patterns.get(pid)
        if p is None:
            continue
        mat = _pattern_matrix(p).reshape(D, H)
        cov += mat
    return cov


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _heatmap(matrix, title, day_labels=None, hour_labels=None, annotate=True, cmap="RdYlBu_r"):
    M = np.asarray(matrix)
    D, H = M.shape
    fig, ax = plt.subplots(figsize=(min(1.0 * H, 20), min(0.7 * D + 2, 10)))
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    if hour_labels is None:
        hour_labels = [f"{h:02d}" for h in range(H)]
    if day_labels is None:
        day_labels = [f"Día {i+1}" for i in range(D)]
    ax.set_xticks(range(H)); ax.set_xticklabels(hour_labels, fontsize=8)
    ax.set_yticks(range(D)); ax.set_yticklabels(day_labels, fontsize=9)
    ax.set_title(title); ax.set_xlabel("Hora del día"); ax.set_ylabel("Día")
    if annotate and H <= 30 and D <= 14:
        for i in range(D):
            for j in range(H):
                ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, shrink=0.95)
    return fig


def _build_sync_payload(assignments, patterns, demand_matrix, *, day_labels=None, hour_labels=None, meta=None):
    dem = np.asarray(demand_matrix, dtype=int)
    D, H = dem.shape
    cov = _assigned_matrix_from(assignments, patterns, D, H)
    diff = cov - dem
    total_dem = int(dem.sum())
    total_cov = int(cov.sum())
    deficit = int(np.clip(dem - cov, 0, None).sum())
    excess = int(np.clip(cov - dem, 0, None).sum())
    coverage_pct = (total_cov / total_dem * 100.0) if total_dem > 0 else 0.0

    fig_dem = _heatmap(dem, "Demanda (agentes-hora)", day_labels, hour_labels, annotate=True, cmap="Reds")
    fig_cov = _heatmap(cov, "Cobertura (agentes-hora)", day_labels, hour_labels, annotate=True, cmap="Blues")
    fig_diff = _heatmap(diff, "Cobertura - Demanda (exceso/deficit)", day_labels, hour_labels, annotate=True, cmap="RdBu_r")

    return {
        "metrics": {
            "total_demand": total_dem,
            "total_coverage": total_cov,
            "coverage_percentage": round(coverage_pct, 1),
            "deficit": deficit,
            "excess": excess,
            "agents": len(assignments or {}),
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
    }


# Configuration helper -----------------------------------------------------
try:
    from .scheduler_clean import merge_config as apply_configuration
except Exception:  # pragma: no cover - fallback if clean module unavailable
    DEFAULT_CONFIG = {
        "solver_time": 300,
        "solver_msg": 1,
        "TARGET_COVERAGE": 98.0,
        "agent_limit_factor": 12,
        "excess_penalty": 2.0,
        "peak_bonus": 1.5,
        "critical_bonus": 2.0,
        "iterations": 30,
        "use_ft": True,
        "use_pt": True,
        "allow_8h": True,
        "allow_10h8": False,
        "allow_pt_4h": True,
        "allow_pt_6h": True,
        "allow_pt_5h": False,
        "break_from_start": 2.5,
        "break_from_end": 2.5,
        "ACTIVE_DAYS": list(range(7)),
    }

    def apply_configuration(cfg=None):
        """Merge provided configuration with defaults."""
        merged = DEFAULT_CONFIG.copy()
        if cfg:
            merged.update({k: v for k, v in cfg.items() if v is not None})
        return merged


def run_complete_optimization(file_stream, config=None, generate_charts=False, job_id=None, return_payload=False):
    cfg = apply_configuration(config)
    solver_time = cfg.get("TIME_SOLVER", cfg.get("solver_time"))
    if solver_time is not None:
        current_app.config["TIME_SOLVER"] = solver_time

    df = pd.read_excel(file_stream)
    demand_matrix = load_demand_matrix_from_df(df)
    analysis = analyze_demand_matrix(demand_matrix)

    use_ft = cfg.get("use_ft", True)
    use_pt = cfg.get("use_pt", True)
    allow_8h = cfg.get("allow_8h", True)
    allow_10h8 = cfg.get("allow_10h8", False)
    allow_pt_4h = cfg.get("allow_pt_4h", True)
    allow_pt_6h = cfg.get("allow_pt_6h", True)
    allow_pt_5h = cfg.get("allow_pt_5h", False)
    break_from_start = cfg.get("break_from_start", 2.5)
    break_from_end = cfg.get("break_from_end", 2.5)
    optimization_profile = cfg.get("optimization_profile", "JEAN")
    agent_limit_factor = cfg.get("agent_limit_factor", 12)
    excess_penalty = cfg.get("excess_penalty", 2.0)
    peak_bonus = cfg.get("peak_bonus", 1.5)
    critical_bonus = cfg.get("critical_bonus", 2.0)
    TARGET_COVERAGE = cfg.get("TARGET_COVERAGE", cfg.get("coverage", 98.0))
    VERBOSE = cfg.get("VERBOSE", False)
    ft_first_pt_last = cfg.get("ft_first_pt_last", True)

    patterns = {}
    active_days = [d for d in range(7) if demand_matrix.sum(axis=1)[d] > 0]
    if optimization_profile == "JEAN Personalizado" and cfg.get("custom_shifts_json"):
        patterns = load_shift_patterns(
            cfg.get("custom_shifts_json"),
            start_hours=list(np.arange(0, 24, 0.5)),
            break_from_start=break_from_start,
            break_from_end=break_from_end,
            slot_duration_minutes=30,
            max_patterns=cfg.get("max_patterns"),
            demand_matrix=demand_matrix,
            keep_percentage=0.3,
            peak_bonus=peak_bonus,
            critical_bonus=critical_bonus,
        )
    else:
        for batch in generate_shifts_coverage_optimized(
            demand_matrix,
            max_patterns=cfg.get("max_patterns"),
            batch_size=cfg.get("batch_size", 2000),
            quality_threshold=cfg.get("quality_threshold", 0),
            use_ft=use_ft,
            use_pt=use_pt,
            allow_8h=allow_8h,
            allow_10h8=allow_10h8,
            allow_pt_4h=allow_pt_4h,
            allow_pt_6h=allow_pt_6h,
            allow_pt_5h=allow_pt_5h,
            ACTIVE_DAYS=active_days,
            break_from_start=break_from_start,
            break_from_end=break_from_end,
            optimization_profile=optimization_profile,
            template_cfg=cfg.get("custom_shifts_json"),
        ):
            patterns.update(batch)
            if cfg.get("max_patterns") and len(patterns) >= cfg["max_patterns"]:
                break

    assignments = {}
    status = "NO_SOLUTION"

    if use_ft and use_pt and ft_first_pt_last:
        assignments, status = optimize_ft_then_pt_strategy(
            patterns,
            demand_matrix,
            agent_limit_factor=agent_limit_factor,
            excess_penalty=excess_penalty,
            TIME_SOLVER=current_app.config.get("TIME_SOLVER"),
        )
        results = analyze_results(assignments, patterns, demand_matrix)
        if results:
            coverage = results["coverage_percentage"]
            score = results["overstaffing"] + results["understaffing"]
            if coverage < TARGET_COVERAGE or score > 0:
                refined, status = optimize_jean_search(
                    patterns,
                    demand_matrix,
                    target_coverage=TARGET_COVERAGE,
                    verbose=VERBOSE,
                    agent_limit_factor=agent_limit_factor,
                    excess_penalty=excess_penalty,
                    peak_bonus=peak_bonus,
                    critical_bonus=critical_bonus,
                    iteration_time_limit=current_app.config.get("TIME_SOLVER"),
                )
                if refined:
                    assignments = refined
    elif optimization_profile == "JEAN":
        assignments, status = optimize_jean_search(
            patterns,
            demand_matrix,
            target_coverage=TARGET_COVERAGE,
            verbose=VERBOSE,
            agent_limit_factor=agent_limit_factor,
            excess_penalty=excess_penalty,
            peak_bonus=peak_bonus,
            critical_bonus=critical_bonus,
            iteration_time_limit=current_app.config.get("TIME_SOLVER"),
        )
    else:
        assignments = solve_in_chunks_optimized(
            patterns,
            demand_matrix,
            optimization_profile=optimization_profile,
            use_ft=use_ft,
            use_pt=use_pt,
            TARGET_COVERAGE=TARGET_COVERAGE,
            agent_limit_factor=agent_limit_factor,
            excess_penalty=excess_penalty,
            peak_bonus=peak_bonus,
            critical_bonus=critical_bonus,
        )
        status = f"CHUNKS_OPTIMIZED_{optimization_profile.upper().replace(' ', '_')}"

    pulp_assignments = assignments
    pulp_status = status
    pulp_metrics = analyze_results(pulp_assignments, patterns, demand_matrix) if pulp_assignments else None

    greedy_assignments, greedy_status = optimize_with_greedy(
        patterns, demand_matrix, cfg=cfg
    )
    greedy_metrics = analyze_results(greedy_assignments, patterns, demand_matrix) if greedy_assignments else None

    result = {
        "pulp": {
            "assignments": pulp_assignments,
            "metrics": pulp_metrics,
            "status": pulp_status,
        },
        "greedy": {
            "assignments": greedy_assignments,
            "metrics": greedy_metrics,
            "status": greedy_status,
        },
        "analysis": analysis,
        "config": cfg,
    }

    if return_payload:
        D, H = demand_matrix.shape
        day_labels = [f"Día {i+1}" for i in range(D)]
        hour_labels = list(range(H))
        payload = _build_sync_payload(
            assignments=pulp_assignments or {},
            patterns=patterns,
            demand_matrix=demand_matrix,
            day_labels=day_labels,
            hour_labels=hour_labels,
            meta={"status": pulp_status},
        )
        payload["status"] = pulp_status
        payload["config"] = cfg
        return payload

    return result


run_optimization = run_complete_optimization


def generate_shift_patterns(demand_matrix, *, top_k=20, cfg=None):
    import numpy as _np

    dm = _np.asarray(demand_matrix)
    if dm.ndim != 2 or dm.shape[0] != 7:
        return []

    active_days = [d for d in range(7) if dm[d].sum() > 0]
    if not active_days:
        return []

    durations = []
    if not cfg or cfg.get("use_ft", True):
        durations.append(8)
    durations.append(4)
    durations = sorted(set(durations))

    try:
        dm_scoring = _np.packbits(dm > 0, axis=1).astype(_np.uint8)
    except Exception:
        dm_scoring = dm

    items = []
    for duration in durations:
        for start in range(24):
            pat = _np.zeros((7, 24), dtype=_np.int8)
            for day in active_days:
                end = min(24, start + duration)
                if end > start:
                    pat[day, start:end] = 1
            flat = pat.flatten()
            name = f"CAND_{duration}_{start:02d}"
            try:
                sc = score_pattern(flat, dm_scoring)
            except Exception:
                continue
            items.append((int(sc), name, flat))

    items.sort(key=lambda x: x[0], reverse=True)
    try:
        k = int(top_k) if top_k is not None else len(items)
    except Exception:
        k = 20
    k = max(0, k)
    if not items or k == 0:
        return []
    max_score = items[0][0]
    best = [it for it in items if it[0] == max_score]
    if len(best) >= k:
        return best[:k]
# ximo duplicando el mejor
    out = list(best)
    while len(out) < k:
        s, name, pat = best[0]
        out.append((s, name + f"_v{len(out)+1}", pat))
    return out
