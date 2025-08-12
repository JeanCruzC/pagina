# -*- coding: utf-8 -*-
import json
import time
import os
import gc
import hashlib
from io import BytesIO
from itertools import combinations, permutations
import heapq

import tempfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend for server-side generation
import matplotlib.pyplot as plt
import seaborn as sns
import psutil

# Lookup table for population count and global context
POP = np.fromiter((bin(i).count("1") for i in range(256)), dtype=np.uint8)
CONTEXT = {}


def unpack_pattern(packed: np.ndarray, cols: int) -> np.ndarray:
    """Unpack a packed weekly pattern to shape (7, cols)."""
    return np.unpackbits(packed.reshape(7, -1), axis=1)[:, :cols]

try:
    import pulp as pl
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False

print(f"[OPTIMIZER] PuLP disponible: {PULP_AVAILABLE}")

# Default configuration values used when no override is supplied
DEFAULT_CONFIG = {
    # Streamlit legacy defaults from ``legacy/app1.py``
    "TIME_SOLVER": 240,
    "TARGET_COVERAGE": 98.0,
    "agent_limit_factor": 12,
    "excess_penalty": 2.0,
    "peak_bonus": 1.5,
    "critical_bonus": 2.0,
    "iterations": 30,
    "max_patterns": None,
    "batch_size": 2000,
    "quality_threshold": 0,
    "use_ft": True,
    "use_pt": True,
    "allow_8h": True,
    "allow_10h8": False,
    "allow_pt_4h": True,
    "allow_pt_6h": True,
    "allow_pt_5h": False,
    "break_from_start": 2.5,
    "break_from_end": 2.5,
    "optimization_profile": "Equilibrado (Recomendado)",
    "ACTIVE_DAYS": list(range(7)),
}


def merge_config(cfg=None):
    """Return a configuration dictionary overlaying defaults with ``cfg``."""
    merged = DEFAULT_CONFIG.copy()
    if cfg:
        merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged


def _build_pattern(days, durations, start_hour, break_len, break_from_start,
                   break_from_end, slot_factor=1):
    """Construct a weekly pattern array.

    ``days`` and ``durations`` must align. A break of ``break_len`` hours is
    placed between ``break_from_start`` and ``break_from_end`` from the shift
    start. ``slot_factor`` controls the number of slots per hour.

    Returns a flattened :class:`numpy.ndarray` of shape ``7 * slots_per_day``.
    """
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
    pattern = np.packbits(pattern.astype(bool), axis=1).astype(np.uint8)
    return pattern.reshape(-1)


def memory_limit_patterns(slots_per_day):
    """Return the max number of patterns that fit in memory."""
    if slots_per_day <= 0:
        return 0
    available = psutil.virtual_memory().available
    cap = min(available, 4 * 1024 ** 3)
    return int(cap // (7 * slots_per_day))


def monitor_memory_usage():
    """Return current memory usage percentage."""
    return psutil.virtual_memory().percent


def adaptive_chunk_size(base=5000):
    """Adjust chunk size based on memory usage."""
    usage = monitor_memory_usage()
    if usage > 80:
        return max(1000, base // 4)
    if usage > 60:
        return max(2000, base // 2)
    return base


def emergency_cleanup(threshold=85.0):
    """Trigger ``gc.collect`` if usage exceeds ``threshold``."""
    if monitor_memory_usage() >= threshold:
        gc.collect()
        return True
    return False


def get_smart_start_hours(demand_matrix, max_hours=12):
    """Return a list of start hours around peak demand."""
    if demand_matrix is None or demand_matrix.size == 0:
        return [float(h) for h in range(24)]

    cols = demand_matrix.shape[1]
    hourly_totals = demand_matrix.sum(axis=0)
    top = np.argsort(hourly_totals)[-max_hours:]
    hours = sorted({round(h / cols * 24, 2) for h in top})
    return hours


def score_pattern(pattern_packed: np.ndarray, demand_packed: np.ndarray) -> int:
    """Return coverage count for packed ``pattern`` vs packed demand."""
    pat_bytes = pattern_packed.reshape(7, -1)
    dm_bytes = demand_packed[:, :pat_bytes.shape[1]]
    return int(POP[pat_bytes & dm_bytes].sum())


def _resize_matrix(matrix, target_cols):
    """Resize ``matrix`` to ``target_cols`` via repeat or max pooling."""
    if matrix.shape[1] == target_cols:
        return matrix
    if matrix.shape[1] < target_cols:
        factor = target_cols // matrix.shape[1]
        return np.repeat(matrix, factor, axis=1)
    factor = matrix.shape[1] // target_cols
    return matrix.reshape(matrix.shape[0], target_cols, factor).max(axis=2)


def score_and_filter_patterns(patterns, demand_matrix, *, keep_percentage=0.3,
                              peak_bonus=1.5, critical_bonus=2.0,
                              efficiency_bonus=1.0):
    """Score patterns against demand and keep the best ones."""
    if demand_matrix is None or not patterns:
        return patterns
    dm = np.asarray(demand_matrix, dtype=float)
    dm_packed = np.packbits(dm > 0, axis=1).astype(np.uint8)
    days, hours = dm.shape
    daily_totals = dm.sum(axis=1)
    hourly_totals = dm.sum(axis=0)
    critical_days = (np.argsort(daily_totals)[-2:]
                     if daily_totals.size > 1 else [int(np.argmax(daily_totals))])
    if np.any(hourly_totals > 0):
        thresh = np.percentile(hourly_totals[hourly_totals > 0], 75)
        peak_hours = np.where(hourly_totals >= thresh)[0]
    else:
        peak_hours = []
    scores = []
    for name, pat in patterns.items():
        pat_bytes = pat.reshape(7, -1)
        cols = pat_bytes.shape[1] * 8
        dm_resized = _resize_matrix(dm, cols)
        coverage_val = POP[pat_bytes & dm_packed[:, :pat_bytes.shape[1]]].sum()
        total_hours = POP[pat_bytes].sum()
        score = float(coverage_val)
        if total_hours > 0:
            efficiency = coverage_val / total_hours
            score += efficiency * efficiency_bonus
        if len(critical_days) > 0 or len(peak_hours) > 0:
            pat_full = np.unpackbits(pat_bytes, axis=1)[:, :cols]
            coverage = np.minimum(pat_full, dm_resized)
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


def load_shift_patterns(cfg, *, start_hours=None, break_from_start=2.0,
                         break_from_end=2.0, slot_duration_minutes=30,
                         max_patterns=None, demand_matrix=None,
                         keep_percentage=0.3, peak_bonus=1.5,
                         critical_bonus=2.0, efficiency_bonus=1.0,
                         max_patterns_per_shift=None, smart_start_hours=False):
    """Load shift patterns from ``cfg`` and return them as a dict."""
    if isinstance(cfg, str):
        with open(cfg, "r") as fh:
            data = json.load(fh)
    else:
        data = cfg
    if slot_duration_minutes is not None and 60 % slot_duration_minutes != 0:
        raise ValueError("slot_duration_minutes must divide 60")
    base_slot_min = slot_duration_minutes or 60
    slots_per_day = 24 * (60 // base_slot_min)
    if max_patterns is None:
        max_patterns = memory_limit_patterns(slots_per_day)
    shifts_coverage = {}
    unique_patterns = {}
    for shift in data.get("shifts", []):
        name = shift.get("name", "SHIFT")
        pat = shift.get("pattern", {})
        brk = shift.get("break", 0)
        slot_min = slot_duration_minutes or shift.get("slot_duration_minutes", 60)
        if 60 % slot_min != 0:
            raise ValueError("slot_duration_minutes must divide 60")
        step = slot_min / 60
        slot_factor = 60 // slot_min
        base_hours = list(start_hours) if start_hours is not None else list(np.arange(0, 24, step))
        if smart_start_hours and demand_matrix is not None:
            smart = get_smart_start_hours(demand_matrix)
            sh_hours = [h for h in base_hours if any(abs(h - s) < step/2 for s in smart)]
        else:
            sh_hours = base_hours
        work_days = pat.get("work_days", [])
        segments_spec = pat.get("segments", [])
        segments = []
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
        shift_patterns = {}
        for days_sel in day_combos:
            for perm in set(permutations(segments, len(days_sel))):
                for sh in sh_hours:
                    pattern = _build_pattern(days_sel, perm, sh, brk_len, brk_start, brk_end, slot_factor)
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


def load_learning_data():
    """Load optimization learning data from disk if available."""
    try:
        if os.path.exists("optimization_learning.json"):
            with open("optimization_learning.json", "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {"executions": [], "best_params": {}, "stats": {}}


def save_learning_data(data):
    """Persist optimization learning data to disk."""
    try:
        with open("optimization_learning.json", "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def get_adaptive_params(demand_matrix, target_coverage):
    """Return tuned parameters based on previous executions."""
    learning_data = load_learning_data()
    input_hash = hashlib.md5(str(demand_matrix).encode()).hexdigest()[:12]
    total_demand = demand_matrix.sum()
    peak_demand = demand_matrix.max()
    similar_runs = [e for e in learning_data.get("executions", []) if e.get("input_hash") == input_hash]
    similar_runs.sort(key=lambda x: x.get("timestamp", 0))
    if similar_runs:
        last_run = similar_runs[-1]
        best_run = max(similar_runs, key=lambda x: x.get("coverage", 0))
        last_cov = last_run.get("coverage", 0)
        coverage_gap = target_coverage - last_cov
        base_params = best_run.get("params", {})
        evolution_factor = min(0.3, len(similar_runs) * 0.05)
        if len(similar_runs) >= 3:
            recent_coverages = [run.get("coverage", 0) for run in similar_runs[-3:]]
            if recent_coverages[-1] <= recent_coverages[-2]:
                evolution_factor *= 2
        if coverage_gap > 10:
            return {
                "agent_limit_factor": max(5, int(base_params.get("agent_limit_factor", 20) * (1 - evolution_factor))),
                "excess_penalty": base_params.get("excess_penalty", 0.1) * (1 - evolution_factor),
                "peak_bonus": base_params.get("peak_bonus", 2.0) * (1 + evolution_factor),
                "critical_bonus": base_params.get("critical_bonus", 2.5) * (1 + evolution_factor),
                "precision_mode": True,
                "learned": True,
                "runs_count": len(similar_runs),
                "evolution_step": "aggressive",
            }
        elif coverage_gap > 3:
            return {
                "agent_limit_factor": max(8, int(base_params.get("agent_limit_factor", 20) * (1 - evolution_factor * 0.5))),
                "excess_penalty": base_params.get("excess_penalty", 0.2) * (1 - evolution_factor * 0.5),
                "peak_bonus": base_params.get("peak_bonus", 1.8) * (1 + evolution_factor * 0.5),
                "critical_bonus": base_params.get("critical_bonus", 2.0) * (1 + evolution_factor * 0.5),
                "precision_mode": True,
                "learned": True,
                "runs_count": len(similar_runs),
                "evolution_step": "moderate",
            }
        elif coverage_gap > 0:
            return {
                "agent_limit_factor": max(12, int(base_params.get("agent_limit_factor", 22) * (1 - evolution_factor * 0.2))),
                "excess_penalty": base_params.get("excess_penalty", 0.3) * (1 - evolution_factor * 0.2),
                "peak_bonus": base_params.get("peak_bonus", 1.5) * (1 + evolution_factor * 0.2),
                "critical_bonus": base_params.get("critical_bonus", 1.8) * (1 + evolution_factor * 0.2),
                "precision_mode": False,
                "learned": True,
                "runs_count": len(similar_runs),
                "evolution_step": "fine_tune",
            }
        else:
            noise = np.random.uniform(-0.1, 0.1)
            return {
                "agent_limit_factor": max(15, int(base_params.get("agent_limit_factor", 20) * (1 + noise))),
                "excess_penalty": max(0.01, base_params.get("excess_penalty", 0.5) * (1 + noise)),
                "peak_bonus": base_params.get("peak_bonus", 1.5) * (1 + noise * 0.5),
                "critical_bonus": base_params.get("critical_bonus", 2.0) * (1 + noise * 0.5),
                "precision_mode": False,
                "learned": True,
                "runs_count": len(similar_runs),
                "evolution_step": "explore",
            }
    return {
        "agent_limit_factor": max(8, int(total_demand / max(1, peak_demand) * 3)),
        "excess_penalty": 0.05,
        "peak_bonus": 2.5,
        "critical_bonus": 3.0,
        "precision_mode": True,
        "learned": False,
        "runs_count": 0,
        "evolution_step": "initial",
    }


def save_execution_result(demand_matrix, params, coverage, total_agents, execution_time):
    """Record an execution summary in the learning file."""
    learning_data = load_learning_data()
    input_hash = hashlib.md5(str(demand_matrix).encode()).hexdigest()[:12]
    efficiency_score = coverage / max(1, total_agents * 0.1)
    balance_score = coverage - abs(coverage - 100) * 0.5
    execution_result = {
        "timestamp": time.time(),
        "input_hash": input_hash,
        "params": {
            "agent_limit_factor": params.get("agent_limit_factor"),
            "excess_penalty": params.get("excess_penalty"),
            "peak_bonus": params.get("peak_bonus"),
            "critical_bonus": params.get("critical_bonus"),
        },
        "coverage": coverage,
        "total_agents": total_agents,
        "efficiency_score": efficiency_score,
        "balance_score": balance_score,
        "execution_time": execution_time,
        "demand_total": float(demand_matrix.sum()),
        "evolution_step": params.get("evolution_step", "unknown"),
    }
    learning_data.setdefault("executions", []).append(execution_result)
    pattern_executions = [e for e in learning_data["executions"] if e.get("input_hash") == input_hash]
    if len(pattern_executions) > 50:
        learning_data["executions"] = [e for e in learning_data["executions"] if e.get("input_hash") != input_hash or e.get("timestamp", 0) >= sorted([p.get("timestamp", 0) for p in pattern_executions])[-50]]
    current_best = learning_data.get("best_params", {}).get(input_hash, {})
    new_score = efficiency_score if coverage >= 98 else coverage * 2
    if not current_best or new_score > current_best.get("score", 0):
        learning_data.setdefault("best_params", {})[input_hash] = {
            "params": execution_result["params"],
            "coverage": coverage,
            "total_agents": total_agents,
            "score": new_score,
            "efficiency_score": efficiency_score,
            "timestamp": time.time(),
        }
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
        "last_updated": time.time(),
    }
    save_learning_data(learning_data)
    return True

# ---------------------------------------------------------------------------
# Demand utilities
# ---------------------------------------------------------------------------

def load_demand_excel(file_stream) -> np.ndarray:
    """Parse an Excel file exported from Ntech and return a matrix."""
    df = pd.read_excel(file_stream)
    day_col = [c for c in df.columns if "Día" in c][0]
    demand_col = [c for c in df.columns if "Erlang" in c or "Requeridos" in c][-1]
    dm = df.pivot_table(index=day_col, values=demand_col, columns=df.index % 24, aggfunc="first").fillna(0)
    dm = dm.reindex(range(1, 8)).fillna(0)
    dm = dm.sort_index()
    matrix = dm.to_numpy(dtype=int)
    if matrix.shape[1] != 24:
        matrix = np.pad(matrix, ((0, 0), (0, 24 - matrix.shape[1])), constant_values=0)
    return matrix


def heatmap(matrix: np.ndarray, title: str) -> BytesIO:
    """Return an image buffer with ``matrix`` rendered as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(matrix, ax=ax, cmap="viridis", cbar=False)
    ax.set_title(title)
    ax.set_xlabel("Hora")
    ax.set_ylabel("Dia")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def load_demand_matrix_from_df(df: pd.DataFrame) -> np.ndarray:
    """Return a 7x24 demand matrix from an Ntech formatted dataframe."""

    demand_matrix = np.zeros((7, 24), dtype=float)

    day_col = "Día"
    time_col = "Horario"
    demand_col = "Suma de Agentes Requeridos Erlang"

    if day_col not in df.columns or time_col not in df.columns or demand_col not in df.columns:
        for col in df.columns:
            if "día" in col.lower() or "dia" in col.lower():
                day_col = col
            elif "horario" in col.lower():
                time_col = col
            elif "erlang" in col.lower() or "requeridos" in col.lower():
                demand_col = col

    for _, row in df.iterrows():
        try:
            day = int(row[day_col])
            if not (1 <= day <= 7):
                continue
            day_idx = day - 1

            horario = str(row[time_col])
            if ":" in horario:
                hour = int(horario.split(":")[0])
            else:
                hour = int(float(horario))
            if not (0 <= hour <= 23):
                continue

            demanda = float(row[demand_col])
            demand_matrix[day_idx, hour] = demanda
        except (ValueError, TypeError, IndexError):
            continue

    return demand_matrix


def analyze_demand_matrix(matrix: np.ndarray) -> dict:
    """Return basic metrics from a demand matrix."""
    daily_demand = matrix.sum(axis=1)
    hourly_demand = matrix.sum(axis=0)
    active_days = [d for d in range(7) if daily_demand[d] > 0]
    inactive_days = [d for d in range(7) if daily_demand[d] == 0]
    working_days = len(active_days)
    active_hours = np.where(hourly_demand > 0)[0]
    first_hour = int(active_hours.min()) if active_hours.size else 8
    last_hour = int(active_hours.max()) if active_hours.size else 20
    operating_hours = last_hour - first_hour + 1
    peak_demand = float(matrix.max()) if matrix.size else 0.0
    avg_demand = float(matrix[active_days].mean()) if active_days else 0.0
    daily_totals = matrix.sum(axis=1)
    hourly_totals = matrix.sum(axis=0)
    critical_days = (
        np.argsort(daily_totals)[-2:] if daily_totals.size > 1 else [int(np.argmax(daily_totals))]
    )
    peak_threshold = (
        np.percentile(hourly_totals[hourly_totals > 0], 75)
        if np.any(hourly_totals > 0)
        else 0
    )
    peak_hours = np.where(hourly_totals >= peak_threshold)[0]
    return {
        "daily_demand": daily_demand,
        "hourly_demand": hourly_demand,
        "active_days": active_days,
        "inactive_days": inactive_days,
        "working_days": working_days,
        "first_hour": first_hour,
        "last_hour": last_hour,
        "operating_hours": operating_hours,
        "peak_demand": peak_demand,
        "average_demand": avg_demand,
        "critical_days": critical_days,
        "peak_hours": peak_hours,
    }


def create_heatmap(matrix, title, cmap="RdYlBu_r"):
    """Return a matplotlib figure with ``matrix`` visualised as a heatmap."""
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_yticks(range(7))
    ax.set_yticklabels([
        "Lunes",
        "Martes",
        "Miércoles",
        "Jueves",
        "Viernes",
        "Sábado",
        "Domingo",
    ])
    for i in range(7):
        for j in range(24):
            ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color="black", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Día de la semana")
    plt.colorbar(im, ax=ax)
    return fig


def generate_all_heatmaps(demand, coverage=None, diff=None) -> dict:
    """Generate heatmaps for demand, coverage and difference matrices."""
    maps = {"demand": create_heatmap(demand, "Demanda por Hora y Día", "Reds")}
    if coverage is not None:
        maps["coverage"] = create_heatmap(coverage, "Cobertura por Hora y Día", "Blues")
    if diff is not None:
        maps["difference"] = create_heatmap(diff, "Diferencias por Hora y Día", "RdBu")
    return maps


PROFILES = {
    "Equilibrado (Recomendado)": {
        "agent_limit_factor": 12,
        "excess_penalty": 2.0,
        "peak_bonus": 1.5,
        "critical_bonus": 2.0,
    },
    "Conservador": {
        "agent_limit_factor": 30,
        "excess_penalty": 0.5,
        "peak_bonus": 1.0,
        "critical_bonus": 1.2,
    },
    "Agresivo": {
        "agent_limit_factor": 15,
        "excess_penalty": 0.05,
        "peak_bonus": 1.5,
        "critical_bonus": 2.0,
    },
    "Máxima Cobertura": {
        "agent_limit_factor": 7,
        "excess_penalty": 0.005,
        "peak_bonus": 3.0,
        "critical_bonus": 4.0,
    },
    "Mínimo Costo": {
        "agent_limit_factor": 35,
        "excess_penalty": 0.8,
        "peak_bonus": 0.8,
        "critical_bonus": 1.0,
    },
    "100% Cobertura Eficiente": {
        "agent_limit_factor": 6,
        "excess_penalty": 0.01,
        "peak_bonus": 3.5,
        "critical_bonus": 4.5,
    },
    "100% Cobertura Total": {
        "agent_limit_factor": 5,
        "excess_penalty": 0.001,
        "peak_bonus": 4.0,
        "critical_bonus": 5.0,
    },
    "Cobertura Perfecta": {
        "agent_limit_factor": 8,
        "excess_penalty": 0.01,
        "peak_bonus": 3.0,
        "critical_bonus": 4.0,
    },
    "100% Exacto": {
        "agent_limit_factor": 6,
        "excess_penalty": 0.005,
        "peak_bonus": 4.0,
        "critical_bonus": 5.0,
    },
    "JEAN": {
        "agent_limit_factor": 30,
        "excess_penalty": 5.0,
        "peak_bonus": 2.0,
        "critical_bonus": 2.5,
    },
    "Aprendizaje Adaptativo": {
        "agent_limit_factor": 8,
        "excess_penalty": 0.01,
        "peak_bonus": 3.0,
        "critical_bonus": 4.0,
    },
}


def apply_configuration(cfg=None):
    """Apply an optimization profile over ``cfg`` and return the result."""
    cfg = merge_config(cfg)
    profile = cfg.get("optimization_profile")
    profile_params = PROFILES.get(profile)
    if profile_params:
        for key, val in profile_params.items():
            cfg.setdefault(key, val)
    return cfg


def create_demand_signature(demand_matrix: np.ndarray) -> str:
    """Return a short hash representing the demand pattern."""
    normalized = demand_matrix / (demand_matrix.max() + 1e-8)
    return hashlib.md5(normalized.tobytes()).hexdigest()[:16]


def load_learning_history() -> dict:
    """Load adaptive learning history from disk if available."""
    try:
        if os.path.exists("learning_history.json"):
            with open("learning_history.json", "r") as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}


def save_learning_history(history: dict) -> None:
    """Persist adaptive learning history to disk."""
    try:
        with open("learning_history.json", "w") as fh:
            json.dump(history, fh, indent=2)
    except Exception:
        pass


def get_adaptive_parameters(demand_signature: str, learning_history: dict) -> dict:
    """Return learned parameters for ``demand_signature`` or defaults."""
    if demand_signature in learning_history:
        learned = learning_history[demand_signature]
        best = min(learned.get("runs", []), key=lambda x: x.get("score", 0))
        return {
            "agent_limit_factor": best["params"]["agent_limit_factor"],
            "excess_penalty": best["params"]["excess_penalty"],
            "peak_bonus": best["params"]["peak_bonus"],
            "critical_bonus": best["params"]["critical_bonus"],
        }
    return {
        "agent_limit_factor": 22,
        "excess_penalty": 0.5,
        "peak_bonus": 1.5,
        "critical_bonus": 2.0,
    }


def update_learning_history(demand_signature: str, params: dict, results: dict, history: dict) -> dict:
    """Update ``history`` with new execution ``results`` for ``demand_signature``."""
    if demand_signature not in history:
        history[demand_signature] = {"runs": []}

    score = results["understaffing"] + results["overstaffing"] * 0.3
    history[demand_signature]["runs"].append(
        {
            "params": params,
            "score": score,
            "total_agents": results["total_agents"],
            "coverage": results["coverage_percentage"],
            "timestamp": time.time(),
        }
    )
    history[demand_signature]["runs"] = history[demand_signature]["runs"][-10:]
    return history


def get_optimal_break_time(start_hour: float, shift_duration: int, day: int, demand_day: np.ndarray,
                           *, cfg=None) -> float:
    """Return the best break time for the given day based on demand."""
    cfg = merge_config(cfg)
    break_earliest = start_hour + cfg["break_from_start"]
    break_latest = start_hour + shift_duration - cfg["break_from_end"]
    if break_latest <= break_earliest:
        return break_earliest
    options = []
    current = break_earliest
    while current <= break_latest:
        options.append(current)
        current += 0.5
    best = break_earliest
    min_impact = float("inf")
    for opt in options:
        hour = int(opt) % 24
        if hour < len(demand_day):
            impact = demand_day[hour]
            if impact < min_impact:
                min_impact = impact
                best = opt
    return best


def generate_shift_patterns(demand_matrix=None, *, top_k=100, cfg=None):
    """Generate patterns and keep only the ``top_k`` highest scoring ones.

    Patterns are evaluated against ``demand_matrix`` as they are produced and a
    min-heap is used to retain only the best ``top_k`` entries.  Each heap entry
    stores ``(score, name, pat_packed)`` where ``pat_packed`` is the flattened
    byte representation of the pattern.  The final result is returned as a list
    sorted by score in descending order.
    """
    cfg = merge_config(cfg)
    use_ft = cfg["use_ft"]
    use_pt = cfg["use_pt"]
    allow_8h = cfg["allow_8h"]
    allow_10h8 = cfg["allow_10h8"]
    allow_pt_4h = cfg["allow_pt_4h"]
    allow_pt_6h = cfg["allow_pt_6h"]
    allow_pt_5h = cfg["allow_pt_5h"]
    active_days = cfg["ACTIVE_DAYS"]

    first_hour = 6
    last_hour = 22
    if demand_matrix is not None:
        analysis = analyze_demand_matrix(demand_matrix)
        first_hour = analysis["first_hour"]
        last_hour = analysis["last_hour"]

    start_hours = np.arange(max(6, first_hour), min(last_hour - 2, 20), 0.5)
    heap = []

    def push_pattern(name: str, pattern: np.ndarray) -> None:
        """Score and push ``pattern`` onto the heap, keeping only ``top_k``."""
        pat = pattern.astype(np.int8)
        score = score_pattern(pat, demand_matrix) if demand_matrix is not None else 0
        packed = pat.tobytes()
        entry = (score, name, packed)
        if len(heap) < top_k:
            heapq.heappush(heap, entry)
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, entry)

    if use_ft and allow_8h:
        for start_hour in start_hours:
            for working_combo in combinations(active_days, min(6, len(active_days))):
                non_working = [d for d in active_days if d not in working_combo]
                for dso_day in non_working + [None]:
                    for brk in get_valid_break_times(start_hour, 8, cfg=cfg):
                        pattern = generate_weekly_pattern_with_break(
                            start_hour, 8, list(working_combo), dso_day, brk, cfg=cfg
                        )
                        suffix = f"_DSO{dso_day}" if dso_day is not None else ""
                        name = (
                            f"FT8_{start_hour:04.1f}_DAYS{''.join(map(str,working_combo))}_BRK{brk:04.1f}{suffix}"
                        )
                        push_pattern(name, pattern)

    if use_pt:
        if allow_pt_4h:
            for start_hour in start_hours:
                for num_days in [4, 5, 6]:
                    if num_days <= len(active_days):
                        for combo in combinations(active_days, num_days):
                            pattern = generate_weekly_pattern_simple(start_hour, 4, list(combo))
                            name = f"PT4_{start_hour:04.1f}_DAYS{''.join(map(str,combo))}"
                            push_pattern(name, pattern)
        if allow_pt_6h:
            for start_hour in start_hours[::2]:
                for num_days in [4]:
                    if num_days <= len(active_days):
                        for combo in combinations(active_days, num_days):
                            pattern = generate_weekly_pattern_simple(start_hour, 6, list(combo))
                            name = f"PT6_{start_hour:04.1f}_DAYS{''.join(map(str,combo))}"
                            push_pattern(name, pattern)
        if allow_pt_5h:
            for start_hour in start_hours[::2]:
                for num_days in [5]:
                    if num_days <= len(active_days):
                        for combo in combinations(active_days, num_days):
                            pattern = generate_weekly_pattern_simple(start_hour, 5, list(combo))
                            name = f"PT5_{start_hour:04.1f}_DAYS{''.join(map(str,combo))}"
                            push_pattern(name, pattern)

    # Return heap contents sorted by score (highest first) and unpack patterns
    ordered = sorted(heap, reverse=True)
    return [
        (score, name, np.frombuffer(packed, dtype=np.int8))
        for score, name, packed in ordered
    ]


def get_valid_break_times(start_hour: float, duration: int, *, cfg=None) -> list:
    """Return a list of possible break start times for a shift."""
    cfg = merge_config(cfg)
    earliest = start_hour + cfg["break_from_start"]
    latest = start_hour + duration - cfg["break_from_end"] - 1
    valid = []
    current = earliest
    while current <= latest:
        if current % 0.5 == 0:
            valid.append(current)
        current += 0.5
    return valid[:7]


def generate_weekly_pattern_with_break(start_hour: float, duration: int, working_days: list, dso_day: int | None,
                                        break_start: float, break_len: int = 1, *, cfg=None) -> np.ndarray:
    """Generate a weekly pattern placing the break at ``break_start``."""
    cfg = merge_config(cfg)
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
    pattern = np.packbits(pattern.astype(bool), axis=1).astype(np.uint8)
    return pattern.reshape(-1)


def generate_weekly_pattern_advanced(start_hour: float, duration: int, working_days: list, break_position: float, *, cfg=None) -> np.ndarray:
    """Generate a weekly pattern with a break at ``break_position`` percentage of the shift."""
    cfg = merge_config(cfg)
    pattern = np.zeros((7, 24), dtype=np.int8)
    for day in working_days:
        for h in range(duration):
            hour_idx = int(start_hour + h) % 24
            if hour_idx < 24:
                pattern[day, hour_idx] = 1
        brk_offset = int(duration * break_position)
        brk_hour = int(start_hour + brk_offset) % 24
        if (
            brk_hour >= int(start_hour + cfg["break_from_start"])
            and brk_hour <= int(start_hour + duration - cfg["break_from_end"])
            and brk_hour < 24
        ):
            pattern[day, brk_hour] = 0
    pattern = np.packbits(pattern.astype(bool), axis=1).astype(np.uint8)
    return pattern.reshape(-1)


def evaluate_solution_quality(coverage_matrix: np.ndarray, demand_matrix: np.ndarray) -> float:
    """Return a quality score for a coverage matrix (lower is better)."""
    total_demand = demand_matrix.sum()
    total_coverage = np.minimum(coverage_matrix, demand_matrix).sum()
    coverage_pct = (total_coverage / total_demand) * 100 if total_demand > 0 else 0
    excess = np.maximum(0, coverage_matrix - demand_matrix).sum()
    efficiency = total_coverage / (total_coverage + excess) if (total_coverage + excess) > 0 else 0
    return (100 - coverage_pct) + (excess * 0.1) + ((1 - efficiency) * 50)

# ---------------------------------------------------------------------------
# Pattern generation
# ---------------------------------------------------------------------------

def generate_weekly_pattern(start_hour, duration, working_days, dso_day=None, break_len=1, *, cfg=None):
    """Return a weekly pattern with a single break per day."""
    cfg = merge_config(cfg)
    break_from_start = cfg["break_from_start"]
    break_from_end = cfg["break_from_end"]
    pattern = np.zeros((7, 24), dtype=np.int8)
    for day in working_days:
        if day != dso_day:
            for h in range(duration):
                t = start_hour + h
                d_off, idx = divmod(int(t), 24)
                pattern[(day + d_off) % 7, idx] = 1
            break_start_idx = start_hour + break_from_start
            break_end_idx = start_hour + duration - break_from_end
            if int(break_start_idx) < int(break_end_idx):
                break_hour = int(break_start_idx) + (int(break_end_idx) - int(break_start_idx)) // 2
            else:
                break_hour = int(break_start_idx)
            for b in range(int(break_len)):
                t = break_hour + b
                d_off, idx = divmod(int(t), 24)
                pattern[(day + d_off) % 7, idx] = 0
    pattern = np.packbits(pattern.astype(bool), axis=1).astype(np.uint8)
    return pattern.reshape(-1)


def generate_weekly_pattern_10h8(start_hour, working_days, eight_hour_day, break_len=1, *, cfg=None):
    """Generate a 10h pattern with one 8h day."""
    cfg = merge_config(cfg)
    break_from_start = cfg["break_from_start"]
    break_from_end = cfg["break_from_end"]
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
            break_hour = int(break_start_idx) + (int(break_end_idx) - int(break_start_idx)) // 2
        else:
            break_hour = int(break_start_idx)
        for b in range(int(break_len)):
            t = break_hour + b
            d_off, idx = divmod(int(t), 24)
            pattern[(day + d_off) % 7, idx] = 0
    pattern = np.packbits(pattern.astype(bool), axis=1).astype(np.uint8)
    return pattern.reshape(-1)


def generate_weekly_pattern_simple(start_hour, duration, working_days):
    """Simple pattern without breaks."""
    pattern = np.zeros((7, 24), dtype=np.int8)
    for day in working_days:
        for h in range(duration):
            t = start_hour + h
            d_off, idx = divmod(int(t), 24)
            pattern[(day + d_off) % 7, idx] = 1
    pattern = np.packbits(pattern.astype(bool), axis=1).astype(np.uint8)
    return pattern.reshape(-1)


def generate_weekly_pattern_pt5(start_hour, working_days):
    """Five 5h days with the last reduced to 4h."""
    pattern = np.zeros((7, 24), dtype=np.int8)
    if not working_days:
        pattern = np.packbits(pattern.astype(bool), axis=1).astype(np.uint8)
        return pattern.reshape(-1)
    four_hour_day = working_days[-1]
    for day in working_days:
        hours = 4 if day == four_hour_day else 5
        for h in range(hours):
            t = start_hour + h
            d_off, idx = divmod(int(t), 24)
            pattern[(day + d_off) % 7, idx] = 1
    pattern = np.packbits(pattern.astype(bool), axis=1).astype(np.uint8)
    return pattern.reshape(-1)


def generate_shifts_coverage_corrected(*, max_patterns=None, batch_size=None, cfg=None):
    """Yield raw patterns respecting configuration flags."""
    cfg = merge_config(cfg)
    use_ft = cfg["use_ft"]
    use_pt = cfg["use_pt"]
    allow_8h = cfg["allow_8h"]
    allow_10h8 = cfg["allow_10h8"]
    allow_pt_4h = cfg["allow_pt_4h"]
    allow_pt_6h = cfg["allow_pt_6h"]
    allow_pt_5h = cfg["allow_pt_5h"]
    active_days = cfg["ACTIVE_DAYS"]

    shifts_coverage = {}
    seen_patterns = set()
    step = 0.5
    start_hours = [h for h in np.arange(0, 24, step) if h <= 23.5]
    for start_hour in start_hours:
        if use_ft and allow_8h:
            for dso_day in active_days:
                working_days = [d for d in active_days if d != dso_day][:6]
                if len(working_days) >= 6:
                    pattern = generate_weekly_pattern(start_hour, 8, working_days, dso_day, cfg=cfg)
                    key = pattern.tobytes()
                    if key not in seen_patterns:
                        seen_patterns.add(key)
                        name = f"FT8_{start_hour:04.1f}_DSO{dso_day}"
                        shifts_coverage[name] = pattern
                        if batch_size and len(shifts_coverage) >= batch_size:
                            yield shifts_coverage
                            shifts_coverage = {}
        if use_ft and allow_10h8:
            for dso_day in active_days:
                working_days = [d for d in active_days if d != dso_day][:6]
                if len(working_days) >= 6:
                    pattern = generate_weekly_pattern_10h8(start_hour, working_days, dso_day, cfg=cfg)
                    key = pattern.tobytes()
                    if key not in seen_patterns:
                        seen_patterns.add(key)
                        name = f"FT10p8_{start_hour:04.1f}_DSO{dso_day}"
                        shifts_coverage[name] = pattern
                        if batch_size and len(shifts_coverage) >= batch_size:
                            yield shifts_coverage
                            shifts_coverage = {}
        if use_pt and allow_pt_4h:
            for num_days in [4, 5, 6]:
                if num_days <= len(active_days):
                    for combo in combinations(active_days, num_days):
                        pattern = generate_weekly_pattern_simple(start_hour, 4, list(combo))
                        key = pattern.tobytes()
                        if key not in seen_patterns:
                            seen_patterns.add(key)
                            name = f"PT4_{start_hour:04.1f}_DAYS{''.join(map(str, combo))}"
                            shifts_coverage[name] = pattern
                            if batch_size and len(shifts_coverage) >= batch_size:
                                yield shifts_coverage
                                shifts_coverage = {}
    if shifts_coverage:
        yield shifts_coverage


def generate_shifts_coverage_optimized(
    demand_matrix,
    *,
    max_patterns=None,
    batch_size=2000,
    quality_threshold=0,
    cfg=None,
    demand_packed=None,
):
    """Yield scored pattern batches.

    Logic mirrors ``generate_shifts_coverage_optimized`` in ``legacy/app1.py``
    but omits any Streamlit UI elements.
    """
    cfg = merge_config(cfg)
    if demand_packed is None:
        demand_packed = np.packbits(demand_matrix > 0, axis=1).astype(np.uint8)
    profile = cfg.get('optimization_profile', 'Equilibrado (Recomendado)')
    if profile == 'JEAN Personalizado':
        slot_minutes = int(cfg.get('slot_duration_minutes', 30))
        start_hours = [h for h in np.arange(0, 24, slot_minutes / 60) if h <= 23.5]
        patterns = load_shift_patterns(
            cfg,
            start_hours=start_hours,
            break_from_start=cfg.get('break_from_start', DEFAULT_CONFIG['break_from_start']),
            break_from_end=cfg.get('break_from_end', DEFAULT_CONFIG['break_from_end']),
            slot_duration_minutes=slot_minutes,
            demand_matrix=demand_matrix,
            keep_percentage=cfg.get('keep_percentage', 0.3),
            peak_bonus=cfg.get('peak_bonus', 1.5),
            critical_bonus=cfg.get('critical_bonus', 2.0),
            efficiency_bonus=cfg.get('efficiency_bonus', 1.0),
            max_patterns=cfg.get('max_patterns')
        )
        if not cfg.get('use_ft', True):
            patterns = {k: v for k, v in patterns.items() if not k.startswith('FT')}
        if not cfg.get('use_pt', True):
            patterns = {k: v for k, v in patterns.items() if not k.startswith('PT')}
        yield patterns
        return
    selected = 0
    seen = set()
    inner = generate_shifts_coverage_corrected(batch_size=batch_size, cfg=cfg)
    for raw_batch in inner:
        batch = {}
        for name, pat in raw_batch.items():
            key = hashlib.md5(pat).digest()
            if key in seen:
                continue
            seen.add(key)
            if score_pattern(pat, demand_packed) >= quality_threshold:
                batch[name] = pat
                selected += 1
                if max_patterns is not None and selected >= max_patterns:
                    break
        if batch:
            yield batch
            gc.collect()
        if max_patterns is not None and selected >= max_patterns:
            break


def optimize_with_precision_targeting(shifts_coverage, demand_matrix, *, cfg=None):
    """Precision solver with greedy fallback.

    This implementation follows the logic of ``optimize_with_precision_targeting``
    in ``legacy/app1.py`` but removes all Streamlit UI calls.
    """
    print("[PRECISION] Iniciando optimize_with_precision_targeting")
    print(f"[PRECISION] Numero de turnos: {len(shifts_coverage)}")
    print(f"[PRECISION] Demanda total: {demand_matrix.sum()}")

    cfg = merge_config(cfg)
    TIME_SOLVER = cfg["TIME_SOLVER"]
    agent_limit_factor = cfg["agent_limit_factor"]
    excess_penalty = cfg["excess_penalty"]
    peak_bonus = cfg["peak_bonus"]
    critical_bonus = cfg["critical_bonus"]
    optimization_profile = cfg["optimization_profile"]

    if not PULP_AVAILABLE:
        print("[PRECISION] PuLP no disponible, usando greedy")
        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix, cfg=cfg)

    try:
        shifts_list = list(shifts_coverage.keys())
        if not shifts_list:
            print("[PRECISION] No hay turnos disponibles")
            return {}, "NO_SHIFTS"

        print("[PRECISION] Creando problema PuLP...")
        prob = pl.LpProblem("Precision_Scheduling", pl.LpMinimize)
        print("[PRECISION] Problema PuLP creado")

        total_demand = demand_matrix.sum()
        peak_demand = demand_matrix.max()
        max_per_shift = max(15, int(total_demand / max(1, len(shifts_list) / 10)))

        print(f"[PRECISION] Limite por turno: {max_per_shift}")
        print("[PRECISION] Creando variables...")

        shift_vars = {}
        for i, shift in enumerate(shifts_list):
            if i % 500 == 0:
                print(f"[PRECISION] Variables creadas: {i}/{len(shifts_list)}")
            shift_vars[shift] = pl.LpVariable(f"shift_{shift}", 0, max_per_shift, pl.LpInteger)

        print("[PRECISION] Variables creadas completamente")
        print("[PRECISION] Creando variables de deficit/exceso...")

        deficit_vars = {}
        excess_vars = {}
        hours = demand_matrix.shape[1]
        patterns_unpacked = {
            s: np.unpackbits(p.reshape(7, -1), axis=1)[:, :hours]
            for s, p in shifts_coverage.items()
        }
        for day in range(7):
            for hour in range(hours):
                deficit_vars[(day, hour)] = pl.LpVariable(f"deficit_{day}_{hour}", 0, None)
                excess_vars[(day, hour)] = pl.LpVariable(f"excess_{day}_{hour}", 0, None)

        print("[PRECISION] Variables de deficit/exceso creadas")
        print("[PRECISION] Definiendo funcion objetivo...")

        total_deficit = pl.lpSum([deficit_vars[(day, hour)] for day in range(7) for hour in range(hours)])
        total_agents = pl.lpSum([shift_vars[shift] for shift in shifts_list])

        prob += (total_deficit * 100000 + total_agents * 0.01)

        print("[PRECISION] Funcion objetivo definida")
        print("[PRECISION] Agregando restricciones de cobertura...")

        restriction_count = 0
        for day in range(7):
            for hour in range(hours):
                coverage = pl.lpSum([
                    shift_vars[shift] * patterns_unpacked[shift][day, hour]
                    for shift in shifts_list
                ])
                demand = demand_matrix[day, hour]

                prob += coverage + deficit_vars[(day, hour)] >= demand
                prob += coverage - excess_vars[(day, hour)] <= demand
                restriction_count += 2

                if restriction_count % 100 == 0:
                    print(f"[PRECISION] Restricciones agregadas: {restriction_count}")

        print(f"[PRECISION] Total restricciones: {restriction_count}")
        print("[PRECISION] Configurando solver...")

        solver = pl.PULP_CBC_CMD(
            msg=1,
            timeLimit=30,
            gapRel=0.1,
            threads=1,
            presolve=1,
            cuts=0,
        )

        print("[PRECISION] Ejecutando solver PuLP...")
        prob.solve(solver)
        print(f"[PRECISION] Solver terminado con status: {prob.status}")

        daily_totals = demand_matrix.sum(axis=1)
        hourly_totals = demand_matrix.sum(axis=0)
        critical_days = np.argsort(daily_totals)[-2:] if len(daily_totals) > 1 else [np.argmax(daily_totals)]
        peak_hours = np.where(hourly_totals >= np.percentile(hourly_totals[hourly_totals > 0], 80))[0]
        total_agents = pl.lpSum(shift_vars.values())
        smart_excess_penalty = 0
        for d in range(7):
            for h in range(hours):
                demand_val = demand_matrix[d, h]
                if demand_val == 0:
                    smart_excess_penalty += excess_vars[(d, h)] * 50000
                elif demand_val <= 2:
                    smart_excess_penalty += excess_vars[(d, h)] * (excess_penalty * 100)
                elif demand_val <= 5:
                    smart_excess_penalty += excess_vars[(d, h)] * (excess_penalty * 20)
                else:
                    smart_excess_penalty += excess_vars[(d, h)] * (excess_penalty * 5)

        precision_bonus = 0
        for cd in critical_days:
            if cd < 7:
                day_multiplier = min(5.0, daily_totals[cd] / max(1, daily_totals.mean()))
                for h in range(hours):
                    if demand_matrix[cd, h] > 0:
                        precision_bonus -= deficit_vars[(cd, h)] * (critical_bonus * 100 * day_multiplier)
        for h in peak_hours:
            if h < hours:
                hour_multiplier = min(3.0, hourly_totals[h] / max(1, hourly_totals.mean()))
                for d in range(7):
                    if demand_matrix[d, h] > 0:
                        precision_bonus -= deficit_vars[(d, h)] * (peak_bonus * 50 * hour_multiplier)

        prob += (
            total_deficit * 100000
            + smart_excess_penalty
            + total_agents * 0.01
            + precision_bonus
        )

        for d in range(7):
            for h in range(hours):
                coverage = pl.lpSum(
                    shift_vars[s] * patterns_unpacked[s][d, h] for s in shifts_list
                )
                demand = demand_matrix[d, h]
                prob += coverage + deficit_vars[(d, h)] >= demand
                prob += coverage - excess_vars[(d, h)] <= demand
                if demand == 0:
                    prob += coverage <= 1

        if optimization_profile in ("JEAN", "JEAN Personalizado"):
            dynamic_limit = max(
                int(total_demand / max(1, agent_limit_factor)), int(peak_demand * 1.1)
            )
        else:
            dynamic_limit = max(
                int(total_demand / max(1, agent_limit_factor - 2)), int(peak_demand * 2)
            )
        prob += total_agents <= dynamic_limit
        total_excess = pl.lpSum(excess_vars.values())
        prob += total_excess <= total_demand * 0.10
        for d in range(7):
            day_demand = demand_matrix[d].sum()
            if day_demand > 0:
                day_coverage = pl.lpSum(
                    shift_vars[s] * np.sum(patterns_unpacked[s][d])
                    for s in shifts_list
                )
                prob += day_coverage <= day_demand * 1.15
                prob += day_coverage >= day_demand * 0.85

        assignments = {}
        if prob.status == pl.LpStatusOptimal:
            for s in shifts_list:
                val = int(shift_vars[s].varValue or 0)
                if val > 0:
                    assignments[s] = val
            return assignments, "PRECISION_TARGETING"

        return optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix, cfg=cfg)

    except Exception as e:  # pragma: no cover - debug only
        print(f"[PRECISION] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, "ERROR"


def optimize_ft_then_pt_strategy(shifts_coverage, demand_matrix, *, cfg=None):
    """Optimize full-time first and fill gaps with part-time."""
    ft_shifts = {k: v for k, v in shifts_coverage.items() if k.startswith('FT')}
    pt_shifts = {k: v for k, v in shifts_coverage.items() if k.startswith('PT')}
    ft_assignments = optimize_ft_no_excess(ft_shifts, demand_matrix, cfg=cfg)
    ft_coverage = np.zeros_like(demand_matrix)
    for name, count in ft_assignments.items():
        pattern = np.unpackbits(ft_shifts[name].reshape(7, -1), axis=1)[:, :demand_matrix.shape[1]]
        ft_coverage += pattern * count
    remaining_demand = np.maximum(0, demand_matrix - ft_coverage)
    pt_assignments = optimize_pt_complete(pt_shifts, remaining_demand, cfg=cfg)
    return {**ft_assignments, **pt_assignments}, "FT_NO_EXCESS_THEN_PT"


def optimize_ft_no_excess(ft_shifts, demand_matrix, *, cfg=None):
    """Linear program focusing on full-time coverage only."""
    cfg = merge_config(cfg)
    agent_limit_factor = cfg["agent_limit_factor"]
    TIME_SOLVER = cfg["TIME_SOLVER"]
    if not ft_shifts:
        return {}
    prob = pl.LpProblem("FT_No_Excess", pl.LpMinimize)
    max_ft_per_shift = max(10, int(demand_matrix.sum() / agent_limit_factor))
    ft_vars = {s: pl.LpVariable(f"ft_{s}", 0, max_ft_per_shift, pl.LpInteger) for s in ft_shifts}
    deficit_vars = {}
    hours = demand_matrix.shape[1]
    patterns_unpacked = {
        s: np.unpackbits(p.reshape(7, -1), axis=1)[:, :hours]
        for s, p in ft_shifts.items()
    }
    for d in range(7):
        for h in range(hours):
            deficit_vars[(d, h)] = pl.LpVariable(f"ft_deficit_{d}_{h}", 0, None)
    total_deficit = pl.lpSum(deficit_vars.values())
    total_ft_agents = pl.lpSum(ft_vars.values())
    prob += total_deficit * 1000 + total_ft_agents * 1
    for d in range(7):
        for h in range(hours):
            coverage = pl.lpSum(ft_vars[s] * patterns_unpacked[s][d, h] for s in ft_shifts)
            demand = demand_matrix[d, h]
            prob += coverage + deficit_vars[(d, h)] >= demand
            prob += coverage <= demand
    prob.solve(pl.PULP_CBC_CMD(msg=0, timeLimit=TIME_SOLVER//2))
    assignments = {}
    if prob.status == pl.LpStatusOptimal:
        for s in ft_shifts:
            val = int(ft_vars[s].varValue or 0)
            if val > 0:
                assignments[s] = val
    return assignments


def optimize_pt_complete(pt_shifts, remaining_demand, *, cfg=None):
    """Solve for part-time assignments covering ``remaining_demand``."""
    cfg = merge_config(cfg)
    agent_limit_factor = cfg["agent_limit_factor"]
    excess_penalty = cfg["excess_penalty"]
    TIME_SOLVER = cfg["TIME_SOLVER"]
    optimization_profile = cfg["optimization_profile"]
    if not pt_shifts or remaining_demand.sum() == 0:
        return {}
    prob = pl.LpProblem("PT_Complete", pl.LpMinimize)
    max_pt_per_shift = max(10, int(remaining_demand.sum() / max(1, agent_limit_factor)))
    pt_vars = {s: pl.LpVariable(f"pt_{s}", 0, max_pt_per_shift, pl.LpInteger) for s in pt_shifts}
    deficit_vars = {}
    excess_vars = {}
    hours = remaining_demand.shape[1]
    patterns_unpacked = {
        s: np.unpackbits(p.reshape(7, -1), axis=1)[:, :hours]
        for s, p in pt_shifts.items()
    }
    for d in range(7):
        for h in range(hours):
            deficit_vars[(d, h)] = pl.LpVariable(f"pt_deficit_{d}_{h}", 0, None)
            excess_vars[(d, h)] = pl.LpVariable(f"pt_excess_{d}_{h}", 0, None)
    total_deficit = pl.lpSum(deficit_vars.values())
    total_excess = pl.lpSum(excess_vars.values())
    total_pt_agents = pl.lpSum(pt_vars.values())
    prob += total_deficit * 1000 + total_excess * (excess_penalty * 20) + total_pt_agents * 1
    if optimization_profile in ("JEAN", "JEAN Personalizado"):
        prob += total_excess == 0
    for d in range(7):
        for h in range(hours):
            coverage = pl.lpSum(pt_vars[s] * patterns_unpacked[s][d, h] for s in pt_shifts)
            demand = remaining_demand[d, h]
            prob += coverage + deficit_vars[(d, h)] >= demand
            prob += coverage - excess_vars[(d, h)] <= demand
    prob.solve(pl.PULP_CBC_CMD(msg=0, timeLimit=TIME_SOLVER//2))
    assignments = {}
    if prob.status == pl.LpStatusOptimal:
        for s in pt_shifts:
            val = int(pt_vars[s].varValue or 0)
            if val > 0:
                assignments[s] = val
    return assignments


def optimize_schedule_greedy_enhanced(shifts_coverage, demand_matrix, *, cfg=None):
    """Greedy fallback used when linear programming fails.

    Ported from ``legacy/app1.py`` for simplicity.
    """
    cfg = merge_config(cfg)
    agent_limit_factor = cfg["agent_limit_factor"]
    excess_penalty = cfg["excess_penalty"]
    peak_bonus = cfg["peak_bonus"]
    critical_bonus = cfg["critical_bonus"]

    shifts_list = list(shifts_coverage.keys())
    assignments = {}
    coverage = np.zeros_like(demand_matrix)
    max_agents = max(50, int(demand_matrix.sum() / agent_limit_factor))

    daily_totals = demand_matrix.sum(axis=1)
    hourly_totals = demand_matrix.sum(axis=0)
    critical_days = (
        np.argsort(daily_totals)[-2:]
        if daily_totals.size > 1
        else [int(np.argmax(daily_totals))]
    )
    peak_hours = (
        np.where(hourly_totals >= np.percentile(hourly_totals[hourly_totals > 0], 75))[0]
        if np.any(hourly_totals > 0)
        else []
    )

    for _ in range(max_agents):
        best_shift = None
        best_score = -float("inf")

        for name in shifts_list:
            pattern = np.unpackbits(
                shifts_coverage[name].reshape(7, -1), axis=1
            )[:, :demand_matrix.shape[1]]
            new_cov = coverage + pattern

            current_def = np.maximum(0, demand_matrix - coverage)
            new_def = np.maximum(0, demand_matrix - new_cov)
            deficit_reduction = np.sum(current_def - new_def)

            current_excess = np.maximum(0, coverage - demand_matrix)
            new_excess = np.maximum(0, new_cov - demand_matrix)

            smart_penalty = 0
            for d in range(7):
                for h in range(demand_matrix.shape[1]):
                    if demand_matrix[d, h] == 0 and new_excess[d, h] > current_excess[d, h]:
                        smart_penalty += 1000
                    elif demand_matrix[d, h] <= 2:
                        smart_penalty += (new_excess[d, h] - current_excess[d, h]) * excess_penalty * 10
                    else:
                        smart_penalty += (new_excess[d, h] - current_excess[d, h]) * excess_penalty

            crit_bonus = 0
            for cd in critical_days:
                if cd < 7:
                    crit_bonus += np.sum(current_def[cd] - new_def[cd]) * critical_bonus * 2

            peak_bonus_score = 0
            for h in peak_hours:
                if h < demand_matrix.shape[1]:
                    peak_bonus_score += np.sum(current_def[:, h] - new_def[:, h]) * peak_bonus * 2

            score = deficit_reduction * 100 + crit_bonus + peak_bonus_score - smart_penalty
            if score > best_score:
                best_score = score
                best_shift = name
                best_pattern = pattern

        if best_shift is None or best_score <= 1.0:
            break

        assignments[best_shift] = assignments.get(best_shift, 0) + 1
        coverage += best_pattern
        if not np.any(np.maximum(0, demand_matrix - coverage)):
            break

    return assignments, "GREEDY_FALLBACK"


def solve_with_pulp(demand_matrix, patterns, config):
    """Solve assignment problem using PuLP.

    Parameters
    ----------
    demand_matrix : numpy.ndarray
        Required coverage for each day/hour.
    patterns : dict[str, numpy.ndarray]
        Mapping of shift name to a flattened weekly pattern.
    config : dict
        Configuration overrides.

    Returns
    -------
    dict
        Dictionary with assignments, coverage matrix, total agents and status.
    """
    if not PULP_AVAILABLE:
        raise RuntimeError("PuLP is not available")

    cfg = merge_config(config)
    hours = demand_matrix.shape[1]
    shifts = list(patterns.keys())
    patterns_unpacked = {
        s: np.unpackbits(p.reshape(7, -1), axis=1)[:, :hours]
        for s, p in patterns.items()
    }

    prob = pl.LpProblem("schedule_with_pulp", pl.LpMinimize)
    max_per_shift = int(demand_matrix.max() * cfg["agent_limit_factor"])

    shift_vars = {
        s: pl.LpVariable(f"shift_{i}", lowBound=0, upBound=max_per_shift, cat=pl.LpInteger)
        for i, s in enumerate(shifts)
    }
    deficit = {
        (d, h): pl.LpVariable(f"def_{d}_{h}", lowBound=0, cat=pl.LpContinuous)
        for d in range(7)
        for h in range(hours)
    }
    excess = {
        (d, h): pl.LpVariable(f"exc_{d}_{h}", lowBound=0, cat=pl.LpContinuous)
        for d in range(7)
        for h in range(hours)
    }

    total_agents = pl.lpSum(shift_vars.values())
    total_deficit = pl.lpSum(deficit.values())
    total_excess = pl.lpSum(excess.values())
    prob += total_agents + 1000 * total_deficit + cfg["excess_penalty"] * total_excess

    for d in range(7):
        for h in range(hours):
            coverage_expr = pl.lpSum(
                shift_vars[s] * patterns_unpacked[s][d, h] for s in shifts
            )
            demand = demand_matrix[d, h]
            prob += coverage_expr + deficit[(d, h)] >= demand
            prob += coverage_expr - excess[(d, h)] <= demand

    solver = pl.PULP_CBC_CMD(msg=0, timeLimit=cfg["TIME_SOLVER"])
    status = prob.solve(solver)

    assignments = {
        s: int(pl.value(v))
        for s, v in shift_vars.items()
        if pl.value(v) and pl.value(v) > 0.5
    }

    coverage = np.zeros_like(demand_matrix, dtype=float)
    for s, count in assignments.items():
        coverage += patterns_unpacked[s] * count

    total_agents_val = int(sum(assignments.values()))
    solver_status = pl.LpStatus.get(prob.status, str(prob.status))
    return {
        "assignments": assignments,
        "coverage_matrix": coverage,
        "total_agents": total_agents_val,
        "status": solver_status,
    }


def solve_in_chunks_optimized(shifts_coverage, demand_matrix, base_chunk_size=10000, *, cfg=None, demand_packed=None):
    """Solve batches sorted by score.

    Based on ``solve_in_chunks_optimized`` from ``legacy/app1.py``.
    """
    scored = []
    seen = set()
    if demand_packed is None:
        demand_packed = np.packbits(demand_matrix > 0, axis=1).astype(np.uint8)
    for name, pat in shifts_coverage.items():
        key = hashlib.md5(pat).digest()
        if key in seen:
            continue
        seen.add(key)
        scored.append((name, pat, score_pattern(pat, demand_packed)))
    scored.sort(key=lambda x: x[2], reverse=True)
    assignments_total = {}
    coverage = np.zeros_like(demand_matrix)
    idx = 0
    while idx < len(scored):
        chunk_size = adaptive_chunk_size(base_chunk_size)
        chunk_dict = {n: p for n, p, _ in scored[idx:idx + chunk_size]}
        remaining = np.maximum(0, demand_matrix - coverage)
        if not np.any(remaining):
            break
        print("\U0001F3AF [OPTIMIZER] Llamando optimize_with_precision_targeting...")
        assigns, _ = optimize_with_precision_targeting(chunk_dict, remaining, cfg=cfg)
        print("\u2705 [OPTIMIZER] optimize_with_precision_targeting completada")
        for n, val in assigns.items():
            assignments_total[n] = assignments_total.get(n, 0) + val
            pat_matrix = np.unpackbits(chunk_dict[n].reshape(7, -1), axis=1)[:, :demand_matrix.shape[1]]
            coverage += pat_matrix * val
        idx += chunk_size
        gc.collect()
        emergency_cleanup()
        if not np.any(np.maximum(0, demand_matrix - coverage)):
            break
    return assignments_total, coverage


def analyze_results(assignments, shifts_coverage, demand_matrix, coverage_matrix=None):
    """Compute coverage metrics from solved assignments."""
    if not assignments:
        return None

    compute_coverage = coverage_matrix is None
    if compute_coverage:
        slots_per_day = (
            (len(next(iter(shifts_coverage.values()))) // 7) * 8
            if shifts_coverage
            else 24
        )
        coverage_matrix = np.zeros((7, slots_per_day), dtype=np.int16)
    else:
        slots_per_day = coverage_matrix.shape[1]

    total_agents = 0
    ft_agents = 0
    pt_agents = 0
    for shift_name, count in assignments.items():
        total_agents += count
        if shift_name.startswith('FT'):
            ft_agents += count
        else:
            pt_agents += count
        if compute_coverage:
            weekly_pattern = shifts_coverage[shift_name]
            pattern_matrix = np.unpackbits(
                weekly_pattern.reshape(7, -1), axis=1
            )[:, :slots_per_day]
            coverage_matrix += pattern_matrix * count

    total_demand = demand_matrix.sum()
    total_covered = np.minimum(coverage_matrix, demand_matrix).sum()
    coverage_percentage = (
        (total_covered / total_demand) * 100 if total_demand > 0 else 0
    )
    diff_matrix = coverage_matrix - demand_matrix
    overstaffing = np.sum(diff_matrix[diff_matrix > 0])
    understaffing = np.sum(np.abs(diff_matrix[diff_matrix < 0]))
    return {
        'total_coverage': coverage_matrix,
        'total_agents': total_agents,
        'ft_agents': ft_agents,
        'pt_agents': pt_agents,
        'coverage_percentage': coverage_percentage,
        'overstaffing': overstaffing,
        'understaffing': understaffing,
        'diff_matrix': diff_matrix,
    }


def _extract_start_hour(name: str) -> float:
    """Best-effort extraction of the start hour from a shift name."""
    for part in name.split('_'):
        if '.' in part and part.replace('.', '').isdigit():
            try:
                return float(part)
            except ValueError:
                continue
    return 0.0


def export_detailed_schedule(assignments, shifts_coverage):
    """Return an Excel workbook detailing the generated schedule."""
    if not assignments:
        return None
    detailed_data = []
    agent_id = 1
    for shift_name, count in assignments.items():
        weekly_pattern = shifts_coverage[shift_name]
        slots_per_day = (len(weekly_pattern) // 7) * 8
        pattern_matrix = np.unpackbits(
            weekly_pattern.reshape(7, -1), axis=1
        )[:, :slots_per_day]
        parts = shift_name.split('_')
        start_hour = _extract_start_hour(shift_name)
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
                    if shift_name.startswith('PT') or shift_name.startswith('FT10p8'):
                        end_idx = (int(work_hours[-1]) + 1) % 24
                        next_day = end_idx <= start_idx
                        horario = f"{start_idx:02d}:00-{end_idx:02d}:00" + ("+1" if next_day else "")
                    else:
                        end_idx = int(work_hours[-1]) + 1
                        next_day = end_idx <= start_idx
                        horario = f"{start_idx:02d}:00-{end_idx % 24:02d}:00" + ("+1" if next_day else "")
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
                        'Tipo': shift_type,
                    })
                else:
                    detailed_data.append({
                        'Agente': f"AGT_{agent_id:03d}",
                        'Dia': ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo'][day],
                        'Horario': "DSO",
                        'Break': "",
                        'Turno': shift_name,
                        'Tipo': 'DSO',
                    })
            agent_id += 1
    df_detailed = pd.DataFrame(detailed_data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_detailed.to_excel(writer, sheet_name='Horarios_Semanales', index=False)
        df_summary = df_detailed.groupby(['Agente', 'Turno']).size().reset_index(name='Dias_Trabajo')
        df_summary.to_excel(writer, sheet_name='Resumen_Agentes', index=False)
        df_shifts = pd.DataFrame([
            {'Turno': shift, 'Agentes': count} for shift, count in assignments.items()
        ])
        df_shifts.to_excel(writer, sheet_name='Turnos_Asignados', index=False)
    return output.getvalue()


def run_complete_optimization(file_stream, config=None):
    """Run the full optimization pipeline and return serialized results.

    The logic mirrors the interactive workflow in ``legacy/app1.py``.
    """
    print("\U0001F50D [SCHEDULER] Iniciando run_complete_optimization")
    print(f"\U0001F50D [SCHEDULER] Config recibido: {config}")

    try:
        cfg = apply_configuration(config)

        print("\U0001F4D6 [SCHEDULER] Leyendo archivo Excel...")
        df = pd.read_excel(file_stream)
        print("\u2705 [SCHEDULER] Archivo Excel leído correctamente")

        print("\U0001F4CA [SCHEDULER] Procesando matriz de demanda...")
        demand_matrix = load_demand_matrix_from_df(df)
        analysis = analyze_demand_matrix(demand_matrix)
        demand_packed = np.packbits(demand_matrix > 0, axis=1).astype(np.uint8)
        CONTEXT["demand_packed"] = demand_packed
        print("\u2705 [SCHEDULER] Matriz de demanda procesada")

        print("\U0001F501 [SCHEDULER] Generando patrones de turnos...")
        patterns = {}
        if cfg.get("optimization_profile") == "JEAN Personalizado":
            slot_minutes = int(cfg.get("slot_duration_minutes", 30))
            start_hours = [h for h in np.arange(0, 24, slot_minutes / 60) if h <= 23.5]
            patterns = load_shift_patterns(
                cfg,
                start_hours=start_hours,
                break_from_start=cfg.get(
                    "break_from_start", DEFAULT_CONFIG["break_from_start"]
                ),
                break_from_end=cfg.get(
                    "break_from_end", DEFAULT_CONFIG["break_from_end"]
                ),
                slot_duration_minutes=slot_minutes,
                demand_matrix=demand_matrix,
                keep_percentage=cfg.get("keep_percentage", 0.3),
                peak_bonus=cfg.get("peak_bonus", 1.5),
                critical_bonus=cfg.get("critical_bonus", 2.0),
                efficiency_bonus=cfg.get("efficiency_bonus", 1.0),
                max_patterns=cfg.get("max_patterns"),
            )
            if not cfg.get("use_ft", True):
                patterns = {k: v for k, v in patterns.items() if not k.startswith("FT")}
            if not cfg.get("use_pt", True):
                patterns = {k: v for k, v in patterns.items() if not k.startswith("PT")}
        else:
            for batch in generate_shifts_coverage_optimized(
                demand_matrix,
                max_patterns=cfg.get("max_patterns"),
                batch_size=cfg.get("batch_size", 2000),
                quality_threshold=cfg.get("quality_threshold", 0),
                cfg=cfg,
                demand_packed=demand_packed,
            ):
                patterns.update(batch)
                if cfg.get("max_patterns") and len(patterns) >= cfg["max_patterns"]:
                    break
        print("[SCHEDULER] Patrones generados")

        print("[SCHEDULER] Iniciando optimizacion...")

        if PULP_AVAILABLE:
            print("[OPTIMIZER] Resolviendo con PuLP (CBC)…")
            pulp_out = solve_with_pulp(demand_matrix, patterns, cfg)
            assignments = pulp_out["assignments"]
            coverage_matrix = pulp_out["coverage_matrix"]
            status = pulp_out["status"]
            total_agents = pulp_out["total_agents"]
        else:
            print("[OPTIMIZER] PuLP no disponible, usando greedy")
            assignments, coverage_matrix = solve_in_chunks_optimized(
                patterns,
                demand_matrix,
                base_chunk_size=cfg.get("base_chunk_size", 10000),
                cfg=cfg,
                demand_packed=demand_packed,
            )
            status = "GREEDY"
            total_agents = sum(assignments.values())

        print(f"[OPTIMIZER] Status: {status}")
        print(f"[OPTIMIZER] Total agents: {total_agents}")
        print("\u2705 [SCHEDULER] Optimización completada")

        metrics = analyze_results(assignments, patterns, demand_matrix, coverage_matrix)
        excel_bytes = export_detailed_schedule(assignments, patterns)

        heatmaps = {}
        if metrics:
            maps = generate_all_heatmaps(
                demand_matrix,
                metrics.get("total_coverage"),
                metrics.get("diff_matrix"),
            )
        else:
            maps = generate_all_heatmaps(demand_matrix)
        for key, fig in maps.items():
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(tmp.name, format="png", bbox_inches="tight")
            plt.close(fig)
            tmp.flush()
            tmp.close()
            heatmaps[key] = tmp.name

        print("\U0001F4E4 [SCHEDULER] Preparando resultados...")

        def _convert(obj):
            """Recursively convert numpy arrays within ``obj`` to Python lists."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj

        result = {
            "analysis": _convert(analysis),
            "assignments": assignments,
            "metrics": _convert(metrics),
            "heatmaps": heatmaps,
        }
        result["effective_config"] = _convert(cfg)
        print("\u2705 [SCHEDULER] Resultados preparados - RETORNANDO")
        return result, excel_bytes

    except Exception as e:
        print(f"\u274C [SCHEDULER] ERROR CRÍTICO: {str(e)}")
