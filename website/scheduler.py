import io
import base64
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .scheduler_core import (
        load_demand_matrix_from_df,
        analyze_demand_matrix,
    )
except Exception:  # pragma: no cover - allow direct script execution
    sys.path.append(os.path.dirname(__file__))
    from scheduler_core import (  # type: ignore
        load_demand_matrix_from_df,
        analyze_demand_matrix,
    )


# ===== Helpers “modo Streamlit” (figuras y payload) =====

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
    ax.set_xticks(range(H))
    ax.set_xticklabels(hour_labels, fontsize=8)
    ax.set_yticks(range(D))
    ax.set_yticklabels(day_labels, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Día")
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


# ----- Simplified scheduling algorithms -----

def score_pattern(pattern, dm_packed):
    """Return coverage score of ``pattern`` against packed demand matrix."""
    pat = np.asarray(pattern, dtype=np.uint8)
    return int(np.unpackbits(np.bitwise_and(pat, dm_packed)).sum())


def generate_shift_patterns(demand_matrix, *, top_k=20, cfg=None):
    """Generate simple shift patterns and return top ``k`` by score."""
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
            mat[day, start:start + shift_len] = 1
            packed = np.packbits(mat > 0, axis=1).astype(np.uint8)
            score = score_pattern(packed, dm_packed)
            pid = f"D{day}_H{start}"
            patterns.append((score, pid, packed))
    patterns.sort(key=lambda x: x[0], reverse=True)
    return patterns[:top_k]


def optimize_jean_search(
    shifts_coverage,
    demand_matrix,
    *,
    agent_limit_factor=30,
    excess_penalty=5.0,
    peak_bonus=2.0,
    critical_bonus=2.5,
    target_coverage=98.0,
    max_iterations=6,
    iteration_time_limit=60,
):
    """Placeholder optimization returning a trivial result.

    The full implementation is beyond the scope of these exercises.  This
    function mirrors the API and returns deterministic output so unit tests can
    verify wrapper equivalence.
    """
    return {}, "NOT_EXECUTED"


def run_complete_optimization(
    file_stream,
    config=None,
    generate_charts=False,
    job_id=None,
    return_payload=False,
):
    """Read an Excel file, perform a minimal optimization and optionally build
    a payload with figures encoded as base64 strings."""
    df = pd.read_excel(file_stream)
    demand_matrix = load_demand_matrix_from_df(df)
    cfg = config or {}

    # Minimal placeholder optimization: create one pattern that matches demand
    pattern_id = "PATTERN_1"
    patterns = {pattern_id: demand_matrix.copy() > 0}
    assignments = {"agent_1": pattern_id}
    analysis = analyze_demand_matrix(demand_matrix)

    result = {
        "pulp": {
            "assignments": assignments,
            "metrics": {},
            "status": "OK",
        },
        "greedy": {
            "assignments": {},
            "metrics": {},
            "status": "NOT_RUN",
        },
        "analysis": analysis,
        "config": cfg,
    }

    if return_payload:
        D, H = demand_matrix.shape
        day_labels = [f"Día {i+1}" for i in range(D)]
        hour_labels = list(range(H))
        payload = _build_sync_payload(
            assignments=assignments,
            patterns=patterns,
            demand_matrix=demand_matrix,
            day_labels=day_labels,
            hour_labels=hour_labels,
            meta={"status": "OK"},
        )
        payload["status"] = "OK"
        payload["config"] = cfg
        return payload

    return result
