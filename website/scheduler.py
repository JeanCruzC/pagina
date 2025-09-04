import io, base64, time, gc, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False


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


def generate_weekly_pattern_simple(start_hour, duration, working_days):
    pattern = np.zeros((7, 24), dtype=np.int8)
    for day in working_days:
        for h in range(duration):
            t = start_hour + h
            d_off, idx = divmod(int(t), 24)
            pattern[(day + d_off) % 7, idx] = 1
    return pattern.flatten()


def generate_weekly_pattern_10h8(
    start_hour,
    working_days,
    eight_hour_day,
    break_from_start=2,
    break_from_end=2,
    break_len=1,
):
    pattern = np.zeros((7, 24), dtype=np.int8)
    for day in working_days:
        duration = 8 if day == eight_hour_day else 10
        for h in range(duration):
            t = start_hour + h
            d_off, idx = divmod(int(t), 24)
            pattern[(day + d_off) % 7, idx] = 1
        b_start = start_hour + break_from_start
        b_end = start_hour + duration - break_from_end
        if int(b_start) < int(b_end):
            b_hour = int(b_start) + (int(b_end) - int(b_start)) // 2
        else:
            b_hour = int(b_start)
        for b in range(int(b_len := break_len)):
            t = b_hour + b
            d_off, idx = divmod(int(t), 24)
            pattern[(day + d_off) % 7, idx] = 0
    return pattern.flatten()


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


def generate_shifts_coverage_corrected(
    *,
    allow_8h=True,
    allow_10h8=False,
    allow_pt_4h=True,
    allow_pt_5h=True,
    allow_pt_6h=False,
    break_from_start=2.0,
    break_from_end=2.0,
    batch_size=2000,
):
    days = list(range(7))
    batch = {}

    def flush():
        nonlocal batch
        if batch:
            yield batch
            batch = {}

    if allow_8h:
        for start in range(8, 21):
            pat = generate_weekly_pattern_simple(start, 8, days[:-1])
            batch[f"FT_8H_{start:02d}"] = pat
            if len(batch) >= batch_size:
                yield from flush()
    if allow_10h8:
        for start in range(8, 19):
            pat = generate_weekly_pattern_10h8(
                start,
                days[:-2],
                eight_hour_day=4,
                break_from_start=break_from_start,
                break_from_end=break_from_end,
            )
            batch[f"FT_10H8_{start:02d}"] = pat
            if len(batch) >= batch_size:
                yield from flush()
    for dur, flag in [(4, allow_pt_4h), (5, allow_pt_5h), (6, allow_pt_6h)]:
        if not flag:
            continue
        for start in range(8, 21 - dur + 1):
            pat = generate_weekly_pattern_simple(start, dur, days[:-1])
            batch[f"PT_{dur}H_{start:02d}"] = pat
            if len(batch) >= batch_size:
                yield from flush()
    yield from flush()


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
    break_from_start=2.0,
    break_from_end=2.0,
    batch_size=2000,
):
    selected = 0
    seen = set()
    for raw in generate_shifts_coverage_corrected(
        allow_8h=allow_8h,
        allow_10h8=allow_10h8,
        allow_pt_4h=allow_pt_4h,
        allow_pt_5h=allow_pt_5h,
        allow_pt_6h=allow_pt_6h,
        break_from_start=break_from_start,
        break_from_end=break_from_end,
        batch_size=batch_size,
    ):
        batch = {}
        for name, pat in raw.items():
            key = hashlib.md5(pat).digest()
            if key in seen:
                continue
            seen.add(key)
            if score_pattern(pat, demand_matrix) >= quality_threshold:
                batch[name] = pat
                selected += 1
                if max_patterns and selected >= max_patterns:
                    break
        if batch:
            yield batch
        if max_patterns and selected >= max_patterns:
            break


def adaptive_chunk_size(base):
    return base


def solve_in_chunks_optimized(shifts_coverage, demand_matrix, base_chunk_size=10000):
    scored = []
    seen = set()
    for name, pat in shifts_coverage.items():
        key = hashlib.md5(pat).digest()
        if key in seen:
            continue
        seen.add(key)
        scored.append((name, pat, score_pattern(pat, demand_matrix)))
    scored.sort(key=lambda x: x[2], reverse=True)

    assignments_total = {}
    coverage = np.zeros_like(demand_matrix)
    idx = 0
    while idx < len(scored):
        chunk_size = adaptive_chunk_size(base_chunk_size)
        chunk = {n: p for n, p, _ in scored[idx : idx + chunk_size]}
        remaining = np.maximum(0, demand_matrix - coverage)
        if not np.any(remaining):
            break
        assigns, _ = optimize_with_precision_targeting(chunk, remaining)
        for n, v in assigns.items():
            assignments_total[n] = assignments_total.get(n, 0) + v
            slots = len(chunk[n]) // 7
            coverage += chunk[n].reshape(7, slots) * v
        idx += chunk_size
        gc.collect()
        if not np.any(np.maximum(0, demand_matrix - coverage)):
            break
    return assignments_total


def optimize_with_precision_targeting(
    shifts_coverage,
    demand_matrix,
    *,
    cfg=None,
    iteration_time_limit=None,
    agent_limit_factor=30,
):
    if not PULP_AVAILABLE:
        return {}, "NO_PULP"
    cfg = cfg or {}
    shifts_list = list(shifts_coverage.keys())
    prob = pulp.LpProblem("PrecisionTargeting", pulp.LpMinimize)
    max_per_shift = max(5, int(demand_matrix.sum() / max(agent_limit_factor, 1)))
    shift_vars = {
        s: pulp.LpVariable(f"x_{i}", 0, max_per_shift, pulp.LpInteger)
        for i, s in enumerate(shifts_list)
    }
    hours = demand_matrix.shape[1]
    deficit_vars = {
        (d, h): pulp.LpVariable(f"def_{d}_{h}", 0, None)
        for d in range(7)
        for h in range(hours)
    }
    excess_vars = {
        (d, h): pulp.LpVariable(f"exc_{d}_{h}", 0, None)
        for d in range(7)
        for h in range(hours)
    }

    total_def = pulp.lpSum(deficit_vars.values())
    total_exc = pulp.lpSum(excess_vars.values())
    total_agents = pulp.lpSum(shift_vars.values())
    prob += total_def + 0.5 * total_exc + 0.001 * total_agents

    for d in range(7):
        for h in range(hours):
            cov = pulp.lpSum(
                [
                    shift_vars[s]
                    * np.array(shifts_coverage[s]).reshape(7, len(shifts_coverage[s]) // 7)[d, h]
                    for s in shifts_list
                ]
            )
            prob += cov + deficit_vars[(d, h)] - excess_vars[(d, h)] == int(demand_matrix[d, h])

    time_limit = iteration_time_limit if iteration_time_limit is not None else cfg.get("solver_time", 240)
    solver = pulp.PULP_CBC_CMD(
        msg=cfg.get("solver_msg", True),
        timeLimit=int(time_limit) if time_limit else None,
        threads=1,
        randomSeed=42,
    )
    prob.solve(solver)

    assignments = {}
    if prob.status == pulp.LpStatusOptimal:
        for s in shifts_list:
            v = int(shift_vars[s].varValue or 0)
            if v > 0:
                assignments[s] = v
        return assignments, "PRECISION_TARGETING"
    return {}, f"STATUS_{prob.status}"


def optimize_ft_then_pt_strategy(shifts_coverage, demand_matrix):
    ft = {k: v for k, v in shifts_coverage.items() if k.startswith("FT")}
    pt = {k: v for k, v in shifts_coverage.items() if k.startswith("PT")}
    ft_ass, _ = optimize_with_precision_targeting(ft, demand_matrix)
    cov_ft = np.zeros_like(demand_matrix)
    for s, c in ft_ass.items():
        slots = len(ft[s]) // 7
        cov_ft += ft[s].reshape(7, slots) * c
    remaining = np.maximum(0, demand_matrix - cov_ft)
    pt_ass, _ = optimize_with_precision_targeting(pt, remaining)
    return {**ft_ass, **pt_ass}, "FT_PT"


def optimize_jean_search(shifts_coverage, demand_matrix, *, cfg=None):
    cfg = cfg or {}
    TIME_SOLVER = cfg.get("solver_time", 120) or None
    MAX_ITERS = int(cfg.get("iterations", 3))
    factor = int(cfg.get("agent_limit_factor", 30))
    best_assignments = {}
    best_score = float("inf")
    iteration = 0
    while iteration < MAX_ITERS and factor >= 1:
        temp_cfg = cfg.copy()
        temp_cfg["agent_limit_factor"] = factor
        assignments, _ = optimize_with_precision_targeting(
            shifts_coverage,
            demand_matrix,
            cfg=temp_cfg,
            iteration_time_limit=TIME_SOLVER,
            agent_limit_factor=factor,
        )
        results = analyze_results(assignments, shifts_coverage, demand_matrix)
        if results:
            score = results["understaffing"] + results["overstaffing"] * 0.3
            if score < best_score or not best_assignments:
                best_score = score
                best_assignments = assignments
        iteration += 1
        factor = max(1, factor // (2 if iteration == 1 else 1))
    return best_assignments, "JEAN"


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
    diff = total_cov - demand_matrix
    over = int(np.sum(diff[diff > 0]))
    under = int(np.sum(np.abs(diff[diff < 0])))
    return {
        "total_coverage": total_cov,
        "coverage_percentage": coverage_percentage,
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
        f"Break configurado: {cfg.get('break_from_start',2.0)}h desde inicio, {cfg.get('break_from_end',2.0)}h antes del fin",
    ]

    pt = []
    if cfg.get("allow_pt_4h"):
        pt.append("4h×6días")
    if cfg.get("allow_pt_6h"):
        pt.append("6h×4días")
    if cfg.get("allow_pt_5h"):
        pt.append("5h×5días")
    ins["pt_habilitados"].append(" / ".join(pt) if pt else "—")

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

def _build_sync_payload(assignments, patterns, demand_matrix, *, day_labels=None, hour_labels=None, meta=None):
    dem = np.asarray(demand_matrix, dtype=int)
    D, H = dem.shape
    cov = _assigned_matrix_from(assignments, patterns, D, H)
    diff = cov - dem

    total_dem = int(dem.sum())
    total_cov = int(cov.sum())
    cap = np.minimum(cov, dem).sum()
    coverage_pure = (cap / total_dem * 100.0) if total_dem > 0 else 0.0
    coverage_real = (total_cov / total_dem * 100.0) if total_dem > 0 else 0.0

    deficit = int(np.clip(dem - cov, 0, None).sum())
    excess = int(np.clip(cov - dem, 0, None).sum())

    fig_dem = _heatmap(dem, "Demanda por Hora y Día", day_labels, hour_labels, annotate=True, cmap="Reds")
    fig_cov = _heatmap(cov, "Cobertura por Hora y Día", day_labels, hour_labels, annotate=True, cmap="Blues")
    fig_diff = _heatmap(diff, "Cobertura - Demanda (exceso/deficit)", day_labels, hour_labels, annotate=True, cmap="RdBu_r")

    return {
        "metrics": {
            "total_demand": total_dem,
            "total_coverage": total_cov,
            "coverage_percentage": round(coverage_pure, 1),
            "coverage_pure": round(coverage_pure, 1),
            "coverage_real": round(coverage_real, 1),
            "deficit": deficit,
            "excess": excess,
            "agents": len(assignments or {}),
        },
        "figures": {
            "demand_png": _fig_to_b64(fig_dem),
            "coverage_png": _fig_to_b64(fig_cov),
            "diff_png": _fig_to_b64(fig_diff),
        },
        "meta": meta or {},
    }


def run_complete_optimization(
    file_stream,
    config=None,
    generate_charts=False,
    job_id=None,
    return_payload=False,
):
    cfg = config or {}
    start_total = time.time()
    TIME_SOLVER = cfg.get("solver_time", 120) or None
    MAX_ITERS = int(cfg.get("iterations", 3))
    df = pd.read_excel(file_stream)
    demand_matrix = load_demand_matrix_from_df(df)
    analysis = _analyze_demand(demand_matrix)

    patterns = {}
    for batch in generate_shifts_coverage_optimized(
        demand_matrix,
        allow_8h=cfg.get("allow_8h", True),
        allow_10h8=cfg.get("allow_10h8", False),
        allow_pt_4h=cfg.get("allow_pt_4h", True),
        allow_pt_5h=cfg.get("allow_pt_5h", True),
        allow_pt_6h=cfg.get("allow_pt_6h", False),
        break_from_start=cfg.get("break_from_start", 2.0),
        break_from_end=cfg.get("break_from_end", 2.0),
    ):
        patterns.update(batch)

    assignments, status = optimize_jean_search(
        patterns,
        demand_matrix,
        cfg={**cfg, "solver_time": TIME_SOLVER, "iterations": MAX_ITERS},
    )
    total_elapsed = time.time() - start_total
    if return_payload:
        D, H = demand_matrix.shape
        day_labels = [f"Día {i+1}" for i in range(D)]
        hour_labels = list(range(H))
        payload = _build_sync_payload(
            assignments=assignments or {},
            patterns=patterns,
            demand_matrix=demand_matrix,
            day_labels=day_labels,
            hour_labels=hour_labels,
            meta={"status": status, "elapsed": round(total_elapsed, 1)},
        )
        payload["status"] = status
        payload["config"] = cfg
        payload["insights"] = _insights_from_analysis(analysis, cfg)
        return payload
    return {"assignments": assignments}
