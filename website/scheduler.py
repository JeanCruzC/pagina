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
    Espera columnas de tu Requerido.xlsx (día/hora/valor) y devuelve matriz DxH.
    7x24 por defecto. Vacíos a 0.
    """
    days = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    H = 24
    mat = np.zeros((7, H), dtype=int)
    cols = {c.lower(): c for c in df.columns}
    day_col = cols.get("dia") or cols.get("día") or list(df.columns)[0]
    hour_col = cols.get("hora") or cols.get("hour") or list(df.columns)[1]
    val_col = (
        cols.get("agentes")
        or cols.get("valor")
        or cols.get("suma de agentes requeridos erlang")
        or list(df.columns)[-1]
    )

    for _, row in df.iterrows():
        d = str(row[day_col]).strip()
        if d not in days:
            continue
        h = int(row[hour_col])
        v = int(row[val_col])
        mat[days.index(d), h] = v
    return mat


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


def optimize_with_precision_targeting(shifts_coverage, demand_matrix):
    if not PULP_AVAILABLE:
        return {}, "NO_PULP"
    shifts_list = list(shifts_coverage.keys())
    prob = pulp.LpProblem("PrecisionTargeting", pulp.LpMinimize)
    max_per_shift = max(5, int(demand_matrix.sum() / 30))
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

    solver = pulp.PULP_CBC_CMD(msg=False)
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


def _fig_to_b64(fig):
    buf = io.BytesIO()
    # Fondo transparente para que se vea bien en tema oscuro
    fig.patch.set_alpha(0)
    for ax in fig.get_axes():
        ax.set_facecolor((0, 0, 0, 0))
    # DPI alto = texto nítido
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _heatmap(matrix, title, day_labels=None, hour_labels=None,
             annotate=True, cmap="RdYlBu_r"):
    M = np.asarray(matrix)
    D, H = M.shape

    # Escala de fuente automática tipo Streamlit
    base = 12 if H <= 24 else 10
    tick_fs = base
    ann_fs = max(8, base - 2)
    title_fs = base + 4

    fig_w = min(1.0 * H, 22)      # ancho máximo ~22in
    fig_h = min(0.65 * D + 2, 12) # alto máximo ~12in
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(M, cmap=cmap, aspect="auto", interpolation="nearest")

    if hour_labels is None:
        hour_labels = [f"{h:02d}" for h in range(H)]
    if day_labels is None:
        day_labels = [f"Día {i+1}" for i in range(D)]

    ax.set_xticks(range(H))
    ax.set_xticklabels(hour_labels, fontsize=tick_fs)
    ax.set_yticks(range(D))
    ax.set_yticklabels(day_labels, fontsize=tick_fs)

    # Estilo legible en fondo oscuro
    ax.tick_params(colors="#E6E6E6")
    for spine in ax.spines.values():
        spine.set_color("#AAAAAA")
        spine.set_linewidth(0.6)

    ax.set_title(title, fontsize=title_fs, color="#FFFFFF", pad=10)
    ax.set_xlabel("Hora del día", fontsize=tick_fs, color="#E6E6E6")
    ax.set_ylabel("Día de la semana", fontsize=tick_fs, color="#E6E6E6")

    # Barra de color con etiquetas grandes
    cbar = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.015)
    cbar.ax.tick_params(labelsize=tick_fs, colors="#E6E6E6")
    cbar.outline.set_edgecolor("#AAAAAA")

    # Anotaciones (números) con color dinámico de contraste
    if annotate and H <= 30 and D <= 14:
        vmax = np.nanmax(M) if M.size else 0
        thr = 0.5 * vmax
        for i in range(D):
            for j in range(H):
                v = M[i, j]
                txt_color = "#111111" if v >= thr else "#FFFFFF"
                ax.text(j, i, f"{int(v)}",
                        ha="center", va="center",
                        fontsize=ann_fs, color=txt_color)

    fig.tight_layout()
    return fig


def _build_sync_payload(assignments, shifts_coverage, demand_matrix):
    res = analyze_results(assignments, shifts_coverage, demand_matrix)
    cov = res["total_coverage"]
    diff = cov - demand_matrix
    fig_dem = _heatmap(demand_matrix, "Demanda por Hora y Día", cmap="Reds")
    fig_cov = _heatmap(cov, "Cobertura por Hora y Día", cmap="Blues")
    fig_diff = _heatmap(diff, "Cobertura - Demanda (exceso/déficit)", cmap="RdBu_r")
    return {
        "metrics": {
            "agents": res["total_agents"],
            "coverage_percentage": round(res["coverage_percentage"], 1),
            "excess": res["overstaffing"],
            "deficit": res["understaffing"],
        },
        "figures": {
            "demand_png": _fig_to_b64(fig_dem),
            "coverage_png": _fig_to_b64(fig_cov),
            "diff_png": _fig_to_b64(fig_diff),
        },
    }


def run_complete_optimization(
    file_stream,
    config=None,
    generate_charts=False,
    job_id=None,
    return_payload=False,
):
    cfg = config or {}
    df = pd.read_excel(file_stream)
    demand_matrix = load_demand_matrix_from_df(df)

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

    assignments = solve_in_chunks_optimized(patterns, demand_matrix)
    if return_payload:
        payload = _build_sync_payload(assignments, patterns, demand_matrix)
        return payload
    return {"assignments": assignments}
