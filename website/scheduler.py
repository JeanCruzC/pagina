import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO


def load_demand_excel(file_stream) -> np.ndarray:
    """Read demand Excel file and return a 7x24 numpy array."""
    df = pd.read_excel(file_stream)
    day_col = [c for c in df.columns if "D\u00eda" in c][0]
    demand_col = [c for c in df.columns if "Erlang" in c or "Requeridos" in c][-1]
    dm = df.pivot_table(index=day_col, values=demand_col, columns=df.index % 24, aggfunc='first').fillna(0)
    dm = dm.reindex(range(1,8)).fillna(0)
    dm = dm.sort_index()
    matrix = dm.to_numpy(dtype=int)
    if matrix.shape[1] != 24:
        matrix = np.pad(matrix, ((0,0),(0,24-matrix.shape[1])), constant_values=0)
    return matrix


def generate_schedule(demand_matrix: np.ndarray) -> np.ndarray:
    """Very simple scheduler that returns the demand as coverage."""
    return demand_matrix.copy()


def heatmap(matrix: np.ndarray, title: str) -> BytesIO:
    """Return PNG image with heatmap for given matrix."""
    fig, ax = plt.subplots(figsize=(10,4))
    sns.heatmap(matrix, ax=ax, cmap="viridis", cbar=False)
    ax.set_title(title)
    ax.set_xlabel("Hora")
    ax.set_ylabel("Dia")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_shifts_coverage_corrected(*, max_patterns: int | None = None, batch_size: int | None = None):
    """Return a very small set of dummy weekly patterns."""
    pattern = np.ones((7, 24), dtype=int)
    yield {"GENERIC": pattern.flatten()}


def generate_shifts_coverage_optimized(demand_matrix: np.ndarray, *, max_patterns: int | None = None, batch_size: int = 2000, quality_threshold: int = 0):
    """Simplified wrapper that yields one batch of patterns."""
    for batch in generate_shifts_coverage_corrected(max_patterns=max_patterns, batch_size=batch_size):
        yield batch


def optimize_with_precision_targeting(shifts_coverage, demand_matrix):
    """Return a naive assignment using the first pattern."""
    if not shifts_coverage:
        return {}, "NO_SHIFTS"
    name = next(iter(shifts_coverage))
    return {name: int(demand_matrix.max())}, "NAIVE"


def optimize_ft_then_pt_strategy(shifts_coverage, demand_matrix):
    """Delegate to ``optimize_with_precision_targeting`` in this simplified version."""
    return optimize_with_precision_targeting(shifts_coverage, demand_matrix)


def solve_in_chunks_optimized(shifts_coverage, demand_matrix, base_chunk_size: int = 10000):
    """Return a direct optimization result for all shifts."""
    assigns, _ = optimize_with_precision_targeting(shifts_coverage, demand_matrix)
    return assigns


def analyze_results(assignments, shifts_coverage, demand_matrix):
    """Compute simple coverage statistics for the generated schedule."""
    if not assignments:
        return None
    coverage = np.zeros_like(demand_matrix, dtype=int)
    for name, count in assignments.items():
        pattern = shifts_coverage.get(name)
        if pattern is None:
            continue
        pat_matrix = np.array(pattern).reshape(demand_matrix.shape)
        coverage += pat_matrix * count

    total_coverage = coverage
    diff_matrix = coverage - demand_matrix
    total_agents = sum(assignments.values())
    return {
        "total_coverage": total_coverage,
        "total_agents": total_agents,
        "ft_agents": total_agents,
        "pt_agents": 0,
        "coverage_percentage": float(np.minimum(coverage, demand_matrix).sum()) / max(1, demand_matrix.sum()) * 100,
        "overstaffing": float(diff_matrix[diff_matrix > 0].sum()),
        "understaffing": float(-diff_matrix[diff_matrix < 0].sum()),
        "diff_matrix": diff_matrix,
    }


def export_detailed_schedule(assignments, shifts_coverage):
    """Generate a very small Excel file with the assignments."""
    if not assignments:
        return None
    days = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    frames = []
    for name, count in assignments.items():
        pattern = shifts_coverage.get(name)
        if pattern is None:
            continue
        mat = np.array(pattern).reshape(7, 24)
        for i in range(7):
            for h in range(24):
                if mat[i, h] > 0:
                    for c in range(count):
                        frames.append({"Agente": f"A{c+1}", "Día": days[i], "Hora": f"{h:02d}:00", "Turno": name})
    df = pd.DataFrame(frames)
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()


def load_learning_data():
    try:
        with open("optimization_learning.json", "r") as fh:
            return json.load(fh)
    except Exception:
        return {"executions": []}


def save_learning_data(data):
    try:
        with open("optimization_learning.json", "w") as fh:
            json.dump(data, fh, indent=2)
    except Exception:
        pass


def get_adaptive_params(demand_matrix, target_coverage):
    return {
        "agent_limit_factor": 12,
        "excess_penalty": 0.5,
        "peak_bonus": 1.5,
        "critical_bonus": 2.0,
        "precision_mode": False,
    }


def save_execution_result(demand_matrix, params, coverage, total_agents, execution_time):
    data = load_learning_data()
    data.get("executions", []).append({
        "coverage": coverage,
        "total_agents": total_agents,
        "execution_time": execution_time,
        "params": params,
    })
    save_learning_data(data)
    return True

