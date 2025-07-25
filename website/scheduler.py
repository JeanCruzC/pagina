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
