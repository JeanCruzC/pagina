
import datetime as dt
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns

# Establish a default theme so any seaborn-based charts elsewhere in the app use
# a consistent style.  The heavy lifting in this module still relies on Plotly,
# but importing seaborn here ensures the dependency is available and configured.
sns.set_theme(style="whitegrid")


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and derive helper fields."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(
        columns={
            "planif. contactos": "planificados",
            "planif contactos": "planificados",
            "contactos": "reales",
            "tramo": "intervalo",
        }
    )
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["intervalo"] = pd.to_datetime(df["intervalo"], format="%H:%M:%S").dt.time
    df["dia_semana"] = df["fecha"].dt.day_name()
    df["semana_iso"] = df["fecha"].dt.isocalendar().week
    df["mes"] = df["fecha"].dt.month
    df["desvio"] = df["reales"] - df["planificados"]
    df["desvio_%"] = df["desvio"] / df["planificados"].replace(0, np.nan) * 100
    df["_dt"] = df.apply(lambda r: dt.datetime.combine(r["fecha"], r["intervalo"]), axis=1)
    return df


def process_timeseries(
    df: pd.DataFrame,
    weight_last: float = 0.7,
    weight_prev: float = 0.3,
    scope: str = "Total",
    view: str = "Día",
) -> Dict[str, Any]:
    """Compute metrics, tables and figures for the time series demo."""

    df = _preprocess(df)

    serie_continua = df.groupby("_dt")[["planificados", "reales"]].sum().sort_index()
    ultima_sem = int(df["semana_iso"].max())
    df_last = df[df["semana_iso"] == ultima_sem]
    serie_last = (
        df_last.groupby("_dt")[["planificados", "reales"]].sum().sort_index()
        if not df_last.empty
        else pd.DataFrame(columns=["planificados", "reales"])
    )

    # KPIs
    y_t_all, y_p_all = serie_continua["reales"], serie_continua["planificados"]
    mae_all = mean_absolute_error(y_t_all, y_p_all)
    rmse_all = np.sqrt(mean_squared_error(y_t_all, y_p_all))
    mape_all = np.mean(np.abs((y_t_all - y_p_all) / y_t_all.replace(0, np.nan))) * 100

    if not serie_last.empty:
        y_t_w, y_p_w = serie_last["reales"], serie_last["planificados"]
        mae_w = mean_absolute_error(y_t_w, y_p_w)
        rmse_w = np.sqrt(mean_squared_error(y_t_w, y_p_w))
        mape_w = np.mean(np.abs((y_t_w - y_p_w) / y_t_w.replace(0, np.nan))) * 100
    else:
        mae_w = rmse_w = mape_w = 0.0

    metrics = {
        "MAE_total": round(mae_all, 2),
        "RMSE_total": round(rmse_all, 2),
        "MAPE_total": round(mape_all, 2),
        "MAE_week": round(mae_w, 2),
        "RMSE_week": round(rmse_w, 2),
        "MAPE_week": round(mape_w, 2),
    }

    if mape_all > 20:
        recommendation = "MAPE global >20%: revisar intervalos con mayor desviación."
    elif mape_w > mape_all:
        recommendation = (
            f"MAPE semana ({mape_w:.2f}%) > global ({mape_all:.2f}%). Investigar cambios."
        )
    else:
        recommendation = "Buen alineamiento planificado vs real."

    weekly = df.groupby("semana_iso")[["planificados", "reales"]].sum().reset_index()
    weekly["desvio_abs"] = weekly["reales"] - weekly["planificados"]
    weekly["desvio_pct"] = (
        weekly["desvio_abs"] / weekly["planificados"].replace(0, np.nan) * 100
    )
    weekly_table = weekly.to_dict("records")

    errors_source = serie_continua if scope == "Total" else serie_last
    errors = errors_source.copy()
    errors["error_abs"] = (errors["reales"] - errors["planificados"]).abs()
    errors["MAPE"] = errors["error_abs"] / errors["planificados"].replace(0, np.nan) * 100
    errors_table = (
        errors.reset_index()[["_dt", "planificados", "reales", "MAPE"]]
        .sort_values("MAPE", ascending=False)
        .head(10)
        .to_dict("records")
    )

    low, high = df["desvio_%"].quantile([0.05, 0.95])
    fig_heat = px.density_heatmap(
        df,
        x="dia_semana",
        y="intervalo",
        z="desvio_%",
        animation_frame="semana_iso",
        color_continuous_scale="RdBu_r",
        range_color=(low, high),
    )

    if view == "Semana":
        df_view = (
            df.groupby("semana_iso")[["planificados", "reales"]].sum().reset_index()
        )
        fig_inter = px.line(
            df_view,
            x="semana_iso",
            y=["planificados", "reales"],
            labels={"value": "Volumen", "variable": "Tipo"},
        )
    elif view == "Mes":
        df_view = df.groupby("mes")[["planificados", "reales"]].sum().reset_index()
        fig_inter = px.line(
            df_view,
            x="mes",
            y=["planificados", "reales"],
            labels={"value": "Volumen", "variable": "Tipo"},
        )
    else:
        fig_inter = px.line(
            serie_continua.reset_index(),
            x="_dt",
            y=["planificados", "reales"],
            labels={"_dt": "Fecha y Hora", "value": "Volumen", "variable": "Tipo"},
        )
        fig_inter.update_xaxes(rangeslider_visible=True)

    return {
        "metrics": metrics,
        "recommendation": recommendation,
        "weekly_table": weekly_table,
        "errors_table": errors_table,
        "heatmap": fig_heat,
        "interactive": fig_inter,
    }
