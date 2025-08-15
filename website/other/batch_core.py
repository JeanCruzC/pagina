"""Batch processing utilities for Erlang calculations."""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from ..services.erlang import (
    service_level_erlang_c,
    waiting_time_erlang_c,
    agents_for_sla,
    chat_sla,
    chat_asa,
    chat_agents_for_sla,
)
from .erlang_core import occupancy_erlang_c


def process_batch_row(
    row: pd.Series,
    sl_target: float,
    awt: float,
    interval_seconds: int,
    default_channel: str,
) -> Dict[str, Any]:
    """Compute metrics for a single batch row."""
    seconds = row.get("Intervalo_Segundos", interval_seconds)
    channel = str(row.get("Tipo_Canal", default_channel)).strip().title()
    if channel not in ["Llamadas", "Chat"]:
        channel = "Llamadas"
    arrival_rate = row["Contactos"] / seconds
    if channel == "Chat":
        aht_list = [row["AHT"]]
        sl = chat_sla(arrival_rate, aht_list, row["Agentes_Actuales"], awt)
        asa = chat_asa(arrival_rate, aht_list, row["Agentes_Actuales"])
        occupancy = occupancy_erlang_c(arrival_rate, row["AHT"], row["Agentes_Actuales"])
        agents_req = chat_agents_for_sla(sl_target, arrival_rate, aht_list, awt)
        agents_req_ceil = int(np.ceil(agents_req))
        sl_req = sl_target
        asa_req = chat_asa(arrival_rate, aht_list, agents_req_ceil)
        occupancy_req = occupancy_erlang_c(arrival_rate, row["AHT"], agents_req_ceil)
    else:
        sl = service_level_erlang_c(arrival_rate, row["AHT"], row["Agentes_Actuales"], awt)
        asa = waiting_time_erlang_c(arrival_rate, row["AHT"], row["Agentes_Actuales"])
        occupancy = occupancy_erlang_c(arrival_rate, row["AHT"], row["Agentes_Actuales"])
        agents_req = agents_for_sla(sl_target, arrival_rate, row["AHT"], awt)
        agents_req_ceil = int(np.ceil(agents_req))
        sl_req = sl_target
        asa_req = waiting_time_erlang_c(arrival_rate, row["AHT"], agents_req_ceil)
        occupancy_req = occupancy_erlang_c(arrival_rate, row["AHT"], agents_req_ceil)
    diff_agents = agents_req - row["Agentes_Actuales"]
    return {
        "SL": sl,
        "ASA": asa,
        "Ocupacion": occupancy,
        "Agentes_Requeridos": agents_req,
        "SL_Requerido": sl_req,
        "ASA_Requerido": asa_req,
        "Ocupacion_Requerido": occupancy_req,
        "Diferencia_Agentes": diff_agents,
        "Tipo_Canal": channel,
    }


def export_results(df_processed: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with status and recommendation columns."""
    df_exp = df_processed.copy()
    df_exp["Status_SL"] = np.where(df_exp["SL"] >= df_exp["SL_Requerido"], "OK", "BAJO")
    df_exp["Recomendacion"] = np.where(
        df_exp["Diferencia_Agentes"] > 0,
        "Agregar " + df_exp["Diferencia_Agentes"].astype(str),
        "Mantener",
    )
    return df_exp


__all__ = ["process_batch_row", "export_results"]
