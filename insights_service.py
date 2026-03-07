"""Serviços para agregação de insights/KPIs exibidos no dashboard."""

from __future__ import annotations

import pandas as pd

from analysis import calculate_kpis
from utils import tournament_label

METRIC_ORDER = [
    "Resultado",
    "Sugestão de Aposta",
    "Sugestão Combo",
    "Sugestão de Gols",
    "Ambos Marcam",
]


def metric_stats_for(metrics_frame: pd.DataFrame, metric_order: list[str] | None = None) -> dict[str, tuple[float, int, int]]:
    """Extrai (acurácia, acertos, total) por métrica de um dataframe de KPIs."""

    order = metric_order or METRIC_ORDER

    def _extract(metric: str) -> tuple[float, int, int]:
        row = metrics_frame[metrics_frame["Métrica"] == metric]
        if row.empty:
            return 0.0, 0, 0
        acc = float(row["Acerto (%)"].iloc[0]) if pd.notna(row["Acerto (%)"].iloc[0]) else 0.0
        hits = int(row["Acertos"].iloc[0]) if pd.notna(row["Acertos"].iloc[0]) else 0
        total = int(row["Total Avaliado"].iloc[0]) if pd.notna(row["Total Avaliado"].iloc[0]) else 0
        return acc, hits, total

    return {metric: _extract(metric) for metric in order}


def build_tournament_stats(df_fin: pd.DataFrame, metric_order: list[str] | None = None) -> list[tuple[str, dict[str, tuple[float, int, int]]]]:
    """Gera estatísticas de KPI por campeonato para jogos finalizados."""

    order = metric_order or METRIC_ORDER
    results: list[tuple[str, dict[str, tuple[float, int, int]]]] = []

    if "tournament_id" not in df_fin.columns:
        return results

    for tourn_id in sorted(df_fin["tournament_id"].dropna().unique()):
        tourn_df = df_fin[df_fin["tournament_id"] == tourn_id]
        if tourn_df.empty:
            continue
        tourn_metrics = calculate_kpis(tourn_df, False)
        results.append((tournament_label(tourn_id), metric_stats_for(tourn_metrics, order)))

    return results
