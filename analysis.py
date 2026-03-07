"""Módulo de análise de dados e cálculo de KPIs."""
import pandas as pd
from typing import Optional, List, Tuple
import numpy as np

from utils import (
    evaluate_market,
    tournament_label,
    MARKET_TO_ODDS_COLS,
)
from analysis_service import compute_hit_columns


def compute_acc2(ok_mask: pd.Series, bad_mask: pd.Series) -> Tuple[float, int, int]:
    """Calcula a acurácia, o número de acertos e o total de itens avaliados."""
    total = int((ok_mask | bad_mask).sum())
    correct = int(ok_mask.sum())
    acc = (correct / total * 100.0) if total > 0 else np.nan
    return acc, correct, total


def _metric_from_hits(hit_series: pd.Series, metric_name: str) -> dict:
    """Calcula estatísticas de acerto a partir de série numérica (1/0/NaN)."""
    correct_mask = hit_series == 1.0
    wrong_mask = hit_series == 0.0
    acc, correct, total = compute_acc2(correct_mask, wrong_mask)
    return {
        "Métrica": metric_name,
        "Acerto (%)": 0 if np.isnan(acc) else round(acc, 1),
        "Acertos": correct,
        "Total Avaliado": total,
    }


def calculate_kpis_for_model(sub: pd.DataFrame, model_name: Optional[str] = None) -> List[dict]:
    """Calcula métricas KPI para um subconjunto (modelo único ou consolidado)."""

    eval_df = compute_hit_columns(sub)
    goal_codes = eval_df.get("goal_bet_suggestion", pd.Series(index=eval_df.index, dtype="object")).astype(str).str.lower()
    no_btts_goal_mask = ~goal_codes.isin(["btts_yes", "btts_no"])

    rows = [
        _metric_from_hits(eval_df["hit_result"], "Resultado"),
        _metric_from_hits(eval_df["hit_bet"], "Sugestão de Aposta"),
        _metric_from_hits(eval_df["hit_combo"], "Sugestão Combo"),
        _metric_from_hits(eval_df["hit_score"], "Placar Previsto"),
        _metric_from_hits(eval_df["hit_goal"].where(no_btts_goal_mask), "Sugestão de Gols"),
        _metric_from_hits(eval_df["hit_btts"], "Ambos Marcam"),
    ]

    if model_name:
        for row in rows:
            row["Modelo"] = model_name

    return rows


def calculate_kpis(df_fin: pd.DataFrame, multi_model: bool) -> pd.DataFrame:
    """
    Calcula os KPIs de acurácia para os modelos de previsão.
    """
    if multi_model:
        rows = []
        selected_models = list(df_fin["model"].dropna().unique())
        for m in selected_models:
            sub = df_fin[df_fin["model"] == m]
            if sub.empty:
                continue

            rows.extend(calculate_kpis_for_model(sub, m))

        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Modelo","Métrica","Acerto (%)","Acertos","Total Avaliado"])

    else:
        rows = calculate_kpis_for_model(df_fin)
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Métrica","Acerto (%)","Acertos","Total Avaliado"])


def prepare_accuracy_chart_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara os dados para o gráfico de acurácia diária."""
    if df.empty or 'date' not in df.columns or 'tournament_id' not in df.columns or 'model' not in df.columns:
        return pd.DataFrame()

    df_eval = compute_hit_columns(df)
    hit_cols = ['hit_result', 'hit_bet', 'hit_goal', 'hit_btts', 'hit_combo', 'hit_score']

    id_vars = ['date', 'tournament_id', 'model']
    df_melted = df_eval.melt(id_vars=id_vars, value_vars=hit_cols, var_name='mercado', value_name='acerto')
    df_melted = df_melted.dropna(subset=['acerto'])
    if df_melted.empty:
        return pd.DataFrame(columns=['Data', 'Campeonato', 'Modelo', 'Métrica', 'Taxa de Acerto (%)'])

    df_melted['day'] = df_melted['date'].dt.date
    agg = df_melted.groupby(['day', 'tournament_id', 'model', 'mercado']).agg(
        total_acertos=('acerto', 'sum'),
        total_jogos=('acerto', 'count')
    ).reset_index()

    agg['taxa_acerto'] = (agg['total_acertos'] / agg['total_jogos'] * 100).round(2)
    agg['mercado'] = agg['mercado'].map({
        'hit_result': 'Resultado Final',
        'hit_bet': 'Sugestão de Aposta',
        'hit_goal': 'Sugestão de Gols',
        'hit_btts': 'Ambos Marcam',
        'hit_combo': 'Sugestão Combo',
        'hit_score': 'Placar Exato'
    })
    agg['tournament_id'] = agg['tournament_id'].apply(tournament_label)

    df_final = agg.rename(columns={
        'day': 'Data',
        'tournament_id': 'Campeonato',
        'model': 'Modelo',
        'mercado': 'Métrica',
        'taxa_acerto': 'Taxa de Acerto (%)'
    })
    df_final['Data'] = pd.to_datetime(df_final['Data'])

    return df_final[['Data', 'Campeonato', 'Modelo', 'Métrica', 'Taxa de Acerto (%)']]


def get_best_model_by_market(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o melhor modelo por campeonato e mercado de aposta com base na taxa de acerto.
    """
    if df.empty or "model" not in df.columns or "tournament_id" not in df.columns:
        return pd.DataFrame()

    df_eval = compute_hit_columns(df)

    eval_map = {
        "hit_result": "Resultado Final",
        "hit_bet": "Sugestão de Aposta",
        "hit_goal": "Sugestão de Gols",
        "hit_btts": "Ambos Marcam",
    }
    value_vars = [c for c in eval_map.keys() if c in df_eval.columns]
    if not value_vars:
        return pd.DataFrame()

    # 2) Reestrutura e descarta mercados sem avaliação
    df_melted = df_eval.melt(
        id_vars=["tournament_id", "model"],
        value_vars=value_vars,
        var_name="mercado",
        value_name="acerto",
    ).dropna(subset=["acerto"])

    if df_melted.empty:
        return pd.DataFrame()

    # 3) Agrupa para calcular acertos e volume
    agg = df_melted.groupby(["tournament_id", "model", "mercado"], as_index=False).agg(
        total_acertos=("acerto", "sum"),
        total_jogos=("acerto", "count"),
    )
    agg = agg[agg["total_jogos"] > 0]
    if agg.empty:
        return pd.DataFrame()

    agg["taxa_acerto"] = (agg["total_acertos"] / agg["total_jogos"] * 100).round(2)

    # 4) Define o campeão por mercado e campeonato (maior taxa, depois volume)
    ordered = agg.sort_values(
        by=["tournament_id", "mercado", "taxa_acerto", "total_jogos"],
        ascending=[True, True, False, False],
    )
    df_best = ordered.drop_duplicates(subset=["tournament_id", "mercado"], keep="first").copy()

    # 5) Formata o DataFrame final
    df_best["mercado"] = df_best["mercado"].map(eval_map)
    df_best["tournament_id"] = df_best["tournament_id"].apply(tournament_label)

    df_final = df_best[
        ["tournament_id", "mercado", "model", "taxa_acerto", "total_jogos"]
    ].rename(
        columns={
            "tournament_id": "Campeonato",
            "mercado": "Mercado de Aposta",
            "model": "Melhor Modelo",
            "taxa_acerto": "Taxa de Acerto (%)",
            "total_jogos": "Total de Jogos Avaliados",
        }
    )

    return df_final.sort_values(by=["Campeonato", "Mercado de Aposta"]).reset_index(drop=True)

def create_summary_pivot_table(best_model_df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa o resumo por **campeonato e modelo**, mantendo o recorte de mercado.

    A tabela final destaca para cada combinação (Campeonato, Modelo) a
    cobertura dos mercados, taxa média de acerto e volume avaliado para
    facilitar a leitura segmentada por status.
    """

    if best_model_df.empty:
        return pd.DataFrame()

    df_copy = best_model_df.copy()

    # Consolida as métricas por Campeonato + Modelo, preservando a lista de mercados
    grouped = df_copy.groupby(["Campeonato", "Melhor Modelo"])

    summary = grouped.agg(
        {
            "Mercado de Aposta": lambda s: ", ".join(sorted(set(map(str, s)))),
            "Taxa de Acerto (%)": "mean",
            "Total de Jogos Avaliados": "sum",
        }
    ).reset_index()

    summary = summary.rename(
        columns={
            "Melhor Modelo": "Modelo",
            "Mercado de Aposta": "Mercados de Aposta",
            "Taxa de Acerto (%)": "Taxa média (%)",
            "Total de Jogos Avaliados": "Jogos avaliados",
        }
    )

    summary["Taxa média (%)"] = summary["Taxa média (%)"].round(2)

    # Ordena para leitura consistente por campeonato e performance média
    summary = summary.sort_values(
        by=["Campeonato", "Taxa média (%)"], ascending=[True, False]
    )

    return summary[[
        "Campeonato",
        "Modelo",
        "Mercados de Aposta",
        "Taxa média (%)",
        "Jogos avaliados",
    ]]

def find_best_bet(row, prob_min: float, odd_min: float, markets_to_search: Optional[List[str]] = None) -> pd.Series:
    """Encontra a melhor aposta para uma linha de dados, considerando os mercados especificados."""
    best_bet, max_prob = None, -1.0

    # Se nenhum mercado for especificado, busca em todos os mercados definidos.
    if markets_to_search is None:
        markets_to_search = list(MARKET_TO_ODDS_COLS.keys())

    for market in markets_to_search:
        if market in MARKET_TO_ODDS_COLS:
            prob_col, odd_col = MARKET_TO_ODDS_COLS[market]
            if prob_col in row and odd_col in row and pd.notna(row[prob_col]) and pd.notna(row[odd_col]):
                prob, odd = row[prob_col], row[odd_col]
                if prob >= prob_min and odd >= odd_min and prob > max_prob:
                    max_prob = prob
                    best_bet = {"market": market, "prob": prob, "odd": odd}

    if best_bet:
        return pd.Series(best_bet)

    return pd.Series({"market": np.nan, "prob": np.nan, "odd": np.nan})

def suggest_btts(row, prob_min: float, odd_min: float) -> pd.Series:
    """
    Sugere a melhor aposta 'Ambos Marcam' se ela atender aos critérios de probabilidade e odd.
    """
    prob_yes = row.get("prob_btts_yes", -1)
    prob_no = row.get("prob_btts_no", -1)

    # Define a aposta com maior probabilidade
    if prob_yes > prob_no:
        market = "btts_yes"
        prob = prob_yes
        odd = row.get("odds_btts_yes")
    else:
        market = "btts_no"
        prob = prob_no
        odd = row.get("odds_btts_no")

    # Retorna NaN se a melhor opção não atender aos critérios
    if pd.isna(prob) or pd.isna(odd) or not (prob >= prob_min and odd >= odd_min):
        return pd.Series({
            "btts_sugg_market": np.nan,
            "btts_sugg_prob": np.nan,
            "btts_sugg_odd": np.nan
        })

    return pd.Series({
        "btts_sugg_market": market,
        "btts_sugg_prob": prob,
        "btts_sugg_odd": odd
    })
