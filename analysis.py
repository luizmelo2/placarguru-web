
"""Módulo de análise de dados e cálculo de KPIs."""
import pandas as pd
from typing import Optional, List, Tuple
import numpy as np

from utils import (
    eval_result_pred_row, eval_bet_row, eval_goal_row,
    eval_btts_suggestion_row, eval_sugestao_combo_row, eval_score_pred_row,
    tournament_label, MARKET_TO_ODDS_COLS,
)


def compute_acc2(ok_mask: pd.Series, bad_mask: pd.Series) -> Tuple[float, int, int]:
    """Calcula a acurácia, acertos e total avaliado."""
    total = int((ok_mask | bad_mask).sum())
    correct = int(ok_mask.sum())
    acc = (correct / total * 100.0) if total > 0 else np.nan
    return acc, correct, total


def _calculate_metric(sub: pd.DataFrame, name: str, func, **kwargs) -> dict:
    """Helper genérico para calcular acurácia."""
    eval_series = sub.apply(func, axis=1, **kwargs)
    acc, correct, total = compute_acc2(eval_series, ~eval_series)
    return {
        "Métrica": name,
        "Acerto (%)": round(acc, 1) if not np.isnan(acc) else 0,
        "Acertos": correct,
        "Total Avaliado": total
    }


def _calculate_goal_suggestion_accuracy(sub: pd.DataFrame) -> dict:
    """Calcula a acurácia da 'Sugestão de Gols'."""
    evals = sub.apply(eval_goal_row, axis=1)
    btts_mask = sub.get("goal_bet_suggestion", pd.Series(dtype=str)).str.lower().isin(['btts_yes', 'btts_no'])
    acc, correct, total = compute_acc2(evals & ~btts_mask, ~evals & ~btts_mask)
    return {
        "Métrica": "Sugestão de Gols",
        "Acerto (%)": round(acc, 1) if not np.isnan(acc) else 0,
        "Acertos": correct,
        "Total Avaliado": total
    }


def _calculate_btts_accuracy(sub: pd.DataFrame) -> dict:
    """Calcula a acurácia para 'Ambos Marcam'."""
    evals = sub.apply(eval_btts_suggestion_row, axis=1)
    acc, correct, total = compute_acc2(evals, ~evals)
    return {
        "Métrica": "Ambos Marcam",
        "Acerto (%)": round(acc, 1) if not np.isnan(acc) else 0,
        "Acertos": correct,
        "Total Avaliado": total
    }


def calculate_kpis_for_model(sub: pd.DataFrame, model_name: Optional[str] = None) -> List[dict]:
    """Calcula todos os KPIs para um dataframe."""
    rows = [
        _calculate_metric(sub, "Resultado", eval_result_pred_row),
        _calculate_metric(sub, "Sugestão de Aposta", eval_bet_row),
        _calculate_metric(sub, "Sugestão Combo", eval_sugestao_combo_row),
        _calculate_metric(sub, "Placar Previsto", eval_score_pred_row),
        _calculate_goal_suggestion_accuracy(sub),
        _calculate_btts_accuracy(sub),
    ]
    if model_name:
        for row in rows:
            row["Modelo"] = model_name
    return rows


def calculate_kpis(df_fin: pd.DataFrame, multi_model: bool) -> pd.DataFrame:
    """Calcula os KPIs de acurácia para os modelos."""
    if multi_model:
        rows = [
            kpi for m in df_fin["model"].dropna().unique()
            for kpi in calculate_kpis_for_model(df_fin[df_fin["model"] == m], m)
        ]
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    return pd.DataFrame(calculate_kpis_for_model(df_fin))


def prepare_accuracy_chart_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara dados para o gráfico de acurácia diária."""
    if df.empty or 'date' not in df.columns:
        return pd.DataFrame()

    df_eval = df.copy()
    hits = {
        'hit_result': eval_result_pred_row,
        'hit_bet': eval_bet_row,
        'hit_goal': eval_goal_row,
        'hit_btts': eval_btts_suggestion_row,
        'hit_combo': eval_sugestao_combo_row,
        'hit_score': eval_score_pred_row,
    }
    for col, func in hits.items():
        df_eval[col] = df_eval.apply(func, axis=1).map({True: 1, False: 0})

    df_melted = df_eval.melt(
        id_vars=['date', 'tournament_id', 'model'],
        value_vars=list(hits.keys()),
        var_name='mercado', value_name='acerto'
    ).dropna(subset=['acerto'])
    df_melted['day'] = df_melted['date'].dt.date

    agg = df_melted.groupby(['day', 'tournament_id', 'model', 'mercado']).agg(
        total_acertos=('acerto', 'sum'),
        total_jogos=('acerto', 'count')
    ).reset_index()
    agg['taxa_acerto'] = (agg['total_acertos'] / agg['total_jogos'] * 100).round(2)

    agg['mercado'] = agg['mercado'].map({
        'hit_result': 'Resultado Final', 'hit_bet': 'Sugestão de Aposta',
        'hit_goal': 'Sugestão de Gols', 'hit_btts': 'Ambos Marcam',
        'hit_combo': 'Sugestão Combo', 'hit_score': 'Placar Exato'
    })
    agg['tournament_id'] = agg['tournament_id'].apply(tournament_label)

    return agg.rename(columns={
        'day': 'Data', 'tournament_id': 'Campeonato', 'model': 'Modelo',
        'mercado': 'Métrica', 'taxa_acerto': 'Taxa de Acerto (%)'
    })


def get_best_model_by_market(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula o melhor modelo por campeonato e mercado."""
    if df.empty or "model" not in df.columns:
        return pd.DataFrame()

    eval_map = {
        "hit_result": eval_result_pred_row, "hit_bet": eval_bet_row,
        "hit_goal": eval_goal_row, "hit_btts": eval_btts_suggestion_row,
    }
    df_eval = df.copy()
    for col, func in eval_map.items():
        df_eval[col] = df_eval.apply(func, axis=1).map({True: 1, False: 0})

    df_melted = df_eval.melt(
        id_vars=["tournament_id", "model"], value_vars=list(eval_map.keys()),
        var_name="mercado", value_name="acerto"
    ).dropna(subset=["acerto"])
    if df_melted.empty:
        return pd.DataFrame()

    agg = df_melted.groupby(["tournament_id", "model", "mercado"], as_index=False).agg(
        total_acertos=("acerto", "sum"), total_jogos=("acerto", "count")
    )
    agg['taxa_acerto'] = (agg['total_acertos'] / agg['total_jogos'] * 100).round(2)

    df_best = agg.sort_values(
        by=["tournament_id", "mercado", "taxa_acerto", "total_jogos"],
        ascending=[True, True, False, False]
    ).drop_duplicates(subset=["tournament_id", "mercado"])

    df_best["mercado"] = df_best["mercado"].map({
        "hit_result": "Resultado Final", "hit_bet": "Sugestão de Aposta",
        "hit_goal": "Sugestão de Gols", "hit_btts": "Ambos Marcam"
    })
    df_best["tournament_id"] = df_best["tournament_id"].apply(tournament_label)

    return df_best.rename(columns={
        "tournament_id": "Campeonato", "mercado": "Mercado de Aposta",
        "model": "Melhor Modelo", "taxa_acerto": "Taxa de Acerto (%)",
        "total_jogos": "Total de Jogos Avaliados",
    })


def create_summary_pivot_table(best_model_df: pd.DataFrame) -> pd.DataFrame:
    """Cria uma tabela pivô de resumo."""
    if best_model_df.empty:
        return pd.DataFrame()

    summary = best_model_df.groupby(["Campeonato", "Melhor Modelo"]).agg(
        Mercados_de_Aposta=("Mercado de Aposta", lambda s: ", ".join(sorted(set(s)))),
        Taxa_media_percentual=("Taxa de Acerto (%)", "mean"),
        Jogos_avaliados=("Total de Jogos Avaliados", "sum"),
    ).reset_index()
    summary["Taxa_media_percentual"] = summary["Taxa_media_percentual"].round(2)
    return summary.sort_values(by=["Campeonato", "Taxa_media_percentual"], ascending=[True, False])


def find_best_bet(row, prob_min: float, odd_min: float, markets: Optional[List[str]] = None) -> pd.Series:
    """Encontra a melhor aposta para uma linha de dados."""
    best_bet, max_prob = None, -1.0
    markets = markets or list(MARKET_TO_ODDS_COLS.keys())

    for market in markets:
        prob_col, odd_col = MARKET_TO_ODDS_COLS.get(market, (None, None))
        if prob_col and odd_col and pd.notna(row.get(prob_col)) and pd.notna(row.get(odd_col)):
            prob, odd = row[prob_col], row[odd_col]
            if prob >= prob_min and odd >= odd_min and prob > max_prob:
                max_prob, best_bet = prob, {"market": market, "prob": prob, "odd": odd}

    return pd.Series(best_bet or {"market": np.nan, "prob": np.nan, "odd": np.nan})


def suggest_btts(row, prob_min: float, odd_min: float) -> pd.Series:
    """Sugere a melhor aposta 'Ambos Marcam'."""
    prob_yes, prob_no = row.get("prob_btts_yes", -1), row.get("prob_btts_no", -1)
    market, prob, odd = ("btts_yes", prob_yes, row.get("odds_btts_yes")) if prob_yes > prob_no else ("btts_no", prob_no, row.get("odds_btts_no"))

    if not (pd.notna(prob) and pd.notna(odd) and prob >= prob_min and odd >= odd_min):
        return pd.Series({"btts_sugg_market": np.nan, "btts_sugg_prob": np.nan, "btts_sugg_odd": np.nan})
    return pd.Series({"btts_sugg_market": market, "btts_sugg_prob": prob, "btts_sugg_odd": odd})
