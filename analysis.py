import pandas as pd
from typing import Optional, List
import numpy as np

from utils import (
    eval_result_pred_row, eval_bet_row, eval_goal_row,
    predict_btts_from_prob, evaluate_market, eval_sugestao_combo_row,
    eval_score_pred_row, tournament_label, MARKET_TO_ODDS_COLS, parse_score_pred
)

def compute_acc2(ok_mask: pd.Series, bad_mask: pd.Series):
    total = int((ok_mask | bad_mask).sum())
    correct = int(ok_mask.sum())
    acc = (correct / total * 100.0) if total > 0 else np.nan
    return acc, correct, total

def _calculate_metric(sub: pd.DataFrame, metric_name: str, eval_func, **kwargs) -> dict:
    """Helper genérico para calcular acurácia de uma métrica."""
    eval_series = sub.apply(eval_func, axis=1, **kwargs)
    correct_mask = eval_series == True
    wrong_mask = eval_series == False
    acc, correct, total = compute_acc2(correct_mask, wrong_mask)
    return {
        "Métrica": metric_name,
        "Acerto (%)": 0 if np.isnan(acc) else round(acc, 1),
        "Acertos": correct,
        "Total Avaliado": total
    }

def calculate_kpis_for_model(sub: pd.DataFrame, model_name: Optional[str] = None) -> List[dict]:
    """Calcula todas as métricas de KPI para um determinado dataframe (subconjunto de um modelo)."""

    rows = [
        _calculate_metric(sub, "Resultado", eval_result_pred_row),
        _calculate_metric(sub, "Sugestão de Aposta", eval_bet_row),
        _calculate_metric(sub, "Sugestão Combo", eval_sugestao_combo_row),
        _calculate_metric(sub, "Placar Previsto", eval_score_pred_row),
    ]

    goal_eval_s = sub.apply(eval_goal_row, axis=1)
    goal_correct_s = goal_eval_s == True
    goal_wrong_s   = goal_eval_s == False

    goal_codes_s = sub.get("goal_bet_suggestion", pd.Series(index=sub.index, dtype="object"))
    btts_mask_s = goal_codes_s.astype(str).str.lower().isin(['btts_yes', 'btts_no'])

    btts_correct_s = goal_correct_s & btts_mask_s
    btts_wrong_s = goal_wrong_s & btts_mask_s
    acc_btts, c_btts, t_btts = compute_acc2(btts_correct_s, btts_wrong_s)
    rows.append({
        "Métrica": "Ambos Marcam (Sugestão)",
        "Acerto (%)": 0 if np.isnan(acc_btts) else round(acc_btts, 1),
        "Acertos": c_btts,
        "Total Avaliado": t_btts
    })

    goal_correct_no_btts = goal_correct_s & ~btts_mask_s
    goal_wrong_no_btts   = goal_wrong_s & ~btts_mask_s
    acc_goal, c_goal, t_goal = compute_acc2(goal_correct_no_btts, goal_wrong_no_btts)
    rows.append({
        "Métrica": "Sugestão de Gols",
        "Acerto (%)": 0 if np.isnan(acc_goal) else round(acc_goal, 1),
        "Acertos": c_goal,
        "Total Avaliado": t_goal
    })

    rh_s = sub.get("result_home", pd.Series(index=sub.index, dtype="float"))
    ra_s = sub.get("result_away", pd.Series(index=sub.index, dtype="float"))
    mv_s = rh_s.notna() & ra_s.notna()

    btts_pred_s = sub.apply(predict_btts_from_prob, axis=1)
    btts_pred_eval_s = pd.Series(index=sub.index, dtype="object")
    for idx in sub.index:
        btts_pred_eval_s.loc[idx] = evaluate_market(btts_pred_s.loc[idx], rh_s.loc[idx], ra_s.loc[idx]) if mv_s.loc[idx] else None

    btts_pred_correct_s = btts_pred_eval_s == True
    btts_pred_wrong_s   = btts_pred_eval_s == False
    acc_btts_pred, c_btts_pred, t_btts_pred = compute_acc2(btts_pred_correct_s, btts_pred_wrong_s)
    rows.append({
        "Métrica": "Ambos Marcam (Prob)",
        "Acerto (%)": 0 if np.isnan(acc_btts_pred) else round(acc_btts_pred, 1),
        "Acertos": c_btts_pred,
        "Total Avaliado": t_btts_pred
    })

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
    if df.empty or 'date' not in df.columns or 'tournament_id' not in df.columns or 'model' not in df.columns:
        return pd.DataFrame()

    df_eval = df.copy()
    # Calcula o acerto para cada mercado e armazena em novas colunas
    df_eval['hit_result'] = df_eval.apply(eval_result_pred_row, axis=1)
    df_eval['hit_bet'] = df_eval.apply(eval_bet_row, axis=1)
    df_eval['hit_goal'] = df_eval.apply(eval_goal_row, axis=1)
    df_eval['hit_btts'] = df_eval.apply(lambda row: evaluate_market(predict_btts_from_prob(row), row.get("result_home"), row.get("result_away")), axis=1)
    df_eval['hit_combo'] = df_eval.apply(eval_sugestao_combo_row, axis=1)
    df_eval['hit_score'] = df_eval.apply(eval_score_pred_row, axis=1)

    # Converte True/False para 1/0 para poder agregar, mantendo Nones como NaN
    hit_cols = ['hit_result', 'hit_bet', 'hit_goal', 'hit_btts', 'hit_combo', 'hit_score']
    for col in hit_cols:
        df_eval[col] = df_eval[col].apply(lambda x: 1 if x is True else (0 if x is False else np.nan))

    # Reestrutura para formato longo (tidy data), agora incluindo o modelo
    id_vars = ['date', 'tournament_id', 'model']
    df_melted = df_eval.melt(id_vars=id_vars, value_vars=hit_cols, var_name='mercado', value_name='acerto')
    df_melted.dropna(subset=['acerto'], inplace=True)

    # Extrai o dia (sem o horário)
    df_melted['day'] = df_melted['date'].dt.date

    # Agrupa por dia, campeonato, modelo e mercado
    agg = df_melted.groupby(['day', 'tournament_id', 'model', 'mercado']).agg(
        total_acertos=('acerto', 'sum'),
        total_jogos=('acerto', 'count')
    ).reset_index()

    # Calcula a taxa de acerto
    agg['taxa_acerto'] = (agg['total_acertos'] / agg['total_jogos'] * 100).round(2)

    # Formata os nomes para exibição
    agg['mercado'] = agg['mercado'].map({
        'hit_result': 'Resultado Final',
        'hit_bet': 'Sugestão de Aposta',
        'hit_goal': 'Sugestão de Gols',
        'hit_btts': 'Ambos Marcam (Prob)',
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

    return df_final[['Data', 'Campeonato', 'Modelo', 'Métrica', 'Taxa de Acerto (%)']]

def get_best_model_by_market(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o melhor modelo por campeonato e mercado de aposta com base na taxa de acerto.
    """
    if df.empty or 'model' not in df.columns or 'tournament_id' not in df.columns:
        return pd.DataFrame()

    # 1. Calcula o acerto para cada mercado de interesse
    df['hit_result'] = df.apply(eval_result_pred_row, axis=1)
    df['hit_bet'] = df.apply(eval_bet_row, axis=1)
    df['hit_goal'] = df.apply(eval_goal_row, axis=1)
    df['hit_btts'] = df.apply(lambda row: evaluate_market(predict_btts_from_prob(row), row.get("result_home"), row.get("result_away")), axis=1)

    # Converte True/False para 1/0 para agregação, mantendo None como NaN
    for col in ['hit_result', 'hit_bet', 'hit_goal', 'hit_btts']:
        df[col] = df[col].apply(lambda x: 1 if x is True else (0 if x is False else np.nan))

    # 2. Reestrutura os dados para formato longo (tidy)
    id_vars = ['tournament_id', 'model']
    value_vars = ['hit_result', 'hit_bet', 'hit_goal', 'hit_btts']
    df_melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='mercado', value_name='acerto')
    df_melted.dropna(subset=['acerto'], inplace=True)

    # 3. Agrupa para calcular acertos e totais
    agg = df_melted.groupby(['tournament_id', 'model', 'mercado']).agg(
        total_acertos=('acerto', 'sum'),
        total_jogos=('acerto', 'count')
    ).reset_index()

    # 4. Calcula a taxa de acerto
    agg['taxa_acerto'] = (agg['total_acertos'] / agg['total_jogos'] * 100).round(2)

    # 5. Encontra o melhor modelo para cada campeonato e mercado
    best_model_idx = agg.groupby(['tournament_id', 'mercado'])['taxa_acerto'].idxmax()
    df_best = agg.loc[best_model_idx].copy()

    # 6. Formata o DataFrame final
    df_best['mercado'] = df_best['mercado'].map({
        'hit_result': 'Resultado Final',
        'hit_bet': 'Sugestão de Aposta',
        'hit_goal': 'Sugestão de Gols',
        'hit_btts': 'Ambos Marcam (Prob)'
    })
    df_best['tournament_id'] = df_best['tournament_id'].apply(tournament_label)

    df_final = df_best[['tournament_id', 'mercado', 'model', 'taxa_acerto', 'total_jogos']]
    df_final = df_final.rename(columns={
        'tournament_id': 'Campeonato',
        'mercado': 'Mercado de Aposta',
        'model': 'Melhor Modelo',
        'taxa_acerto': 'Taxa de Acerto (%)',
        'total_jogos': 'Total de Jogos Avaliados'
    })

    return df_final.sort_values(by=['Campeonato', 'Mercado de Aposta']).reset_index(drop=True)

def create_summary_pivot_table(best_model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria uma tabela resumo (pivot) mostrando o melhor modelo e sua taxa de acerto.
    """
    if best_model_df.empty:
        return pd.DataFrame()

    # Cria uma nova coluna combinando o modelo e a taxa de acerto
    df_copy = best_model_df.copy()
    df_copy['display_value'] = df_copy.apply(
        lambda row: f"{row['Melhor Modelo']} ({row['Taxa de Acerto (%)']:.1f}%)", axis=1
    )

    pivot_df = df_copy.pivot_table(
        index='Campeonato',
        columns='Mercado de Aposta',
        values='display_value',
        aggfunc='first'
    ).fillna('-')

    return pivot_df

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

def suggest_btts(row) -> pd.Series:
    """Sugere a melhor aposta 'Ambos Marcam' com base na maior probabilidade."""
    prob_yes = row.get("prob_btts_yes", -1)
    prob_no = row.get("prob_btts_no", -1)

    # Retorna NaN se as probabilidades não estiverem disponíveis
    if pd.isna(prob_yes) or pd.isna(prob_no):
        return pd.Series({
            "btts_sugg_market": np.nan,
            "btts_sugg_prob": np.nan,
            "btts_sugg_odd": np.nan
        })

    if prob_yes > prob_no:
        market = "btts_yes"
        prob = prob_yes
        odd = row.get("odds_btts_yes")
    else:
        market = "btts_no"
        prob = prob_no
        odd = row.get("odds_btts_no")

    return pd.Series({
        "btts_sugg_market": market,
        "btts_sugg_prob": prob,
        "btts_sugg_odd": odd
    })
