import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta, date, datetime
from typing import Any, Tuple, Optional, List

# Importa funções e constantes do utils.py
from utils import *

# ============================
# Configuração da página
# ============================
st.set_page_config(
    layout="wide",
    page_title="Análise de Desempenho",
)

st.title("Análise de Desempenho")

# ============================
# Lógica de Sugestão e Acurácia
# ============================
MARKET_COLUMNS = {
    "H": ("prob_H", "odds_H"), "D": ("prob_D", "odds_D"), "A": ("prob_A", "odds_A"),
    "over_0_5": ("prob_over_0_5", "odds_match_goals_0.5_over"),
    "over_1_5": ("prob_over_1_5", "odds_match_goals_1.5_over"),
    "over_2_5": ("prob_over_2_5", "odds_match_goals_2.5_over"),
    "over_3_5": ("prob_over_3_5", "odds_match_goals_3.5_over"),
    "under_0_5": ("prob_under_0_5", "odds_match_goals_0.5_under"),
    "under_1_5": ("prob_under_1_5", "odds_match_goals_1.5_under"),
    "under_2_5": ("prob_under_2_5", "odds_match_goals_2.5_under"),
    "under_3_5": ("prob_under_3_5", "odds_match_goals_3.5_under"),
    "btts_yes": ("prob_btts_yes", "odds_btts_yes"),
    "btts_no": ("prob_btts_no", "odds_btts_no"),
}

def find_best_bet(row, prob_min: float, odd_min: float) -> pd.Series:
    best_bet, max_prob = None, -1.0
    for market, (prob_col, odd_col) in MARKET_COLUMNS.items():
        if prob_col in row and odd_col in row and pd.notna(row[prob_col]) and pd.notna(row[odd_col]):
            prob, odd = row[prob_col], row[odd_col]
            if prob >= prob_min and odd >= odd_min and prob > max_prob:
                max_prob = prob
                best_bet = {"best_bet_market": market, "best_bet_prob": prob, "best_bet_odd": odd}
    return pd.Series(best_bet) if best_bet else pd.Series({"best_bet_market": np.nan, "best_bet_prob": np.nan, "best_bet_odd": np.nan})

# ============================
# UI de Filtros
# ============================
def filtros_analise_ui(df: pd.DataFrame) -> dict:
    st.sidebar.header("Parâmetros da Análise")
    prob_min = st.sidebar.slider("Probabilidade Mínima (%)", 0, 100, 65, 1, "%d%%") / 100.0
    odd_min = st.sidebar.slider("Odd Mínima", 1.0, 5.0, 1.3, 0.01)

    st.sidebar.header("Filtros de Jogos")
    tourn_opts = sorted(df["tournament_id"].dropna().unique().tolist()) if "tournament_id" in df.columns else []
    model_opts = sorted(df["model"].dropna().unique()) if "model" in df.columns else []

    models_sel = st.sidebar.multiselect(FRIENDLY_COLS["model"], model_opts, default=model_opts)
    tournaments_sel = st.sidebar.multiselect(FRIENDLY_COLS["tournament_id"], tourn_opts, default=tourn_opts, format_func=tournament_label)

    selected_date_range = ()
    if "date" in df.columns and df["date"].notna().any():
        min_date, max_date = df["date"].dropna().min().date(), df["date"].dropna().max().date()
        selected_date_range = st.sidebar.date_input("Período (intervalo)", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    return dict(prob_min=prob_min, odd_min=odd_min, tournaments_sel=tournaments_sel, models_sel=models_sel, selected_date_range=selected_date_range)

# ============================
# App principal
# ============================
try:
    content, _, _ = fetch_release_file(RELEASE_URL)
    df = load_data(content)

    if df.empty:
        st.error("O arquivo `PrevisaoJogos.xlsx` está vazio ou não pôde ser lido.")
    else:
        flt = filtros_analise_ui(df)

        mask = pd.Series(True, index=df.index)
        if flt["tournaments_sel"] and "tournament_id" in df.columns: mask &= df["tournament_id"].isin(flt["tournaments_sel"])
        if flt["models_sel"] and "model" in df.columns: mask &= df["model"].isin(flt["models_sel"])
        if flt["selected_date_range"] and len(flt["selected_date_range"]) == 2 and "date" in df.columns:
            start, end = flt["selected_date_range"]
            mask &= (df["date"].dt.date.between(start, end)) | (df["date"].isna())

        df_filtered = df[mask].copy()
        best_bets_df = df_filtered.apply(lambda row: find_best_bet(row, flt["prob_min"], flt["odd_min"]), axis=1)
        df_analysis = df_filtered.join(best_bets_df).dropna(subset=["best_bet_market"])

        st.subheader("Análise de Acurácia Comparativa")
        st.info(f"Analisando {len(df_analysis)} jogos que atendem aos critérios de Prob. >= {flt['prob_min']:.0%} e Odd >= {flt['odd_min']:.2f}.")

        df_finished = df_analysis[df_analysis['status'].apply(norm_status_key) == 'finished'].copy()
        df_finished = df_finished.dropna(subset=['result_home', 'result_away'])

        if df_finished.empty:
            st.warning("Nenhum jogo finalizado encontrado para os critérios e filtros selecionados.")
        else:
            df_finished['res_best_bet'] = df_finished.apply(lambda row: evaluate_market(row['best_bet_market'], row['result_home'], row['result_away']), axis=1)
            df_finished['res_result_pred'] = df_finished.apply(eval_result_pred_row, axis=1)
            df_finished['res_goal_sugg'] = df_finished.apply(eval_goal_row, axis=1)
            df_finished.dropna(subset=['res_best_bet', 'res_result_pred', 'res_goal_sugg'], inplace=True)

            accuracy_data = []
            if not df_finished.empty:
                grouped = df_finished.groupby(['tournament_id', 'model'])
                for (tournament, model), group in grouped:
                    total = len(group)
                    hits_best_bet = group['res_best_bet'].sum()
                    hits_result_pred = group['res_result_pred'].sum()
                    hits_goal_sugg = group['res_goal_sugg'].sum()

                    accuracy_data.append({"Campeonato": tournament_label(tournament), "Modelo": model, "Métrica": "Melhor Aposta (Dinâmica)", "Acerto (%)": (hits_best_bet / total) * 100, "Acertos": int(hits_best_bet), "Total": total})
                    accuracy_data.append({"Campeonato": tournament_label(tournament), "Modelo": model, "Métrica": "Resultado do Jogo", "Acerto (%)": (hits_result_pred / total) * 100, "Acertos": int(hits_result_pred), "Total": total})
                    accuracy_data.append({"Campeonato": tournament_label(tournament), "Modelo": model, "Métrica": "Sugestão de Gols", "Acerto (%)": (hits_goal_sugg / total) * 100, "Acertos": int(hits_goal_sugg), "Total": total})

            if not accuracy_data:
                st.warning("Não há dados de acurácia para exibir.")
            else:
                metrics_df = pd.DataFrame(accuracy_data)

                st.dataframe(metrics_df.sort_values(by=["Campeonato", "Modelo", "Acerto (%)"], ascending=[True, True, False]), use_container_width=True, hide_index=True)

                # Gráfico aprimorado com barras agrupadas
                chart = alt.Chart(metrics_df).mark_bar().encode(
                    x=alt.X('Modelo:N', title='Modelo', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Acerto (%):Q', scale=alt.Scale(domain=[0, 100]), title='Acurácia (%)'),
                    color=alt.Color('Métrica:N', title='Métrica'),
                    xOffset='Métrica:N', # Agrupa as barras por métrica
                    tooltip=['Campeonato', 'Modelo', 'Métrica', 'Acertos', 'Total', alt.Tooltip('Acerto (%):Q', format='.1f')]
                ).properties(
                    height=300
                ).facet(
                    facet=alt.Facet('Campeonato:N', columns=2, title="Desempenho por Campeonato")
                )

                st.altair_chart(chart, use_container_width=True)

except Exception as e:
    st.error(f"Erro inesperado durante a análise: {e}")
    st.exception(e)
