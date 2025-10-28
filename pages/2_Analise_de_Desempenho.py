import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta, date, datetime
from typing import Any, Tuple, Optional, List

# Importa funções e constantes do utils.py
from utils import *  # RELEASE_URL, fetch_release_file, load_data, FRIENDLY_COLS,
                     # tournament_label, norm_status_key, evaluate_market,
                     # eval_result_pred_row, eval_goal_row

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
    "Hx": ("prob_Hx", "odds_Hx"), "xA": ("prob_xA", "odds_xA"),
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

def find_best_bet(row, prob_min: float, odd_min: float, markets_to_search: Optional[List[str]] = None) -> pd.Series:
    """Encontra a melhor aposta para uma linha de dados, considerando os mercados especificados."""
    best_bet, max_prob = None, -1.0

    # Se nenhum mercado for especificado, busca em todos os mercados definidos.
    if markets_to_search is None:
        markets_to_search = list(MARKET_COLUMNS.keys())

    for market in markets_to_search:
        if market in MARKET_COLUMNS:
            prob_col, odd_col = MARKET_COLUMNS[market]
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
    # Carrega dados da release
    content, _, _ = fetch_release_file(RELEASE_URL)
    df = load_data(content)

    if df.empty:
        st.error("O arquivo `PrevisaoJogos.xlsx` está vazio ou não pôde ser lido.")
    else:
        flt = filtros_analise_ui(df)

        # Filtros
        mask = pd.Series(True, index=df.index)
        if flt["tournaments_sel"] and "tournament_id" in df.columns:
            mask &= df["tournament_id"].isin(flt["tournaments_sel"])
        if flt["models_sel"] and "model" in df.columns:
            mask &= df["model"].isin(flt["models_sel"])
        if flt["selected_date_range"] and len(flt["selected_date_range"]) == 2 and "date" in df.columns:
            start, end = flt["selected_date_range"]
            mask &= (df["date"].dt.date.between(start, end)) | (df["date"].isna())

        df_filtered = df[mask].copy()

        # --- Lógica das Três Buscas Dinâmicas ---

        # 1. Melhor Resultado (1x2)
        markets_1x2 = ["H", "D", "A"]
        best_1x2_df = df_filtered.apply(
            lambda row: find_best_bet(row, flt["prob_min"], flt["odd_min"], markets_to_search=markets_1x2),
            axis=1
        ).rename(columns={"market": "bet_1x2_market", "prob": "bet_1x2_prob", "odd": "bet_1x2_odd"})

        # 2. Melhor Aposta de Gols
        markets_goals = [
            "over_0_5", "over_1_5", "over_2_5", "over_3_5",
            "under_0_5", "under_1_5", "under_2_5", "under_3_5",
            "btts_yes", "btts_no"
        ]
        best_goals_df = df_filtered.apply(
            lambda row: find_best_bet(row, flt["prob_min"], flt["odd_min"], markets_to_search=markets_goals),
            axis=1
        ).rename(columns={"market": "bet_goals_market", "prob": "bet_goals_prob", "odd": "bet_goals_odd"})

        # 3. Melhor Aposta (Geral)
        best_overall_df = df_filtered.apply(
            lambda row: find_best_bet(row, flt["prob_min"], flt["odd_min"]),
            axis=1
        ).rename(columns={"market": "bet_overall_market", "prob": "bet_overall_prob", "odd": "bet_overall_odd"})

        # 4. Sugestão de "Ambos Marcam"
        btts_sugg_df = df_filtered.apply(suggest_btts, axis=1)

        # Junta os resultados das buscas ao dataframe principal
        df_analysis = df_filtered.join(best_1x2_df).join(best_goals_df).join(best_overall_df).join(btts_sugg_df)

        # Remove jogos onde nenhuma aposta foi encontrada em nenhuma das categorias
        df_analysis.dropna(
            subset=["bet_1x2_market", "bet_goals_market", "bet_overall_market", "btts_sugg_market"],
            how='all',
            inplace=True
        )

        st.subheader("Análise de Acurácia Comparativa")
        st.info(f"Analisando {len(df_analysis)} jogos que atendem aos critérios de Prob. >= {flt['prob_min']:.0%} e Odd >= {flt['odd_min']:.2f}.")

        # Considere apenas finalizados com placar válido
        df_finished = df_analysis[df_analysis['status'].apply(norm_status_key) == 'finished'].copy()
        df_finished = df_finished.dropna(subset=['result_home', 'result_away'])

        if df_finished.empty:
            st.warning("Nenhum jogo finalizado encontrado para os critérios e filtros selecionados.")
        else:
            # Avaliações para cada tipo de aposta
            df_finished['res_bet_1x2'] = df_finished.apply(
                lambda row: evaluate_market(row['bet_1x2_market'], row['result_home'], row['result_away']), axis=1
            )
            df_finished['res_bet_goals'] = df_finished.apply(
                lambda row: evaluate_market(row['bet_goals_market'], row['result_home'], row['result_away']), axis=1
            )
            df_finished['res_bet_overall'] = df_finished.apply(
                lambda row: evaluate_market(row['bet_overall_market'], row['result_home'], row['result_away']), axis=1
            )
            df_finished['res_btts_sugg'] = df_finished.apply(
                lambda row: evaluate_market(row['btts_sugg_market'], row['result_home'], row['result_away']), axis=1
            )

            accuracy_data = []
            if not df_finished.empty:
                # Dicionário para mapear colunas de resultado para nomes de métricas amigáveis
                metric_map = {
                    'res_bet_1x2': 'Melhor Resultado (1x2)',
                    'res_bet_goals': 'Melhor Aposta de Gols',
                    'res_bet_overall': 'Melhor Aposta (Geral)',
                    'res_btts_sugg': 'Sugestão "Ambos Marcam"'
                }

                grouped = df_finished.groupby(['tournament_id', 'model'])
                for (tournament, model), group in grouped:
                    for res_col, metric_name in metric_map.items():
                        # Filtra o grupo para as apostas que foram realmente feitas (não NaN)
                        valid_bets = group[res_col].dropna()
                        total = len(valid_bets)

                        if total > 0:
                            # Booleans (True=1, False=0) -> soma = acertos
                            hits = int(valid_bets.sum())
                            accuracy = (hits / total) * 100

                            accuracy_data.append({
                                "Campeonato": tournament_label(tournament),
                                "Modelo": model,
                                "Métrica": metric_name,
                                "Acerto (%)": accuracy,
                                "Acertos": hits,
                                "Total": total
                            })

            if not accuracy_data:
                st.warning("Não há dados de acurácia para exibir.")
            else:
                metrics_df = pd.DataFrame(accuracy_data)

                st.dataframe(
                    metrics_df.sort_values(by=["Campeonato", "Modelo", "Acerto (%)"], ascending=[True, True, False]),
                    use_container_width=True,
                    hide_index=True
                )

                # ----------------------------
                # Gráfico de barras agrupadas e facetado por campeonato
                # (forma correta: facet via método, não no alt.Facet)
                # ----------------------------
                base = alt.Chart(metrics_df).mark_bar().encode(
                    x=alt.X('Modelo:N', title='Modelo', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Acerto (%):Q', scale=alt.Scale(domain=[0, 100]), title='Acurácia (%)'),
                    color=alt.Color('Métrica:N', title='Métrica'),
                    xOffset='Métrica:N',
                    tooltip=['Campeonato', 'Modelo', 'Métrica', 'Acertos', 'Total', alt.Tooltip('Acerto (%):Q', format='.1f')]
                ).properties(height=300)

                chart = base.facet(
                    facet='Campeonato:N',   # apenas o field/canal
                    columns=2,              # aqui sim, no método facet
                    title='Desempenho por Campeonato'
                ).resolve_scale(y='shared')  # deixe 'independent' se preferir escalas separadas

                st.altair_chart(chart, use_container_width=True)

except Exception as e:
    st.error(f"Erro inesperado durante a análise: {e}")
    st.exception(e)
