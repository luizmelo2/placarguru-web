
"""Página de Análise de Desempenho do Placar Guru."""
import streamlit as st
import pandas as pd
import altair as alt

from utils import (
    RELEASE_URL, fetch_release_file, load_data,
    tournament_label, norm_status_key, evaluate_market
)
from ui_components import filtros_analise_ui
from analysis import find_best_bet, suggest_btts

st.set_page_config(layout="wide", page_title="Análise de Desempenho")
st.title("Análise de Desempenho")

try:
    content, _, _ = fetch_release_file(RELEASE_URL)
    df = load_data(content)

    if df.empty:
        st.error("O arquivo `PrevisaoJogos.xlsx` não pôde ser lido.")
    else:
        flt = filtros_analise_ui(df)
        mask = pd.Series(True, index=df.index)
        if flt["tournaments_sel"]:
            mask &= df["tournament_id"].isin(flt["tournaments_sel"])
        if flt["models_sel"]:
            mask &= df["model"].isin(flt["models_sel"])
        if flt["selected_date_range"] and len(flt["selected_date_range"]) == 2:
            start, end = flt["selected_date_range"]
            mask &= (df["date"].dt.date.between(start, end)) | (df["date"].isna())

        df_filtered = df[mask].copy()

        bet_finders = {
            "1x2": (find_best_bet, {"markets_to_search": ["H", "D", "A"]}),
            "goals": (find_best_bet, {"markets_to_search": [
                "over_0_5", "over_1_5", "over_2_5", "over_3_5",
                "under_0_5", "under_1_5", "under_2_5", "under_3_5",
                "btts_yes", "btts_no"
            ]}),
            "overall": (find_best_bet, {}),
            "btts": (suggest_btts, {}),
        }

        for name, (func, kwargs) in bet_finders.items():
            res = df_filtered.apply(lambda row: func(row, flt["prob_min"], flt["odd_min"], **kwargs), axis=1)
            res.columns = [f"bet_{name}_{c}" for c in res.columns]
            df_filtered = df_filtered.join(res)

        df_analysis = df_filtered.dropna(
            subset=[f"bet_{name}_market" for name in bet_finders],
            how='all'
        )

        st.subheader("Análise de Acurácia Comparativa")
        st.info(f"Analisando {len(df_analysis)} jogos com Prob. >= {flt['prob_min']:.0%} e Odd >= {flt['odd_min']:.2f}.")

        df_finished = df_analysis[df_analysis['status'].apply(norm_status_key) == 'finished'].copy()
        df_finished = df_finished.dropna(subset=['result_home', 'result_away'])

        if df_finished.empty:
            st.warning("Nenhum jogo finalizado encontrado para os critérios.")
        else:
            for name in bet_finders:
                df_finished[f'res_bet_{name}'] = df_finished.apply(
                    lambda r: evaluate_market(r[f'bet_{name}_market'], r['result_home'], r['result_away']), axis=1
                )

            metric_map = {
                'res_bet_1x2': 'Melhor 1x2', 'res_bet_goals': 'Melhor Gols',
                'res_bet_overall': 'Melhor Geral', 'res_bet_btts': 'Sugestão BTTS'
            }

            accuracy_data = [
                {
                    "Campeonato": tournament_label(tournament), "Modelo": model, "Métrica": metric_name,
                    "Acerto (%)": (group[res_col].sum() / len(group[res_col].dropna())) * 100,
                    "Acertos": int(group[res_col].sum()), "Total": len(group[res_col].dropna())
                }
                for (tournament, model), group in df_finished.groupby(['tournament_id', 'model'])
                for res_col, metric_name in metric_map.items() if not group[res_col].dropna().empty
            ]

            if not accuracy_data:
                st.warning("Não há dados de acurácia para exibir.")
            else:
                metrics_df = pd.DataFrame(accuracy_data)
                st.dataframe(metrics_df.sort_values(by=["Campeonato", "Modelo", "Acerto (%)"], ascending=[True, True, False]), use_container_width=True, hide_index=True)

                chart = alt.Chart(metrics_df).mark_bar().encode(
                    x=alt.X('Modelo:N', title='Modelo', axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('Acerto (%):Q', scale=alt.Scale(domain=[0, 100]), title='Acurácia (%)'),
                    color='Métrica:N', xOffset='Métrica:N',
                    tooltip=['Campeonato', 'Modelo', 'Métrica', 'Acertos', 'Total', alt.Tooltip('Acerto (%):Q', format='.1f')]
                ).properties(height=300).facet(facet='Campeonato:N', columns=2)
                st.altair_chart(chart, use_container_width=True)

except Exception as e:
    st.error(f"Erro inesperado: {e}")
    st.exception(e)
