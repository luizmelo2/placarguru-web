"""Página de Análise de Desempenho do Placar Guru."""
import streamlit as st
import pandas as pd
import altair as alt
from typing import Optional, List

# Importa funções e constantes do utils.py
from utils import (
    RELEASE_URL, fetch_release_file, load_data,
    tournament_label, norm_status_key, evaluate_market
)
from ui_components import filtros_analise_ui
from analysis import find_best_bet, suggest_btts

# ============================
# Configuração da página
# ============================
st.set_page_config(
    layout="wide",
    page_title="Análise de Desempenho",
)

st.title("Análise de Desempenho")

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
            lambda row: find_best_bet(
                row, flt["prob_min"], flt["odd_min"],
                markets_to_search=markets_1x2
            ),
            axis=1
        ).rename(columns={
            "market": "bet_1x2_market", "prob": "bet_1x2_prob",
            "odd": "bet_1x2_odd"
        })

        # 2. Melhor Aposta de Gols
        markets_goals = [
            "over_0_5", "over_1_5", "over_2_5", "over_3_5",
            "under_0_5", "under_1_5", "under_2_5", "under_3_5",
            "btts_yes", "btts_no"
        ]
        best_goals_df = df_filtered.apply(
            lambda row: find_best_bet(
                row, flt["prob_min"], flt["odd_min"],
                markets_to_search=markets_goals
            ),
            axis=1
        ).rename(columns={
            "market": "bet_goals_market", "prob": "bet_goals_prob",
            "odd": "bet_goals_odd"
        })

        # 3. Melhor Aposta (Geral)
        best_overall_df = df_filtered.apply(
            lambda row: find_best_bet(row, flt["prob_min"], flt["odd_min"]),
            axis=1
        ).rename(columns={
            "market": "bet_overall_market", "prob": "bet_overall_prob",
            "odd": "bet_overall_odd"
        })

        # 4. Sugestão de "Ambos Marcam"
        btts_sugg_df = df_filtered.apply(
            lambda row: suggest_btts(row, flt["prob_min"], flt["odd_min"]),
            axis=1
        )

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
            consolidated_accuracy_data = []
            if not df_finished.empty:
                # Dicionário para mapear colunas de resultado para nomes de métricas amigáveis
                metric_map = {
                    'res_bet_1x2': 'Melhor Resultado (1x2)',
                    'res_bet_goals': 'Melhor Aposta de Gols',
                    'res_bet_overall': 'Melhor Aposta (Geral)',
                    'res_btts_sugg': 'Sugestão "Ambos Marcam"'
                }

                def compute_accuracy_rows(group_df, model, tournament=None):
                    rows = []
                    for res_col, metric_name in metric_map.items():
                        # Filtra o grupo para as apostas que foram realmente feitas (não NaN)
                        valid_bets = group_df[res_col].dropna()
                        total = len(valid_bets)

                        if total > 0:
                            # Booleans (True=1, False=0) -> soma = acertos
                            hits = int(valid_bets.sum())
                            errors = total - hits
                            accuracy = (hits / total) * 100

                            row = {
                                "Modelo": model,
                                "Métrica": metric_name,
                                "Acerto (%)": accuracy,
                                "Acertos": hits,
                                "Erros": errors,
                                "Total": total
                            }
                            if tournament is not None:
                                row["Campeonato"] = tournament_label(tournament)
                            rows.append(row)
                    return rows

                grouped = df_finished.groupby(['tournament_id', 'model'])
                for (tournament, model), group in grouped:
                    accuracy_data.extend(compute_accuracy_rows(group, model, tournament))

                consolidated_grouped = df_finished.groupby(['model'])
                for model, group in consolidated_grouped:
                    consolidated_accuracy_data.extend(compute_accuracy_rows(group, model))

            if not accuracy_data:
                st.warning("Não há dados de acurácia para exibir.")
            else:
                metrics_df = pd.DataFrame(accuracy_data)
                consolidated_metrics_df = pd.DataFrame(consolidated_accuracy_data)

                if flt.get("metrics_sel"):
                    metrics_df = metrics_df[metrics_df["Métrica"].isin(flt["metrics_sel"])]
                    consolidated_metrics_df = consolidated_metrics_df[
                        consolidated_metrics_df["Métrica"].isin(flt["metrics_sel"])
                    ]

                if metrics_df.empty:
                    st.warning("Não há dados de acurácia para os filtros de métricas selecionados.")
                    st.stop()

                if not consolidated_metrics_df.empty:
                    st.subheader("Tabela Consolidada (Todos os Campeonatos)")
                    st.dataframe(
                        consolidated_metrics_df.sort_values(
                            by=["Modelo", "Acerto (%)"],
                            ascending=[True, False]
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

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
                    tooltip=[
                        'Campeonato', 'Modelo', 'Métrica', 'Acertos', 'Total',
                        alt.Tooltip('Acerto (%):Q', format='.1f')
                    ]
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
