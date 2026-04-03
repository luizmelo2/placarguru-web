"""Tela dedicada de ranking de performance por mercado e modelo."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import altair as alt

from analysis import build_model_ranking_by_market, build_weekly_accuracy_by_model
from ui_components import render_glassy_table
from utils import RELEASE_URL, fetch_release_file, load_data, norm_status_key


def _game_key(frame: pd.DataFrame) -> pd.Series:
    """Monta chave de jogo único para diferenciar partidas de previsões por modelo."""

    id_candidates = ["match_id", "fixture_id", "game_id", "id"]
    for col in id_candidates:
        if col in frame.columns and frame[col].notna().any():
            return frame[col].astype(str)

    cols = [c for c in ["date", "tournament_id", "home", "away"] if c in frame.columns]
    if cols:
        return frame[cols].astype(str).agg("|".join, axis=1)

    return pd.Series(frame.index.astype(str), index=frame.index)


st.set_page_config(layout="wide", page_title="Ranking por Mercado")
st.title("Ranking por Mercado x Modelo")
st.caption("Tabela dedicada para comparar acerto por mercado entre modelos.")

try:
    content, _, _ = fetch_release_file(RELEASE_URL)
    # Para o filtro de quantidade funcionar corretamente, carregamos todo o histórico
    # e aplicamos o recorte apenas pela quantidade de jogos passados.
    df = load_data(content, months_back=0)

    if df.empty:
        st.warning("Não há dados disponíveis para análise.")
        st.stop()

    status_norm = df.get("status", pd.Series("", index=df.index)).astype(str).map(norm_status_key)
    rh = pd.to_numeric(df.get("result_home", pd.Series(index=df.index)), errors="coerce")
    ra = pd.to_numeric(df.get("result_away", pd.Series(index=df.index)), errors="coerce")
    df_finished = df[(status_norm == "finished") & rh.notna() & ra.notna()].copy()

    if df_finished.empty:
        st.warning("Não há jogos finalizados com placar válido para montar os rankings.")
        st.stop()

    if "date" in df_finished.columns:
        df_finished = df_finished.sort_values("date", ascending=False)

    df_finished["_game_key"] = _game_key(df_finished)
    jogos_unicos = df_finished["_game_key"].drop_duplicates().tolist()

    max_games = int(len(jogos_unicos))
    default_games = min(30, max_games)
    qtd_jogos = st.sidebar.number_input(
        "Quantidade de jogos passados",
        min_value=1,
        max_value=max_games,
        value=default_games,
        step=1,
        help="Define quantos jogos finalizados mais recentes serão usados na análise.",
    )

    selected_game_keys = set(jogos_unicos[: int(qtd_jogos)])
    recorte = df_finished[df_finished["_game_key"].isin(selected_game_keys)].copy()

    st.info(
        f"Analisando os últimos {len(selected_game_keys)} jogos finalizados "
        f"({len(recorte)} previsões no total)."
    )

    rankings = build_model_ranking_by_market(recorte)
    if not rankings:
        st.warning("Não foi possível gerar os rankings para o recorte selecionado.")
        st.stop()

    st.subheader("Ranking de acerto por mercado")
    st.caption(
        "Observação: o filtro usa quantidade de jogos únicos. "
        "O total avaliado por mercado pode ser menor quando faltarem previsões específicas naquele mercado."
    )
    for market_name, ranking_df in rankings.items():
        if ranking_df.empty:
            continue
        safe_market = str(market_name).strip().lower().replace(" ", "_").replace("—", "-")
        safe_key = f"ranking_market_{safe_market}"
        render_glassy_table(
            ranking_df,
            caption=f"Ranking por modelo — {market_name}",
            key=safe_key,
        )

    st.subheader("Evolução de acerto por modelo (Mercado de Gols)")
    block_size = st.number_input(
        "Corte por quantidade de jogos",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        help="Cada ponto do gráfico representa um bloco sequencial dessa quantidade de jogos por modelo.",
    )
    st.caption(
        f"Cada ponto representa um bloco sequencial de {int(block_size)} jogos por modelo "
        "(última data do bloco no eixo X)."
    )
    weekly_df = build_weekly_accuracy_by_model(
        recorte,
        market_label="Sugestão de Gols",
        block_size=int(block_size),
    )
    if weekly_df.empty:
        st.caption("Sem dados suficientes para o gráfico de evolução no recorte atual.")
    else:
        modelos_disponiveis = sorted(weekly_df["Modelo"].dropna().unique().tolist())
        modelos_sel = st.multiselect(
            "Modelos exibidos no gráfico",
            options=modelos_disponiveis,
            default=modelos_disponiveis,
            help="Filtre modelos para melhorar a comparação visual.",
        )
        if not modelos_sel:
            st.caption("Selecione ao menos um modelo para exibir o gráfico.")
        else:
            weekly_filtered = weekly_df[weekly_df["Modelo"].isin(modelos_sel)]
            base = (
                alt.Chart(weekly_filtered)
                .mark_line(point=alt.OverlayMarkDef(size=70))
                .encode(
                    x=alt.X("Data de Corte:T", title="Data de corte"),
                    y=alt.Y("Acerto (%):Q", title="Acerto no corte (%)"),
                    tooltip=[
                        alt.Tooltip("Bloco:Q", title=f"Bloco ({int(block_size)} jogos)"),
                        alt.Tooltip("Data de Corte:T", title="Data de corte", format="%d/%m/%Y"),
                        "Modelo:N",
                        alt.Tooltip("Acerto (%):Q", format=".2f"),
                        alt.Tooltip("Acertos:Q", title="Acertos no corte"),
                        alt.Tooltip("Total:Q", title="Total no corte"),
                    ],
                )
                .properties(height=170)
            )
            chart = base.facet(
                facet=alt.Facet("Modelo:N", title=None),
                columns=2,
            ).resolve_scale(y="independent")
            st.altair_chart(chart, use_container_width=True)

except Exception as exc:
    st.error(f"Erro inesperado ao montar ranking por mercado: {exc}")
