"""Tela dedicada de ranking de performance por mercado e modelo."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis import build_model_ranking_by_market
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
        # Mantém os jogos em ordem cronológica crescente para consistência visual.
        df_finished = df_finished.sort_values("date", ascending=True, na_position="last")

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

    # Como a lista está crescente, os "últimos" jogos ficam no final.
    selected_game_keys = set(jogos_unicos[-int(qtd_jogos):])
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

except Exception as exc:
    st.error(f"Erro inesperado ao montar ranking por mercado: {exc}")
