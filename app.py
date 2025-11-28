"""Módulo principal da aplicação Placar Guru."""
import streamlit as st


# Exemplo de código para remover o cabeçalho usando CSS

# Ocultar barra superior (header)

hide_streamlit_style = """ 
<style>
header {visibility: hidden;} 
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


hide_header_style = """
<style>
.st-emotion-cache-1jicfl2 {
    display: none !important;
}
</style>
"""

reduce_header_height_style = """
<style>
.block-container {
    padding-top: 1rem;
}
</style>
"""

st.markdown(hide_header_style, unsafe_allow_html=True)


import streamlit.components.v1 as components
import json
import uuid
import base64
import pandas as pd
import altair as alt
from datetime import timedelta, date, datetime
from zoneinfo import ZoneInfo  # Python 3.9+
from string import Template

# --- novos imports para baixar a release e controlar cache/tempo ---
from email.utils import parsedate_to_datetime


# Importa funções e constantes do utils.py
from utils import (
    fetch_release_file, RELEASE_URL, load_data, FRIENDLY_COLS,
    tournament_label, market_label, norm_status_key, fmt_score_pred_text,
    status_label, FINISHED_TOKENS,
)
from styles import inject_custom_css, apply_altair_theme, chart_tokens





# ============================
# Configuração da página
# ============================
st.set_page_config(
    layout="wide",
    page_title="Placar Guru",
    initial_sidebar_state="collapsed",
)

# Estado inicial: Light por padrão
st.session_state.setdefault("pg_dark_mode", False)
dark_mode = bool(st.session_state["pg_dark_mode"])

# --- Estilos mobile-first + cores e tema dos gráficos ---
inject_custom_css(dark_mode)
apply_altair_theme(dark_mode)
chart_theme = chart_tokens(dark_mode)

# Barra superior inspirada no modelo (Futebol + Data Science Placar Guru)
st.markdown(
    """
    <div class="pg-topbar">
      <div class="pg-topbar__brand">
        <div class="pg-logo" aria-hidden="true" role="presentation">
          <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
            <path class="pg-logo-shield" d="M12 10h40l-3.2 32.5L32 56 15.2 42.5 12 10Z" />
            <rect class="pg-logo-chart" x="18" y="30" width="6" height="14" rx="2" />
            <rect class="pg-logo-chart" x="26" y="26" width="6" height="18" rx="2" />
            <rect class="pg-logo-chart" x="34" y="34" width="6" height="10" rx="2" />
            <rect class="pg-logo-chart" x="42" y="22" width="6" height="22" rx="2" />
            <circle class="pg-logo-ball" cx="34.5" cy="21.5" r="8" />
            <path class="pg-logo-ball" d="M28 20c2.6 1.2 5.2 1.2 7.8 0l2.7 3.2-2.2 4.8h-4.8L29.2 23z" fill="none" />
            <circle class="pg-logo-glow" cx="34.5" cy="21.5" r="3.4" />
          </svg>
        </div>
        <div>
          <p class="pg-eyebrow">Futebol + Data Science</p>
          <div class="pg-appname">Futebol + Data Science Placar Guru</div>
        </div>
      </div>
      <div class="pg-topbar__actions">
        <span class="pg-chip">Insights preditivos em tempo real</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Toggle manual de modo mobile (controle explícito para layout responsivo)
col_m1, col_m2 = st.columns([1.2, 4])
with col_m1:
    modo_mobile = st.toggle("📱 Mobile", value=True)
with col_m2:
    st.markdown(
        """
        <div class="pg-subhead">
          <span class="pg-chip ghost">Altere para desktop para ver a grade</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

from reporting import generate_pdf_report
from ui_components import filtros_ui, display_list_view, is_guru_highlight, render_glassy_table
from analysis import prepare_accuracy_chart_data, get_best_model_by_market, create_summary_pivot_table, calculate_kpis
# ============================
# Exibição amigável
# ============================
def apply_friendly_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica formatações e traduções em um DataFrame para exibição amigável."""
    out = df.copy()

    # Tradução de mercados + fallback amigável
    for col in ["bet_suggestion", "goal_bet_suggestion", "result_predicted"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda v: market_label(v))

    # Resultado Final (só quando finished)
    def _fmt_score(row):
        if norm_status_key(row.get("status","")) in FINISHED_TOKENS:
            rh, ra = row.get("result_home"), row.get("result_away")
            if pd.notna(rh) and pd.notna(ra):
                try:
                    return f"{int(rh)}-{int(ra)}"
                except Exception:
                    return f"{rh}-{ra}"
            return "N/A"
        return ""
    if {"status", "result_home", "result_away"}.issubset(out.columns):
        out["final_score"] = out.apply(_fmt_score, axis=1)

    if "status" in out.columns:
        out["status"] = out["status"].apply(status_label)
    if "tournament_id" in out.columns:
        out["tournament_id"] = out["tournament_id"].apply(tournament_label)

    # Placar Previsto com fallback amigável
    if "score_predicted" in out.columns:
        out["score_predicted"] = out["score_predicted"].apply(lambda x: fmt_score_pred_text(x))

    # Nova Previsão BTTS
    if "btts_suggestion" in out.columns:
        out["btts_prediction"] = out["btts_suggestion"].apply(market_label, default="-")

    return out.rename(columns=FRIENDLY_COLS)


# ============================
# App principal
# ============================
try:
    # 1) Baixa o Excel da release
    content, etag, last_mod = fetch_release_file(RELEASE_URL)

    # 2) Converte Last-Modified em datetime na sua TZ
    tz_sp = ZoneInfo("America/Sao_Paulo")
    if last_mod:
        try:
            last_update_dt = parsedate_to_datetime(last_mod).astimezone(tz_sp)
        except Exception:
            last_update_dt = datetime.now(tz=tz_sp)
    else:
        last_update_dt = datetime.now(tz=tz_sp)

    df = load_data(content)

    if "init_from_url" not in st.session_state:
        st.session_state.init_from_url = True
        raw_model = st.query_params.get("model", ["Combo"])
        st.session_state.model_init_raw = list(raw_model) if isinstance(raw_model, list) else [raw_model]

    if df.empty:
        st.error("O arquivo `PrevisaoJogos.xlsx` está vazio ou não pôde ser lido.")
    else:
        # Filtros principais no sidebar (incluindo campeonatos)
        flt = filtros_ui(df, modo_mobile)
        tournaments_sel, models_sel, teams_sel = flt["tournaments_sel"], flt["models_sel"], flt["teams_sel"]
        bet_sel, goal_sel = flt["bet_sel"], flt["goal_sel"]
        selected_date_range, sel_h, sel_d, sel_a = flt["selected_date_range"], flt["sel_h"], flt["sel_d"], flt["sel_a"]
        q_team = flt["q_team"]

        use_list_view = True if modo_mobile else st.checkbox("Usar visualização em lista (mobile)", value=False)

        # Máscara combinada (sem status)
        final_mask = pd.Series(True, index=df.index)

        # ▶️ Aplicar filtro global de campeonatos
        if tournaments_sel and "tournament_id" in df.columns:
            final_mask &= df["tournament_id"].isin(tournaments_sel)

        if models_sel and "model" in df.columns:
            final_mask &= df["model"].isin(models_sel)

        if teams_sel and {"home", "away"}.issubset(df.columns):
            home_ser = df["home"].astype(str)
            away_ser = df["away"].astype(str)
            final_mask &= (home_ser.isin(teams_sel) | away_ser.isin(teams_sel))

        if q_team and {"home", "away"}.issubset(df.columns):
            q = str(q_team).strip()
            if q:
                home_contains = df["home"].astype(str).str.contains(q, case=False, na=False)
                away_contains = df["away"].astype(str).str.contains(q, case=False, na=False)
                final_mask &= (home_contains | away_contains)

        if bet_sel and "bet_suggestion" in df.columns:
            final_mask &= df["bet_suggestion"].astype(str).isin([str(x) for x in bet_sel])

        if goal_sel and "goal_bet_suggestion" in df.columns:
            final_mask &= df["goal_bet_suggestion"].astype(str).isin([str(x) for x in goal_sel])

        if selected_date_range and isinstance(selected_date_range, (list, tuple)) and len(selected_date_range) == 2 and "date" in df.columns:
            start_date, end_date = selected_date_range
            final_mask &= (df["date"].dt.date.between(start_date, end_date)) | (df["date"].isna())

        if "odds_H" in df.columns:
            final_mask &= ((df["odds_H"] >= sel_h[0]) & (df["odds_H"] <= sel_h[1])) | (df["odds_H"].isna())
        if "odds_D" in df.columns:
            final_mask &= ((df["odds_D"] >= sel_d[0]) & (df["odds_D"] <= sel_d[1])) | (df["odds_D"].isna())
        if "odds_A" in df.columns:
            final_mask &= ((df["odds_A"] >= sel_a[0]) & (df["odds_A"] <= sel_a[1])) | (df["odds_A"].isna())

        df_filtered = df[final_mask]

        # Abas Agendados x Finalizados (KPIs só em Finalizados)
        if df_filtered.empty:
            st.warning("Nenhum dado corresponde aos filtros atuais.")
        else:
            status_norm_all = df_filtered["status"].astype(str).map(norm_status_key) if "status" in df_filtered.columns else pd.Series("", index=df_filtered.index)

            df_ag  = df_filtered[status_norm_all != "finished"]
            df_fin = df_filtered[status_norm_all == "finished"]
            df_fin_full = df_fin.copy()
            status_view = st.radio(
                "Dashboard por status",
                options=["🗓️ Agendados", "✅ Finalizados"],
                horizontal=True,
                key="pg_status_view",
            )

            if status_view.startswith("🗓️"):
                curr_df = df_ag
                curr_label = "Agendados"
            else:
                curr_df = df_fin
                curr_label = "Finalizados"

            total_games = len(curr_df)
            highlight_count = int(curr_df.apply(is_guru_highlight, axis=1).sum()) if not curr_df.empty else 0
            tourn_count = int(curr_df["tournament_id"].nunique()) if (not curr_df.empty and "tournament_id" in curr_df.columns) else 0
            today_count = 0
            if not curr_df.empty and "date" in curr_df.columns and curr_df["date"].notna().any():
                today_count = int(curr_df[curr_df["date"].dt.date == date.today()].shape[0])

            metrics_df = pd.DataFrame()
            acc_result = acc_bet = 0.0
            if curr_label == "Finalizados" and not curr_df.empty:
                metrics_df = calculate_kpis(curr_df, multi_model=False)
                _res = metrics_df.loc[metrics_df["Métrica"] == "Resultado"]
                _bet = metrics_df.loc[metrics_df["Métrica"] == "Sugestão de Aposta"]
                if not _res.empty:
                    acc_result = float(_res.iloc[0]["Acerto (%)"])
                if not _bet.empty:
                    acc_bet = float(_bet.iloc[0]["Acerto (%)"])

            st.markdown(
                f"""
                <div class="pg-hero">
                  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; flex-wrap:wrap;">
                    <div>
                      <div class="pg-meta">Dashboard — {curr_label}</div>
                      <h2 style="margin:4px 0;">Informações essenciais por status</h2>
                      <div class="text-muted" style="font-size:13px;">Atualizado em {last_update_dt.strftime('%d/%m %H:%M')} (hora local)</div>
                    </div>
                    <span class="badge">Tema: {'Dark' if dark_mode else 'Light'}</span>
                  </div>
                  <div class="pg-kpi-grid">
                    <div class="pg-kpi">
                      <div class="label">Total no filtro</div>
                      <div class="value">{total_games}</div>
                      <div class="delta">{today_count} hoje</div>
                    </div>
                    <div class="pg-kpi">
                      <div class="label">Sugestão Guru</div>
                      <div class="value">{highlight_count}</div>
                      <div class="delta">Prob > 60% & Odd > 1.20</div>
                    </div>
                    <div class="pg-kpi">
                      <div class="label">Torneios</div>
                      <div class="value">{tourn_count}</div>
                      <div class="delta">Filtro ativo</div>
                    </div>
                    <div class="pg-kpi">
                      <div class="label">Acurácia (finalizados)</div>
                      <div class="value">{acc_result:.1f}%</div>
                      <div class="delta">Sugestões {acc_bet:.1f}%</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ---------- Padrão: FINALIZADOS = últimos 3 dias + ordenação desc ----------
            has_date_col = ("date" in df.columns) and df["date"].notna().any()
            if has_date_col:
                _min_all = df["date"].dropna().min().date()
                _max_all = df["date"].dropna().max().date()
            else:
                _min_all = _max_all = None

            user_gave_range = (
                isinstance(selected_date_range, (list, tuple)) and
                len(selected_date_range) == 2 and
                has_date_col and
                selected_date_range != (_min_all, _max_all)
            )

            if not user_gave_range and ("date" in df_fin.columns) and df_fin["date"].notna().any():
                _today = date.today()
                _start = _today - timedelta(days=3)
                _end   = _today
                df_fin_recent = df_fin[df_fin["date"].dt.date.between(_start, _end) | df_fin["date"].isna()]
                # Se o recorte automático de 3 dias zerar a lista, volta para o conjunto completo
                if not df_fin_recent.empty:
                    df_fin = df_fin_recent
                else:
                    df_fin = df_fin_full

            if "date" in df_fin.columns:
                df_fin = df_fin.sort_values("date", ascending=False, na_position="last")
            # --------------------------------------------------------------------------

            # --- VISÃO POR STATUS (seleção acima) ---
            if status_view.startswith("🗓️"):
                if df_ag.empty:
                    st.info("Sem jogos agendados neste recorte.")
                else:
                    pdf_data = generate_pdf_report(df_ag)
                    st.download_button(
                        label="Exportar para PDF",
                        data=pdf_data,
                        file_name=f"relatorio_jogos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                    )
                    if use_list_view:
                        display_list_view(df_ag)
                    else:
                        cols_to_show = [
                            "date", "home", "away", "tournament_id", "model",
                            "status", "result_predicted", "score_predicted",
                            "bet_suggestion", "goal_bet_suggestion",
                            "btts_suggestion", "odds_H", "odds_D", "odds_A",
                            "result_home", "result_away"
                        ]
                        existing_cols = [c for c in cols_to_show if c in df_ag.columns]
                        render_glassy_table(
                            apply_friendly_for_display(df_ag[existing_cols]),
                            caption="Jogos agendados",
                        )

            else:
                if df_fin.empty:
                    st.info("Sem jogos finalizados neste recorte.")
                else:
                    st.markdown("<div class='pg-hide-card'>", unsafe_allow_html=True)
                    hide_info, hide_toggle = st.columns([4, 1.3])
                    with hide_info:
                        st.markdown(
                            f"""
                            <div class="pg-hide-copy">
                              <p class="pg-eyebrow">Lista de jogos finalizados</p>
                              <div class="pg-hide-title">Exibir ou ocultar rapidamente</div>
                              <p class="pg-hide-desc">Use quando quiser focar apenas nas métricas e gráficos de desempenho.</p>
                              <div class="pg-hide-chips">
                                <span class="pg-chip ghost">{len(df_fin)} jogos</span>
                                <span class="pg-chip ghost">{'Visão lista' if use_list_view else 'Visão tabela'}</span>
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with hide_toggle:
                        hide_games = st.toggle(
                            "Ocultar lista de jogos",
                            key="pg_hide_fin_list",
                            value=st.session_state.get("pg_hide_fin_list", False),
                            help="Oculte a listagem para ver somente KPIs, gráficos e tabelas avançadas.",
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

                    if not hide_games:
                        if use_list_view:
                            display_list_view(df_fin)
                        else:
                            cols_to_show = [
                                "date", "home", "away", "tournament_id", "model",
                                "status", "result_predicted", "score_predicted",
                                "bet_suggestion", "goal_bet_suggestion",
                                "btts_suggestion", "odds_H", "odds_D", "odds_A",
                                "result_home", "result_away"
                            ]
                            existing_cols = [c for c in cols_to_show if c in df_fin.columns]
                            render_glassy_table(
                                apply_friendly_for_display(df_fin[existing_cols]),
                                caption="Jogos finalizados",
                            )

                    # ---------- KPIs e gráfico por modelo (apenas finalizados) ----------
                    rh = df_fin.get("result_home", pd.Series(index=df_fin.index, dtype="float"))
                    ra = df_fin.get("result_away", pd.Series(index=df_fin.index, dtype="float"))
                    mask_valid = rh.notna() & ra.notna()

                    # Códigos reais H/D/A
                    real_code = pd.Series(index=df_fin.index, dtype="object")
                    real_code.loc[mask_valid & (rh > ra)] = "H"
                    real_code.loc[mask_valid & (rh == ra)] = "D"
                    real_code.loc[mask_valid & (rh < ra)] = "A"

                    selected_models = list(df_fin["model"].dropna().unique()) if "model" in df_fin.columns else []
                    multi_model = len(selected_models) > 1

                    metrics_df = calculate_kpis(df_fin, multi_model)

                    def _metric_value(name: str) -> float:
                        row = metrics_df[metrics_df["Métrica"] == name]
                        if row.empty:
                            return 0.0
                        try:
                            return float(row["Acerto (%)"].iloc[0])
                        except Exception:
                            return 0.0

                    avg_accuracy = round(metrics_df["Acerto (%)"].mean(), 1) if not metrics_df.empty else 0.0
                    resultado_acc = _metric_value("Resultado")
                    aposta_acc = _metric_value("Sugestão de Aposta")
                    gols_acc = _metric_value("Sugestão de Gols")
                    total_finished = len(df_fin)
                    total_tourn_fin = df_fin["tournament_id"].nunique() if "tournament_id" in df_fin.columns else 0
                    markets_covered = df_fin["bet_suggestion"].nunique() if "bet_suggestion" in df_fin.columns else 0
                    guru_destaques = int(df_fin.apply(is_guru_highlight, axis=1).sum()) if not df_fin.empty else 0

                    st.markdown("<div class='pg-stats-stack'>", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class="pg-stats-section">
                          <div class="pg-stats-header">
                            <div>
                              <p class="pg-eyebrow">Sessão de estatísticas</p>
                              <h3 style="margin: 0;">Insights dos jogos finalizados</h3>
                              <p class="pg-stats-desc">KPIs premium, gráficos e melhores modelos alinhados ao protótipo.</p>
                            </div>
                            <div class="pg-stats-tags">
                              <span class="pg-chip ghost">Status: Finalizados</span>
                              <span class="pg-chip ghost">Tema {"Dark" if dark_mode else "Light"}</span>
                            </div>
                          </div>
                          <div class="pg-stat-grid">
                            <div class="pg-stat-card">
                              <p class="pg-stat-label">Acurácia média</p>
                              <div class="pg-stat-value">{avg_accuracy:.1f}%</div>
                              <p class="pg-stat-foot">Base {total_finished} partidas avaliadas</p>
                            </div>
                            <div class="pg-stat-card">
                              <p class="pg-stat-label">Resultado previsto</p>
                              <div class="pg-stat-value">{resultado_acc:.1f}%</div>
                              <p class="pg-stat-foot">{guru_destaques} destaques neon Guru</p>
                            </div>
                            <div class="pg-stat-card">
                              <p class="pg-stat-label">Sugestão de aposta</p>
                              <div class="pg-stat-value">{aposta_acc:.1f}%</div>
                              <p class="pg-stat-foot">{markets_covered} mercados cobertos</p>
                            </div>
                            <div class="pg-stat-card">
                              <p class="pg-stat-label">Gols / BTTS</p>
                              <div class="pg-stat-value">{gols_acc:.1f}%</div>
                              <p class="pg-stat-foot">{total_tourn_fin} campeonatos avaliados</p>
                            </div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        """
                        <div class='pg-stats-panel'>
                          <div class="pg-stats-header">
                            <div>
                              <p class="pg-eyebrow">Gráfico de acertos</p>
                              <h4 style="margin:0;">Precisão por métrica</h4>
                              <p class="pg-stats-desc">Compare modelos, mercados e a taxa de acerto consolidada.</p>
                            </div>
                            <div class="pg-stats-tags">
                              <span class="pg-chip ghost">Interativo</span>
                              <span class="pg-chip ghost">Ordene por coluna</span>
                            </div>
                          </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Sempre exibir a tabela de precisão por métrica
                    if not metrics_df.empty:
                        metric_order = [
                            "Resultado",
                            "Sugestão de Aposta",
                            "Sugestão Combo",
                            "Sugestão de Gols",
                            "Ambos Marcam",
                        ]
                        metrics_df_display = metrics_df.copy()
                        metrics_df_display["Métrica"] = pd.Categorical(
                            metrics_df_display["Métrica"],
                            categories=metric_order,
                            ordered=True,
                        )
                        metrics_df_display = metrics_df_display.sort_values("Métrica")
                        render_glassy_table(metrics_df_display, caption="Precisão por métrica")

                    if multi_model:
                        # Gráfico de barras agrupadas por modelo
                        if not metrics_df.empty:
                            chart = (
                                alt.Chart(metrics_df)
                                .mark_bar()
                                .encode(
                                    x=alt.X('Métrica:N', title=''),
                                    y=alt.Y('Acerto (%):Q', scale=alt.Scale(domain=[0,100])),
                                    color=alt.Color('Modelo:N', scale=alt.Scale(range=chart_theme["palette"])),
                                    xOffset='Modelo:N',
                                    tooltip=['Modelo:N','Métrica:N','Acertos:Q','Total Avaliado:Q', alt.Tooltip('Acerto (%):Q', format='.1f')]
                                )
                                .properties(height=240 if modo_mobile else 280)
                            )
                            text = (
                                alt.Chart(metrics_df)
                                .mark_text(dy=-8, color=chart_theme["text"])
                                .encode(
                                    x='Métrica:N',
                                    y='Acerto (%):Q',
                                    detail='Modelo:N',
                                    text=alt.Text('Acerto (%):Q', format='.1f'),
                                    color=alt.Color('Modelo:N', scale=alt.Scale(range=chart_theme["palette"]))
                                )
                            )
                            st.markdown("<div class='pg-chart-card'>", unsafe_allow_html=True)
                            st.altair_chart(chart + text, use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        # Definir as métricas que queremos exibir e a ordem
                        metric_order = ["Resultado", "Sugestão de Aposta", "Sugestão Combo", "Sugestão de Gols", "Ambos Marcam"]

                        # Criar colunas dinamicamente
                        cols = st.columns(len(metric_order))

                        # Iterar e exibir cada métrica
                        for i, metric_name in enumerate(metric_order):
                            metric_data = metrics_df[metrics_df["Métrica"] == metric_name]
                            if not metric_data.empty:
                                acc = metric_data["Acerto (%)"].iloc[0]
                                hits = metric_data["Acertos"].iloc[0]
                                total = metric_data["Total Avaliado"].iloc[0]
                                short_name = metric_name.replace(" (Sugestão)", "").replace(" (Prob)", "")
                                cols[i].metric(short_name, f"{acc}%", f"{hits}/{total}")

                        chart = alt.Chart(metrics_df).mark_bar(color=chart_theme["accent"]).encode(
                            x=alt.X('Métrica:N', title=''),
                            y=alt.Y('Acerto (%):Q', scale=alt.Scale(domain=[0, 100])),
                            tooltip=['Métrica:N', 'Acertos:Q', 'Total Avaliado:Q', alt.Tooltip('Acerto (%):Q', format='.1f')]
                        ).properties(height=220 if modo_mobile else 260)
                        text = alt.Chart(metrics_df).mark_text(dy=-8, color=chart_theme["text"]).encode(
                            x='Métrica:N',
                            y='Acerto (%):Q',
                            text=alt.Text('Acerto (%):Q', format='.1f')
                        )
                        st.markdown("<div class='pg-chart-card'>", unsafe_allow_html=True)
                        st.altair_chart(chart + text, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                                        # --- Gráficos de linha de acurácia por dia (aninhados no cartão de desempenho) ---
                    accuracy_data = prepare_accuracy_chart_data(df_fin)
                    if not accuracy_data.empty:
                        tournaments = sorted(accuracy_data['Campeonato'].unique())
                        muted = "#94a3b8" if dark_mode else "#475569"
                        shadow = "0 18px 48px rgba(0,0,0,0.35)" if dark_mode else "0 18px 48px rgba(0,0,0,0.12)"

                        panel_style_tpl = Template(
                            """
                            <style>
                              :root {
                                --bg: ${bg};
                                --panel: ${panel};
                                --stroke: ${stroke};
                                --text: ${text};
                                --muted: ${muted};
                                --primary: ${primary};
                                --primary-2: ${primary2};
                                --shadow: ${shadow};
                              }
                              body { margin:0; background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif; }
                              .pg-stats-panel { border: 1px solid var(--stroke); border-radius: 16px; padding: 14px; background: linear-gradient(135deg, color-mix(in srgb, var(--panel) 92%, transparent), color-mix(in srgb, var(--panel) 84%, transparent)); box-shadow: var(--shadow); }
                              .pg-stats-header { display:flex; justify-content: space-between; gap: 12px; align-items: center; flex-wrap: wrap; }
                              .pg-stats-desc { color: var(--muted); margin: 4px 0 0 0; }
                              .pg-eyebrow { margin:0; font-size:11px; letter-spacing:0.08em; text-transform:uppercase; color: var(--muted); }
                              .pg-stats-tags { display:flex; gap:8px; align-items:center; flex-wrap: wrap; }
                              .pg-chip { border-radius: 999px; padding: 6px 10px; font-weight: 700; border: 1px solid var(--stroke); color: var(--text); background: color-mix(in srgb, var(--panel) 90%, transparent); }
                              .pg-chip.ghost { background: color-mix(in srgb, var(--panel) 85%, transparent); color: var(--muted); }
                              .pg-chart-grid { display:grid; grid-template-columns: 1fr; gap: 12px; margin-top: 10px; }
                              .pg-chart-cluster { border: 1px solid color-mix(in srgb, var(--stroke) 75%, transparent); border-radius: 16px; padding: 12px; background: linear-gradient(140deg, color-mix(in srgb, var(--panel) 92%, transparent), color-mix(in srgb, var(--panel) 84%, transparent)); box-shadow: var(--shadow); }
                              .pg-chart-cluster__head { display:flex; justify-content: space-between; align-items: center; gap:12px; margin-bottom: 8px; }
                              .pg-chart-card { border: 1px solid var(--stroke); border-radius: 18px; padding: 10px 12px; background: linear-gradient(140deg, color-mix(in srgb, var(--panel) 88%, transparent), color-mix(in srgb, var(--panel) 96%, transparent)); box-shadow: var(--shadow); position: relative; overflow: hidden; }
                              .pg-chart-card::before { content: ''; position: absolute; inset: 0; background: radial-gradient(circle at 12% 16%, color-mix(in srgb, var(--primary) 18%, transparent), transparent 32%); pointer-events: none; opacity: 0.8; }
                              .pg-chart-card > * { position: relative; z-index: 1; }
                              .pg-chart-card__title { font-weight: 800; font-size: 15px; margin: 4px 0 10px; color: var(--text); }
                              .pg-chart-card .vega-embed, .pg-chart-card .vega-embed * { color: var(--text); }
                              .vega-actions { display: none; }
                              svg text { fill: var(--text); }
                            </style>
                            """
                        )

                        panel_style = panel_style_tpl.substitute(
                            bg=chart_theme["background"],
                            panel=chart_theme["panel"],
                            stroke=chart_theme["stroke"],
                            text=chart_theme["text"],
                            muted=muted,
                            primary=chart_theme["accent"],
                            primary2=chart_theme["palette"][1],
                            shadow=shadow,
                        )

                        st.markdown(panel_style, unsafe_allow_html=True)

                        # Monta um único painel HTML com os gráficos em acordeões, renderizados via vega-embed
                        panel_blocks = []
                        embed_scripts = []

                        for idx, tourn in enumerate(tournaments):
                            tourn_data = accuracy_data[accuracy_data['Campeonato'] == tourn]
                            models = sorted(tourn_data['Modelo'].unique())

                            chart_cards = []
                            spec_list = []
                            id_list = []

                            for m_idx, model in enumerate(models):
                                model_data = tourn_data[tourn_data['Modelo'] == model].copy()
                                model_data['Data'] = pd.to_datetime(model_data['Data']).dt.strftime('%Y-%m-%d')
                                chart = (
                                    alt.Chart(model_data)
                                    .mark_line(point=True)
                                    .encode(
                                        x=alt.X('Data:T', title='Dia'),
                                        y=alt.Y('Taxa de Acerto (%):Q', scale=alt.Scale(domain=[0, 100]), title='Taxa de Acerto'),
                                        color=alt.Color('Métrica:N', title="Métrica de Aposta", scale=alt.Scale(range=chart_theme["palette"]))
                                        ,
                                        tooltip=['Data:T', 'Métrica:N', alt.Tooltip('Taxa de Acerto (%):Q', format='.1f')]
                                    )
                                    .properties(
                                        height=240 if modo_mobile else 280,
                                        width=780 if modo_mobile else 1180,
                                        background=chart_theme.get("plot_bg", "transparent"),
                                    )
                                )

                                slot_id = f"pg-chart-{idx}-{m_idx}"
                                id_list.append(slot_id)
                                spec_list.append(chart.to_dict())
                                chart_cards.append(
                                    f"""
                                    <div class='pg-chart-card nested'>
                                      <div class='pg-chart-card__title'>Modelo: {model}</div>
                                      <div id='{slot_id}' class='pg-chart-slot'></div>
                                    </div>
                                    """
                                )

                            chart_cards_html = "".join(chart_cards)
                            panel_blocks.append(
                                f"""
                                <details class="pg-details pg-chart-accordion" open>
                                  <summary>
                                    <div>
                                      <div class="pg-details-title">{tourn}</div>
                                      <div class="pg-details-hint">Campeonato • {len(models)} modelo(s) • métricas múltiplas</div>
                                    </div>
                                    <div class="pg-stats-tags">
                                      <span class="pg-chip ghost">{len(models)} modelo(s)</span>
                                      <span class="pg-chip ghost">Diário</span>
                                    </div>
                                  </summary>
                                  <div class="pg-details-body">
                                    <div class="pg-chart-grid">{chart_cards_html}</div>
                                  </div>
                                </details>
                                """
                            )

                            embed_scripts.append(
                                {
                                    "ids": id_list,
                                    "specs": spec_list,
                                }
                            )

                        palette = {
                            "bg": "#0b1224" if dark_mode else "#f8fafc",
                            "panel": "#0f172a" if dark_mode else "#ffffff",
                            "glass": "rgba(255,255,255,0.04)" if dark_mode else "rgba(255,255,255,0.65)",
                            "stroke": "#1f2937" if dark_mode else "#e2e8f0",
                            "text": "#e2e8f0" if dark_mode else "#0f172a",
                            "muted": "#94a3b8" if dark_mode else "#475569",
                            "primary": "#60a5fa" if dark_mode else "#2563eb",
                            "primary2": "#22d3ee",
                            "neon": "#bfff3b",
                            "shadow": "0 20px 60px rgba(0,0,0,0.35)" if dark_mode else "0 20px 60px rgba(0,0,0,0.12)",
                        }

                        panel_css_tpl = Template(
                            """
                        <style>
                          :root {
                            --bg: ${bg};
                            --panel: ${panel};
                            --glass: ${glass};
                            --stroke: ${stroke};
                            --text: ${text};
                            --muted: ${muted};
                            --primary: ${primary};
                            --primary-2: ${primary2};
                            --neon: ${neon};
                            --shadow: ${shadow};
                          }
                          body {
                            margin: 0;
                            background: transparent;
                            color: var(--text);
                            font-family: 'Inter', system-ui, -apple-system, sans-serif;
                          }
                          .pg-stats-panel {
                            border: 1px solid var(--stroke);
                            border-radius: 16px;
                            padding: 14px;
                            background: linear-gradient(135deg, color-mix(in srgb, var(--panel) 92%, transparent), color-mix(in srgb, var(--panel) 84%, transparent));
                            box-shadow: var(--shadow);
                          }
                          .pg-stats-header { display: flex; justify-content: space-between; gap: 12px; align-items: center; flex-wrap: wrap; }
                          .pg-eyebrow { margin: 0; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
                          .pg-stats-desc { color: var(--muted); margin: 4px 0 0 0; }
                          .pg-stats-tags { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
                          .pg-chip {
                            display: inline-flex;
                            align-items: center;
                            gap: 6px;
                            padding: 6px 10px;
                            border-radius: 999px;
                            border: 1px solid color-mix(in srgb, var(--stroke) 70%, transparent);
                            background: color-mix(in srgb, var(--panel) 94%, transparent);
                            color: var(--text);
                            font-weight: 700;
                            font-size: 12px;
                            letter-spacing: -0.01em;
                          }
                          .pg-chip.ghost { background: color-mix(in srgb, var(--panel) 88%, transparent); color: var(--muted); }
                          .pg-chart-grid { display: grid; grid-template-columns: 1fr; gap: 12px; margin-top: 8px; }
                          .pg-chart-card {
                            position: relative;
                            border: 1px solid color-mix(in srgb, var(--stroke) 75%, transparent);
                            border-radius: 16px;
                            padding: 12px;
                            background: linear-gradient(140deg, color-mix(in srgb, var(--panel) 92%, transparent), color-mix(in srgb, var(--panel) 84%, transparent));
                            box-shadow: var(--shadow);
                            overflow: hidden;
                          }
                          .pg-chart-card::before {
                            content: '';
                            position: absolute;
                            inset: -30% 40% auto auto;
                            width: 60%; height: 60%;
                            background: radial-gradient(circle at 30% 30%, color-mix(in srgb, var(--primary) 26%, transparent), transparent 62%);
                            opacity: 0.3;
                          }
                          .pg-chart-card__title { font-weight: 800; font-size: 15px; margin: 4px 0 10px; color: var(--text); }
                          .pg-chart-slot .vega-embed { background: color-mix(in srgb, var(--panel) 96%, transparent); border-radius: 12px; padding: 4px; box-shadow: inset 0 1px 0 rgba(255,255,255,0.05); }
                          .pg-chart-slot .vega-actions { display: none; }
                          .pg-details {
                            margin-top: 12px;
                            border: 1px solid var(--stroke);
                            border-radius: 14px;
                            background: linear-gradient(120deg, color-mix(in srgb, var(--panel) 92%, transparent), color-mix(in srgb, var(--panel) 98%, transparent));
                            box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
                            overflow: hidden;
                          }
                          .pg-details summary {
                            list-style: none;
                            cursor: pointer;
                            display: flex;
                            align-items: center;
                            justify-content: space-between;
                            gap: 10px;
                            padding: 10px 12px;
                            color: var(--text);
                            font-weight: 700;
                          }
                          .pg-details summary::-webkit-details-marker { display: none; }
                          .pg-details summary:focus { outline: 2px solid color-mix(in srgb, var(--primary) 50%, transparent); outline-offset: 2px; }
                          .pg-details-body { padding: 0 12px 12px 12px; }
                          .pg-chart-accordion { margin-top: 8px; border: 1px solid color-mix(in srgb, var(--stroke) 72%, transparent); }
                          .pg-chart-accordion summary > div:first-child { display: grid; gap: 4px; }
                        </style>
                        """
                        )
                        panel_css = panel_css_tpl.safe_substitute(palette)

                        panel_html_tpl = Template(
                            """
                        ${css}
                        <div class='pg-stats-panel pg-desempenho-panel'>
                          <div class="pg-stats-header">
                            <div>
                              <p class="pg-eyebrow">Desempenho Diário por Campeonato e Métrica</p>
                              <h4 style="margin:0;">Evolução de acerto por torneio e modelo</h4>
                              <p class="pg-stats-desc">Acompanhe a curva diária de precisão para cada campeonato, modelo e métrica.</p>
                            </div>
                            <div class="pg-stats-tags">
                              <span class="pg-chip ghost">Linhas</span>
                              <span class="pg-chip ghost">Interativo</span>
                            </div>
                          </div>
                          <div class='pg-chart-grid nested'>
                            ${blocks}
                          </div>
                        </div>
                        <script src="https://cdn.jsdelivr.net/npm/vega@5.25.0"></script>
                        <script src="https://cdn.jsdelivr.net/npm/vega-lite@5.16.3"></script>
                        <script src="https://cdn.jsdelivr.net/npm/vega-embed@6.24.0"></script>
                        <script>
                          const pgRenderQueues = ${queues};
                          pgRenderQueues.forEach(({ids, specs}) => {
                            ids.forEach((slotId, i) => {
                              const target = document.getElementById(slotId);
                              if (target) {
                                vegaEmbed(target, specs[i], {
                                  actions: false,
                                  renderer: 'svg'
                                });
                              }
                            });
                          });
                        </script>
                        """
                        )
                        panel_html = panel_html_tpl.safe_substitute(
                            css=panel_css,
                            blocks="".join(panel_blocks),
                            queues=json.dumps(embed_scripts),
                        )

                        # Altura calculada: cabeçalho + acordeões (cada modelo ~320px)
                        est_height = 260 + sum([140 + len(entry["ids"]) * (320 if modo_mobile else 360) for entry in embed_scripts])
                        est_height = min(est_height, 1400) if modo_mobile else min(est_height, 1800)

                        # Largura expansiva para ocupar a área útil (iframe padrão do Streamlit é 700px)
                        panel_width = 1100 if modo_mobile else 1500

                        components.html(panel_html, height=est_height, scrolling=True, width=panel_width)
                    else:
                        st.info("Não há dados suficientes para gerar os gráficos de desempenho diário.")

# --- Tabela de Melhor Modelo por Campeonato e Mercado ---
                    best_model_data = get_best_model_by_market(df_fin.copy())
                    if not best_model_data.empty:
                        summary_pivot_table = create_summary_pivot_table(best_model_data)

                        tbl1_id = f"tbl-best-model-{uuid.uuid4().hex[:8]}"
                        tbl2_id = f"tbl-best-summary-{uuid.uuid4().hex[:8]}"

                        table1_html = best_model_data.to_html(
                            index=False,
                            classes="display compact pg-glass-table",
                            border=0,
                            table_id=tbl1_id,
                        )
                        table2_html = summary_pivot_table.to_html(
                            index=False,
                            classes="display compact pg-glass-table",
                            border=0,
                            table_id=tbl2_id,
                        )

                        table1_csv_b64 = base64.b64encode(
                            best_model_data.to_csv(index=False).encode("utf-8")
                        ).decode("utf-8")
                        table2_csv_b64 = base64.b64encode(
                            summary_pivot_table.to_csv(index=False).encode("utf-8")
                        ).decode("utf-8")

                        palette = {
                            "bg": "#0b1224" if st.session_state.get("pg_dark_mode", False) else "#f8fafc",
                            "panel": "#0f172a" if st.session_state.get("pg_dark_mode", False) else "#ffffff",
                            "glass": "rgba(255,255,255,0.08)" if st.session_state.get("pg_dark_mode", False) else "rgba(255,255,255,0.65)",
                            "stroke": "#1f2937" if st.session_state.get("pg_dark_mode", False) else "#e2e8f0",
                            "text": "#e2e8f0" if st.session_state.get("pg_dark_mode", False) else "#0f172a",
                            "muted": "#94a3b8" if st.session_state.get("pg_dark_mode", False) else "#475569",
                            "primary": "#60a5fa" if st.session_state.get("pg_dark_mode", False) else "#2563eb",
                            "primary2": "#22d3ee",
                            "neon": "#bfff3b",
                            "shadow": "0 20px 60px rgba(0,0,0,0.35)" if st.session_state.get("pg_dark_mode", False) else "0 20px 60px rgba(0,0,0,0.12)",
                        }

                        
                        best_panel_tpl = Template(
                            """
                            <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css" />
                            <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css" />
                            <style>
                              :root {
                                --bg: ${bg};
                                --panel: ${panel};
                                --glass: ${glass};
                                --stroke: ${stroke};
                                --text: ${text};
                                --muted: ${muted};
                                --primary: ${primary};
                                --primary-2: ${primary2};
                                --neon: ${neon};
                                --shadow: ${shadow};
                                --accent: ${primary};
                                --white: #ffffff;
                                --text-strong: ${text};
                              }
                              body { background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, -apple-system,sans-serif; }
                              .pg-eyebrow { text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; font-size: 11px; color: var(--muted); margin: 0 0 4px 0; }
                              .pg-stats-desc { color: var(--muted); margin: 4px 0 0 0; }
                              .pg-chip { display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px; border:1px solid color-mix(in srgb, var(--stroke) 80%, transparent); background: color-mix(in srgb, var(--panel) 88%, transparent); color: var(--text); font-weight:700; font-size:12px; }
                              .pg-chip.ghost { background: color-mix(in srgb, var(--panel) 75%, transparent); color: var(--muted); }
                              .pg-stats-panel { border:1px solid var(--stroke); border-radius:16px; padding:16px; background: linear-gradient(135deg, color-mix(in srgb, var(--panel) 92%, transparent), color-mix(in srgb, var(--panel) 84%, transparent)); box-shadow: var(--shadow); }
                              .pg-best-panel { padding: 18px; background: linear-gradient(135deg, color-mix(in srgb, var(--panel) 92%, transparent), color-mix(in srgb, var(--panel) 86%, transparent)); border: 1px solid var(--stroke); border-radius: 18px; box-shadow: 0 12px 60px color-mix(in srgb, var(--shadow) 15%, transparent), 0 1px 0 color-mix(in srgb, var(--white) 10%, transparent) inset; }
                              .pg-best-panel .pg-stats-header { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; margin-bottom:12px; }
                              .pg-best-panel .pg-table-stack { display:flex; flex-direction:column; gap:16px; }
                              .pg-best-panel .pg-table-block { padding:12px; background: color-mix(in srgb, var(--panel) 94%, transparent); border:1px solid color-mix(in srgb, var(--stroke) 82%, transparent); border-radius:14px; box-shadow: 0 6px 36px color-mix(in srgb, var(--shadow) 14%, transparent), 0 1px 0 color-mix(in srgb, var(--white) 12%, transparent) inset; }
                              .pg-best-panel .pg-table-block h5 { color: var(--text); }
                              /* Base table styling even if DataTables fails */
                              .pg-best-panel table { width:100%; border-collapse:separate; border-spacing:0; background: color-mix(in srgb, var(--panel) 97%, transparent); border:1px solid color-mix(in srgb, var(--stroke) 82%, transparent); border-radius: 12px; overflow:hidden; box-shadow: inset 0 1px 0 color-mix(in srgb, var(--white) 14%, transparent); }
                              .pg-best-panel thead th { background: linear-gradient(135deg, color-mix(in srgb, var(--panel) 90%, transparent), color-mix(in srgb, var(--panel) 78%, transparent)); color: var(--text-strong); font-weight:700; border-bottom:1px solid color-mix(in srgb, var(--stroke) 85%, transparent); cursor:pointer; padding:10px 12px; }
                              .pg-best-panel tbody td { padding:10px 12px; color: var(--text); border:none; }
                              .pg-best-panel tbody tr:nth-child(odd) { background: color-mix(in srgb, var(--panel) 96%, transparent); }
                              .pg-best-panel tbody tr:nth-child(even) { background: color-mix(in srgb, var(--panel) 92%, transparent); }
                              .pg-best-panel tbody tr:hover { background: color-mix(in srgb, var(--accent) 10%,var(--panel)); box-shadow: 0 10px 30px color-mix(in srgb, var(--shadow) 18%, transparent); }
                              /* DataTables overrides */
                              .pg-best-panel .dataTables_wrapper { background: color-mix(in srgb, var(--panel) 94%, transparent); padding: 10px 10px 14px; border-radius: 14px; border:1px solid color-mix(in srgb, var(--stroke) 80%, transparent); box-shadow: inset 0 1px 0 color-mix(in srgb, var(--white) 10%, transparent); }
                              .pg-best-panel table.dataTable { width:100% !important; border-collapse:separate !important; border-spacing:0 !important; background: color-mix(in srgb, var(--panel) 97%, transparent) !important; border:1px solid color-mix(in srgb, var(--stroke) 82%, transparent) !important; border-radius: 12px; overflow:hidden; box-shadow: inset 0 1px 0 color-mix(in srgb, var(--white) 14%, transparent); }
                              .pg-best-panel table.dataTable thead th { background: linear-gradient(135deg, color-mix(in srgb, var(--panel) 90%, transparent), color-mix(in srgb, var(--panel) 78%, transparent)) !important; color: var(--text-strong) !important; font-weight:700 !important; border-bottom:1px solid color-mix(in srgb, var(--stroke) 85%, transparent) !important; cursor:pointer; }
                              .pg-best-panel table.dataTable tbody tr.pg-row-odd { background: color-mix(in srgb, var(--panel) 96%, transparent) !important; }
                              .pg-best-panel table.dataTable tbody tr.pg-row-even { background: color-mix(in srgb, var(--panel) 92%, transparent) !important; }
                              .pg-best-panel table.dataTable tbody tr:hover { background: color-mix(in srgb, var(--accent) 10%,var(--panel)) !important; box-shadow: 0 10px 30px color-mix(in srgb, var(--shadow) 18%, transparent); }
                              .pg-best-panel table.dataTable tbody td, .pg-best-panel table.dataTable thead th { padding:10px 12px !important; color: var(--text) !important; border:none !important; }
                              .pg-best-panel .dataTables_scroll { border-radius: 12px; overflow:hidden; border:1px solid color-mix(in srgb, var(--stroke) 80%, transparent); box-shadow: inset 0 1px 0 color-mix(in srgb, var(--white) 12%,transparent); }
                              .pg-best-panel .dataTables_wrapper .dataTables_filter, .pg-best-panel .dataTables_wrapper .dataTables_info, .pg-best-panel .dataTables_wrapper .dataTables_paginate { display:none; }
                              .pg-best-panel .dt-buttons { display:flex; flex-wrap:wrap; gap:8px; margin: 4px 0 10px; }
                              .pg-best-panel .dt-button { background: color-mix(in srgb, var(--panel) 80%, transparent); color: var(--text); border:1px solid color-mix(in srgb, var(--stroke) 78%, transparent); border-radius: 10px; padding: 6px 10px; font-weight:700; box-shadow: 0 4px 16px color-mix(in srgb, var(--shadow) 16%, transparent); transition: transform 150ms ease, box-shadow 150ms ease; }
                              .pg-best-panel .dt-button:hover { transform: translateY(-1px); box-shadow: 0 10px 24px color-mix(in srgb, var(--shadow) 22%, transparent); color: var(--accent); }
                              .pg-best-panel .pg-dt-fallback { display:flex; gap:12px; flex-wrap:wrap; margin-top:10px; }
                              .pg-best-panel .pg-dt-fallback a { text-decoration:none; color: var(--accent); font-weight:700; padding:6px 10px; border-radius:10px; background: color-mix(in srgb, var(--panel) 86%, transparent); border:1px solid color-mix(in srgb, var(--stroke) 78%, transparent); }
                              .pg-best-panel .pg-dt-active thead th { color: var(--accent) !important; }
                              .pg-best-panel table.dataTable.pg-glass-table { width:100% !important; }
                              table.dataTable.display > tbody > tr:nth-child(odd) > * { background: transparent !important; }
                              table.dataTable.display > tbody > tr:nth-child(even) > * { background: transparent !important; }
                            </style>
                            <div class='pg-stats-panel pg-best-panel'>
                              <div class="pg-stats-header">
                                <div>
                                  <p class="pg-eyebrow">Modelos vencedores</p>
                                  <h4 style="margin:0;">Sessão de Melhor Modelo</h4>
                                  <p class="pg-stats-desc">Compare o desempenho por campeonato e mercado com tabelas ordenáveis no visual glassy.</p>
                                </div>
                                <div class="pg-stats-tags">
                                  <span class="pg-chip ghost">Interativo</span>
                                  <span class="pg-chip ghost">Ordenável</span>
                                </div>
                              </div>
                              <div class="pg-table-stack">
                                <div class="pg-table-block">
                                  <p class="pg-eyebrow">Visão detalhada</p>
                                  <h5 style="margin:0;">Melhor Modelo por Campeonato e Mercado</h5>
                                  ${table1}
                                  <div class="pg-dt-fallback">
                                    <a href="data:text/csv;base64,${t1csv}" download="melhor-modelo-campeonato.csv">⬇ Exportar CSV</a>
                                  </div>
                                </div>
                                <div class="pg-table-block">
                                  <p class="pg-eyebrow">Resumo</p>
                                  <h5 style="margin:0;">Resumo do Melhor Modelo por Mercado</h5>
                                  ${table2}
                                  <div class="pg-dt-fallback">
                                    <a href="data:text/csv;base64,${t2csv}" download="resumo-melhor-modelo-mercado.csv">⬇ Exportar CSV</a>
                                  </div>
                                </div>
                              </div>
                            </div>
                            <script src="https://cdn.jsdelivr.net/npm/jquery@3.6.0/dist/jquery.min.js"></script>
                            <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
                            <script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
                            <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.html5.min.js"></script>
                            <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
                            <script>
                              const pgFallbackSort = (tblId) => {
                                const el = document.getElementById(tblId);
                                if (!el) return;
                                const getCell = (row, idx) => row.children[idx]?.innerText?.toLowerCase?.() || '';
                                Array.from(el.querySelectorAll('th')).forEach((th, idx) => {
                                  th.addEventListener('click', () => {
                                    const rows = Array.from(el.querySelectorAll('tbody tr'));
                                    const asc = !th.classList.contains('pg-sort-asc');
                                    el.querySelectorAll('th').forEach(h => h.classList.remove('pg-sort-asc','pg-sort-desc'));
                                    th.classList.add(asc ? 'pg-sort-asc' : 'pg-sort-desc');
                                    rows.sort((a,b)=>{
                                      const va = getCell(asc ? a : b, idx);
                                      const vb = getCell(asc ? b : a, idx);
                                      return va.localeCompare(vb, 'pt', {numeric:true});
                                    }).forEach(r => el.querySelector('tbody').appendChild(r));
                                  });
                                });
                              };
                              const pgInitDt = (tblId) => {
                                const el = document.getElementById(tblId);
                                if (!el || !window.jQuery) { pgFallbackSort(tblId); return; }
                                const $tbl = window.jQuery(el);
                                if ($tbl.length === 0 || !window.jQuery.fn?.dataTable) { pgFallbackSort(tblId); return; }
                                if (window.jQuery.fn.dataTable.isDataTable($tbl)) {
                                  $tbl.DataTable().destroy();
                                }
                                $tbl.DataTable({
                                  paging:false,
                                  searching:false,
                                  info:false,
                                  ordering:true,
                                  order: [],
                                  scrollX:true,
                                  autoWidth:false,
                                  stripeClasses: ['pg-row-odd','pg-row-even'],
                                  dom: '<"pg-dt-top"B>t',
                                  buttons: [
                                    { extend:'copyHtml5', text:'📋 Copiar', className:'pg-dt-btn'},
                                    { extend:'csvHtml5', text:'⬇ CSV', className:'pg-dt-btn'},
                                    { extend:'excelHtml5', text:'⬇ Excel', className:'pg-dt-btn'}
                                  ]
                                });
                                $tbl.on('click', 'th', () => $tbl.addClass('pg-dt-active'));
                              };
                              (function ensureReady(){
                                const ready = window.jQuery && window.jQuery.fn && window.jQuery.fn.dataTable && window.jQuery.fn.dataTable.Buttons;
                                if (!ready) { setTimeout(ensureReady, 120); return; }
                                pgInitDt('${tbl1_id}');
                                pgInitDt('${tbl2_id}');
                              })();
                            </script>
                            """
                        )
                        panel_html = best_panel_tpl.safe_substitute(
                            bg=palette["bg"],
                            panel=palette["panel"],
                            glass=palette["glass"],
                            stroke=palette["stroke"],
                            text=palette["text"],
                            muted=palette["muted"],
                            primary=palette["primary"],
                            primary2=palette["primary2"],
                            neon=palette["neon"],
                            shadow=palette["shadow"],
                            table1=table1_html,
                            table2=table2_html,
                            t1csv=table1_csv_b64,
                            t2csv=table2_csv_b64,
                        )

                        est_height = 420 + (len(best_model_data) + len(summary_pivot_table)) * 24
                        est_height = max(520, min(est_height, 1400))
                        est_width = 1200 if modo_mobile else 1500

                        components.html(panel_html, height=est_height, width=est_width, scrolling=True)
                    else:
                        st.info("Não há dados suficientes para gerar a tabela de melhores modelos.")

        # --- Rodapé: Última Atualização + alternância de tema (agora no rodapé) ---
        st.markdown('<hr style="border: 0; border-top: 1px solid #1f2937; margin: 1rem 0 0.5rem 0;" />', unsafe_allow_html=True)

        fcol1, fcol2 = st.columns([3, 2])
        with fcol1:
            st.markdown(
                f"""
                <div style=\"color:#9CA3AF; font-size:0.95rem;\">
                  <strong>Última atualização:</strong> {last_update_dt.strftime('%d/%m/%Y %H:%M')}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with fcol2:
            st.toggle(
                "🌗 Modo escuro",
                key="pg_dark_mode",
                value=st.session_state.get("pg_dark_mode", False),
                help="Alterne para ver o tema escuro premium. Modo padrão: Light.",
            )

        # Botão para forçar atualização (limpa o cache de dados e re-executa o app)
        if st.button("🔄 Atualizar agora"):
            st.cache_data.clear()
            st.rerun()

except FileNotFoundError:
    st.error("FATAL: `PrevisaoJogos.xlsx` não encontrado.")
except Exception as e:
    st.error(f"Erro inesperado: {e}")
