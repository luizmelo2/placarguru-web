"""Módulo principal da aplicação Placar Guru."""
import streamlit as st
import pandas as pd
import altair as alt
from datetime import timedelta, date, datetime
from zoneinfo import ZoneInfo  # Python 3.9+

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
          <div class="pg-appname">Placar Guru - Futebol + Data Science</div>
        </div>
      </div>
      <div class="pg-topbar__nav">
        <!--<span class="pg-tab active">Dashboard</span>-->
      </div>
      <div class="pg-topbar__actions">
        <!--<span class="pg-chip">Insights preditivos em tempo real</span>-->
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
          <span class="pg-chip ghost">Layout mobile-first ativo</span>
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
        # ----------------- FILTRO GLOBAL NO TOPO: CAMPEONATOS (opção C2 com box) -----------------
        if "tournament_id" in df.columns and df["tournament_id"].notna().any():
            tourn_opts_all = sorted(df["tournament_id"].dropna().unique().tolist())
        else:
            tourn_opts_all = []

        st.session_state.setdefault("sel_tournaments", list(tourn_opts_all))

        # mantém apenas os válidos se a lista mudar
        valid_sel = [t for t in st.session_state.sel_tournaments if t in tourn_opts_all]
        if valid_sel != st.session_state.sel_tournaments:
            st.session_state.sel_tournaments = valid_sel

        # se ficou vazio e há opções, volta para "todos"
        if not st.session_state.sel_tournaments and tourn_opts_all:
            st.session_state.sel_tournaments = list(tourn_opts_all)

        with st.container():
            st.markdown('<div class="tourn-box">', unsafe_allow_html=True)
            hcol1, hcol2 = st.columns([5,3])
            with hcol1:
                st.markdown('<div class="tourn-title">🏆 Campeonatos</div>', unsafe_allow_html=True)
            with hcol2:
                csel_all, cclear = st.columns(2)
                with csel_all:
                    if st.button("Selecionar Todos", use_container_width=True):
                        st.session_state.sel_tournaments = list(tourn_opts_all)
                        st.rerun()
                with cclear:
                    if st.button("Limpar", use_container_width=True):
                        st.session_state.sel_tournaments = []
                        st.rerun()

            # sem "default" — valor vem do session_state (key)
            st.multiselect(
                label="",
                options=tourn_opts_all,
                key="sel_tournaments",
                format_func=tournament_label,
                placeholder="Selecione um ou mais campeonatos…",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        top_tournaments_sel = list(st.session_state.sel_tournaments)
        # -----------------------------------------------------------------------------------------

        # Filtros adicionais (sem torneios; eles vêm do topo)
        flt = filtros_ui(df, modo_mobile, tournaments_sel_external=top_tournaments_sel)
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
                    if not st.checkbox("Ocultar lista de jogos", value=False):
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
                    if multi_model:
                        render_glassy_table(metrics_df, caption="Acurácia por modelo")

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

                    # --- Gráficos de linha de acurácia por dia (todos aninhados dentro da sessão) ---
                    accuracy_data = prepare_accuracy_chart_data(df_fin)
                    with st.container():
                        # Painel único para manter todos os campeonatos e modelos aninhados na mesma sessão
                        st.markdown(
                            """
                            <div class='pg-stats-panel'>
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
                            """,
                            unsafe_allow_html=True,
                        )

                        if not accuracy_data.empty:
                            tournaments = sorted(accuracy_data['Campeonato'].unique())
                            cols_per_row = 1 if modo_mobile else 2

                            for row_start in range(0, len(tournaments), cols_per_row):
                                row_tournaments = tournaments[row_start:row_start + cols_per_row]
                                row_cols = st.columns(len(row_tournaments))

                                for col_idx, tourn in enumerate(row_tournaments):
                                    tourn_data = accuracy_data[accuracy_data['Campeonato'] == tourn]
                                    models = sorted(tourn_data['Modelo'].unique())

                                    with row_cols[col_idx]:
                                        st.markdown(
                                            f"""
                                            <div class="pg-chart-cluster">
                                              <div class="pg-chart-cluster__head">
                                                <div>
                                                  <p class="pg-eyebrow">Campeonato</p>
                                                  <h4 style="margin:0;">{tourn}</h4>
                                                </div>
                                                <div class="pg-stats-tags">
                                                  <span class="pg-chip ghost">{len(models)} modelo(s)</span>
                                                  <span class="pg-chip ghost">Métricas múltiplas</span>
                                                </div>
                                              </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                                        for model in models:
                                            model_data = tourn_data[tourn_data['Modelo'] == model]

                                            line_chart = alt.Chart(model_data).mark_line(point=True).encode(
                                                x=alt.X('Data:T', title='Dia'),
                                                y=alt.Y('Taxa de Acerto (%):Q', scale=alt.Scale(domain=[0, 100]), title='Taxa de Acerto'),
                                                color=alt.Color('Métrica:N', title="Métrica de Aposta", scale=alt.Scale(range=chart_theme["palette"])) ,
                                                tooltip=['Data:T', 'Métrica:N', alt.Tooltip('Taxa de Acerto (%):Q', format='.1f')]
                                            ).properties(
                                                height=280
                                            )
                                            st.markdown(
                                                f"""
                                                <div class='pg-chart-card nested'>
                                                  <div class='pg-chart-card__title'>Modelo: {model}</div>
                                                """,
                                                unsafe_allow_html=True,
                                            )
                                            st.altair_chart(line_chart, use_container_width=True)
                                            st.markdown("</div>", unsafe_allow_html=True)

                                        st.markdown("</div>", unsafe_allow_html=True)

                            # fecha grid + painel
                            st.markdown("</div></div>", unsafe_allow_html=True)
                        else:
                            st.markdown("</div></div>", unsafe_allow_html=True)
                            st.info("Não há dados suficientes para gerar os gráficos de desempenho diário.")
                    # --- Tabela de Melhor Modelo por Campeonato e Mercado ---
                    st.markdown("<div class='pg-stats-panel'>", unsafe_allow_html=True)
                    st.subheader("Melhor Modelo por Campeonato e Mercado")
                    best_model_data = get_best_model_by_market(df_fin.copy())
                    if not best_model_data.empty:
                        render_glassy_table(best_model_data, caption="Melhor modelo por campeonato e mercado")

                        st.subheader("Resumo do Melhor Modelo por Mercado")
                        summary_pivot_table = create_summary_pivot_table(best_model_data)
                        render_glassy_table(summary_pivot_table, caption="Resumo por mercado")
                    else:
                        st.info("Não há dados suficientes para gerar a tabela de melhores modelos.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

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
