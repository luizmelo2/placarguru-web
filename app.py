"""Módulo principal da aplicação Placar Guru."""
import streamlit as st
import streamlit.errors as st_errors

import json
import uuid
import base64
import pandas as pd
import altair as alt
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+
from string import Template

# --- novos imports para baixar a release e controlar cache/tempo ---
from email.utils import parsedate_to_datetime


# Importa funções e constantes do utils.py
from utils import (
    fetch_release_file, RELEASE_URL, load_data,
    tournament_label, norm_status_key, apply_friendly_for_display,
)
from styles import inject_custom_css, apply_altair_theme, chart_tokens
from state import (
    detect_viewport_width,
    MOBILE_BREAKPOINT,
    set_filter_state,
    get_filter_state,
    TABLE_COLUMN_PRESETS,
    DEFAULT_TABLE_DENSITY,
    reset_filters,
    FilterState,
    recent_date_window,
)

# ============================
# Configuração da página
# ============================


st.set_page_config(
    layout="wide",
    page_title="Previsões",
    initial_sidebar_state="expanded",
)

_SECRET_ERROR_CLASSES = tuple(
    err
    for err in (
        getattr(st_errors, "StreamlitSecretNotFoundError", None),
        FileNotFoundError,
        KeyError,
    )
    if err is not None
)

from ui_components import (
    filtros_ui,
    display_list_view,
    is_guru_highlight,
    build_guru_highlight_data,
    render_glassy_table,
    render_app_header,
    render_chip,
    render_custom_navigation,
    inject_topbar_branding,
    inject_header_fix_css,
    render_mobile_quick_filters,
)
import streamlit.components.v1 as components

# Garante que o nome da página principal apareça como "Previsões" na navegação lateral customizada
render_custom_navigation()

# Aplica correção do header somente se estiver habilitada em secrets ou query string
try:
    force_header_patch = bool(st.secrets.get("force_header_patch", False))
except _SECRET_ERROR_CLASSES:
    force_header_patch = False
force_header_patch = force_header_patch or st.query_params.get("force_header", ["0"])[0] == "1"
inject_header_fix_css(force_header_patch)


def init_theme_state() -> None:
    """Garante o estado inicial do tema."""

    st.session_state.setdefault("pg_dark_mode", False)


init_theme_state()
dark_mode = False

# --- Estilos mobile-first + cores e tema dos gráficos ---
inject_custom_css(dark_mode)
apply_altair_theme(dark_mode)
chart_theme = chart_tokens(dark_mode)
inject_topbar_branding()

topbar_placeholder = st.empty()
viewport_width = detect_viewport_width()
modo_mobile = viewport_width < MOBILE_BREAKPOINT
st.session_state["pg_mobile_auto"] = modo_mobile
auto_view_label = f"Visual: {'mobile' if modo_mobile else 'desktop'} ({viewport_width}px)"

from reporting import generate_pdf_report
from analysis import prepare_accuracy_chart_data, get_best_model_by_market, create_summary_pivot_table, calculate_kpis

def _apply_filters(
    df: pd.DataFrame,
    tournaments_sel: list,
    models_sel: list,
    teams_sel: list,
    q_team: str,
    bet_sel: list,
    goal_sel: list,
    selected_date_range: tuple,
    sel_h: tuple,
    sel_d: tuple,
    sel_a: tuple,
    guru_only: bool,
    guru_flag_all: pd.Series,
) -> pd.DataFrame:
    """Aplica filtros de seleção e retorna o DataFrame filtrado."""
    final_mask = pd.Series(True, index=df.index)

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

    if (
        selected_date_range
        and isinstance(selected_date_range, (list, tuple))
        and len(selected_date_range) == 2
        and "date" in df.columns
    ):
        start_date, end_date = selected_date_range
        final_mask &= (df["date"].dt.date.between(start_date, end_date)) | (df["date"].isna())

    if "odds_H" in df.columns:
        final_mask &= ((df["odds_H"] >= sel_h[0]) & (df["odds_H"] <= sel_h[1])) | (df["odds_H"].isna())
    if "odds_D" in df.columns:
        final_mask &= ((df["odds_D"] >= sel_d[0]) & (df["odds_D"] <= sel_d[1])) | (df["odds_D"].isna())
    if "odds_A" in df.columns:
        final_mask &= ((df["odds_A"] >= sel_a[0]) & (df["odds_A"] <= sel_a[1])) | (df["odds_A"].isna())

    if guru_only:
        final_mask &= guru_flag_all

    return df[final_mask]


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
        guru_only = flt.get("guru_only", False)
        selected_date_range, sel_h, sel_d, sel_a = flt["selected_date_range"], flt["sel_h"], flt["sel_d"], flt["sel_a"]
        q_team = flt.get("search_query", "")
        tournament_opts = flt.get("tournament_opts", [])
        min_date, max_date = flt.get("min_date"), flt.get("max_date")

        st.session_state.setdefault("pg_list_view_pref", True)
        if modo_mobile:
            st.session_state["pg_list_view_pref"] = True
        else:
            st.session_state["pg_list_view_pref"] = st.checkbox(
                "Usar visualização em lista (mobile)",
                value=bool(st.session_state.get("pg_list_view_pref", True)),
                help="Mantém a listagem em cards mesmo no desktop quando preferir.",
            )

        st.session_state["pg_table_density"] = DEFAULT_TABLE_DENSITY
        table_density = DEFAULT_TABLE_DENSITY

        use_list_view = bool(st.session_state.get("pg_list_view_pref", modo_mobile))

        shared_state = get_filter_state()
        defaults = flt.get("defaults", {})
        if modo_mobile:
            tournaments_sel, selected_date_range, q_team_input = render_mobile_quick_filters(
                tournaments_sel=tournaments_sel,
                tournament_opts=tournament_opts,
                selected_date_range=selected_date_range,
                min_date=min_date,
                max_date=max_date,
                shared_state=shared_state,
            )
            q_team_input = q_team_input or ""

            clear_col, chips_col = st.columns([1, 2])
            with clear_col:
                if st.button("Limpar recorte", use_container_width=True):
                    cleared = reset_filters(defaults)
                    st.session_state["pg_table_density"] = DEFAULT_TABLE_DENSITY
                    tournaments_sel = cleared.tournaments_sel or []
                    models_sel = cleared.models_sel or []
                    teams_sel = cleared.teams_sel or []
                    bet_sel = cleared.bet_sel or []
                    goal_sel = cleared.goal_sel or []
                    selected_date_range = cleared.selected_date_range
                    sel_h, sel_d, sel_a = cleared.sel_h, cleared.sel_d, cleared.sel_a
                    q_team_input = cleared.search_query
                    shared_state = cleared
            with chips_col:
                st.markdown(
                    f"<div class='pg-chip ghost' aria-hidden='true'>Ativos agora: {shared_state.active_count}</div>",
                    unsafe_allow_html=True,
                )

            q_team = q_team_input
            set_filter_state(shared_state)

        guru_flags_all, guru_scope_all, guru_flag_all = build_guru_highlight_data(df)

        model_unique = df["model"].nunique() if "model" in df.columns else 0
        tournaments_sel_count = (
            [] if (tournaments_sel and len(tournaments_sel) == len(tournament_opts)) else tournaments_sel
        )
        models_sel_count = (
            [] if (models_sel and model_unique and len(models_sel) == model_unique) else models_sel
        )

        current_state = FilterState(
            tournaments_sel=tournaments_sel_count,
            models_sel=models_sel_count,
            teams_sel=teams_sel,
            bet_sel=bet_sel,
            goal_sel=goal_sel,
            selected_date_range=selected_date_range,
            sel_h=sel_h,
            sel_d=sel_d,
            sel_a=sel_a,
            search_query=q_team or "",
            guru_only=guru_only,
        )
        active_filters = current_state.active_count

        df_filtered = _apply_filters(
            df=df,
            tournaments_sel=tournaments_sel,
            models_sel=models_sel,
            teams_sel=teams_sel,
            q_team=q_team,
            bet_sel=bet_sel,
            goal_sel=goal_sel,
            selected_date_range=selected_date_range,
            sel_h=sel_h,
            sel_d=sel_d,
            sel_a=sel_a,
            guru_only=guru_only,
            guru_flag_all=guru_flag_all,
        )
        df_filtered = df_filtered.assign(
            guru_highlight_scope=guru_scope_all.loc[df_filtered.index],
            guru_highlight=guru_flag_all.loc[df_filtered.index],
        )

        if guru_only and not df_filtered.empty and not guru_flags_all.empty:
            filtered_flags = guru_flags_all.loc[df_filtered.index]
            col_map = {
                "Resultado": "result_predicted",
                "Sugestão": "bet_suggestion",
                "Gols": "goal_bet_suggestion",
                "Ambos Marcam": "btts_suggestion",
            }
            for label, col in col_map.items():
                if col in df_filtered.columns and label in filtered_flags.columns:
                    df_filtered.loc[~filtered_flags[label], col] = pd.NA

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

            export_disabled = curr_df.empty
            export_state_label = "Exportação pronta" if not export_disabled else "Aplique filtros para habilitar PDF"
            live_messages = [
                export_state_label,
                f"Última atualização {last_update_dt.strftime('%d/%m %H:%M')}",
                f"{active_filters} filtros ativos",
                auto_view_label,
            ]
            header_html = render_app_header(live_messages=live_messages)
            with topbar_placeholder.container():
                brand_col, action_col = st.columns([4, 1.4])
                brand_col.markdown(header_html, unsafe_allow_html=True)


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
                _start, _end = recent_date_window(days=3)
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
                    st.markdown(
                        f"""
                        <div class="pg-hide-copy">
                          <p class="pg-eyebrow">Lista de jogos agendados</p>
                          <div class="pg-hide-title">Jogos futuros no recorte</div>
                          <div class="pg-hide-chips">
                            <span class="pg-chip ghost">{len(df_ag)} jogos</span>
                            <span class="pg-chip ghost">{'Visão lista' if use_list_view else 'Visão tabela'}</span>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if use_list_view:
                        display_list_view(df_ag, hide_missing=guru_only)
                    else:
                        preset_key = "compact" if table_density == "compact" and modo_mobile else ("mobile" if modo_mobile else "desktop")
                        cols_to_show = TABLE_COLUMN_PRESETS.get(preset_key, TABLE_COLUMN_PRESETS["desktop"])
                        existing_cols = [
                            c for c in cols_to_show
                            if c in df_ag.columns and (df_ag[c].notna().any() if c.startswith("odds") else True)
                        ]
                        render_glassy_table(
                            apply_friendly_for_display(df_ag[existing_cols]),
                            caption="Jogos agendados",
                            density=table_density,
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

                    if hide_games:
                        st.markdown(
                            "<div class='pg-chip ghost'>Lista oculta. Desative o toggle acima para reexibir os jogos.</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        if use_list_view:
                            display_list_view(df_fin, hide_missing=guru_only)
                        else:
                            preset_key = "compact" if table_density == "compact" and modo_mobile else ("mobile" if modo_mobile else "desktop")
                            cols_to_show = TABLE_COLUMN_PRESETS.get(preset_key, TABLE_COLUMN_PRESETS["desktop"])
                            existing_cols = [
                                c for c in cols_to_show
                                if c in df_fin.columns and (df_fin[c].notna().any() if c.startswith("odds") else True)
                            ]
                            render_glassy_table(
                                apply_friendly_for_display(df_fin[existing_cols]),
                                caption="Jogos finalizados",
                                density=table_density,
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
                    overall_metrics = calculate_kpis(df_fin, False)

                    metric_order = [
                        "Resultado",
                        "Sugestão de Aposta",
                        "Sugestão Combo",
                        "Sugestão de Gols",
                        "Ambos Marcam",
                    ]

                    def _metric_stats_for(metrics_frame: pd.DataFrame) -> dict[str, tuple[float, int, int]]:
                        def _extract(metric: str) -> tuple[float, int, int]:
                            row = metrics_frame[metrics_frame["Métrica"] == metric]
                            if row.empty:
                                return 0.0, 0, 0
                            acc = float(row["Acerto (%)"].iloc[0]) if pd.notna(row["Acerto (%)"].iloc[0]) else 0.0
                            hits = int(row["Acertos"].iloc[0]) if pd.notna(row["Acertos"].iloc[0]) else 0
                            total = int(row["Total Avaliado"].iloc[0]) if pd.notna(row["Total Avaliado"].iloc[0]) else 0
                            return acc, hits, total

                        return {metric: _extract(metric) for metric in metric_order}

                    overall_stats = _metric_stats_for(overall_metrics)

                    campeonatos_stats: list[tuple[str, dict[str, tuple[float, int, int]]]] = []
                    if "tournament_id" in df_fin.columns:
                        for tourn_id in sorted(df_fin["tournament_id"].dropna().unique()):
                            tourn_df = df_fin[df_fin["tournament_id"] == tourn_id]
                            if tourn_df.empty:
                                continue
                            tourn_metrics = calculate_kpis(tourn_df, False)
                            campeonatos_stats.append((tournament_label(tourn_id), _metric_stats_for(tourn_metrics)))

                    def _render_stat_row(title: str, desc: str, stats: dict[str, tuple[float, int, int]], tag: str | None = None):
                        st.markdown(f"**{title}**")
                        st.caption(desc)
                        if tag:
                            st.caption(f"[{tag}]")
                        cols = st.columns(len(metric_order))
                        for col, metric in zip(cols, metric_order):
                            acc, hits, total = stats.get(metric, (0.0, 0, 0))
                            col.metric(metric, f"{acc:.1f}%", f"{hits} / {total} jogos")

                    st.markdown("### Insights dos jogos finalizados")
                    st.caption("Percentual de acertos e volume avaliado por mercado.")
                    _render_stat_row(
                        "Consolidado",
                        "Resumo geral dos jogos finalizados.",
                        overall_stats,
                    )

                    for tourn_name, stats in campeonatos_stats:
                        st.divider()
                        _render_stat_row(
                            tourn_name,
                            "Precisão por mercado para jogos finalizados do campeonato.",
                            stats,
                            tag="Recorte do campeonato",
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

                        st.markdown(
                            """
                            <div class='pg-stats-panel'>
                              <div class="pg-stats-header">
                                <div>
                                  <p class="pg-eyebrow">Modelos vencedores</p>
                                  <h4 style="margin:0;">Sessão de Melhor Modelo</h4>
                                  <p class="pg-stats-desc">Compare o desempenho por campeonato e mercado no mesmo layout das demais tabelas glassy.</p>
                                </div>
                              </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        render_glassy_table(
                            best_model_data,
                            caption="Melhor Modelo por Campeonato e Mercado",
                        )
                        render_glassy_table(
                            summary_pivot_table,
                            caption="Resumo do Melhor Modelo por Mercado",
                        )
                    else:
                        st.info("Não há dados suficientes para gerar a tabela de melhores modelos.")

        export_data = generate_pdf_report(curr_df) if not export_disabled else b""
        st.download_button(
            label="Exportar recorte para PDF",
            data=export_data,
            file_name=f"placar_guru_{curr_label.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            help="Gere um PDF com o recorte atual. Habilita ao aplicar filtros que retornem jogos.",
            disabled=export_disabled,
        )
        if export_disabled:
            st.caption("O botão é habilitado ao aplicar filtros que retornem jogos neste recorte.")

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
            st.empty()

        # Botão para forçar atualização (limpa o cache de dados e re-executa o app)
        if st.button("🔄 Atualizar agora"):
            st.cache_data.clear()
            st.rerun()

except FileNotFoundError:
    st.error("FATAL: `PrevisaoJogos.xlsx` não encontrado.")
except Exception as e:
    st.error(f"Erro inesperado: {e}")
