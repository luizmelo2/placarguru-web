"""Módulo principal da aplicação Placar Guru."""
import streamlit as st
import streamlit.errors as st_errors

import os
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
from state import (
    detect_viewport_width,
    MOBILE_BREAKPOINT,
    set_filter_state,
    get_filter_state,
    TABLE_COLUMN_PRESETS,
    DEFAULT_TABLE_DENSITY,
    reset_filters,
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
    render_glassy_table,
    render_app_header,
    render_chip,
    render_custom_navigation,
    inject_topbar_branding,
    inject_header_fix_css,
    render_mobile_quick_filters,
)

def _query_param_first(key: str, default: str = "") -> str:
    """Lê query param aceitando str ou lista de valores."""
    raw = st.query_params.get(key, default)
    if isinstance(raw, list):
        return str(raw[0]) if raw else str(default)
    return str(raw)


def _months_back_config(default: int = 2) -> int:
    """Obtém recorte padrão em meses via env/secrets com fallback seguro."""

    raw_env = os.getenv("PG_MONTHS_BACK", "").strip()
    if raw_env:
        try:
            return max(int(raw_env), 0)
        except Exception:
            return default
    try:
        return max(int(st.secrets.get("months_back", default)), 0)
    except _SECRET_ERROR_CLASSES:
        return default
    except Exception:
        return default


# Garante que o nome da página principal apareça como "Previsões" na navegação lateral customizada
render_custom_navigation()

# Aplica correção do header somente se estiver habilitada em secrets ou query string
try:
    force_header_patch = bool(st.secrets.get("force_header_patch", False))
except _SECRET_ERROR_CLASSES:
    force_header_patch = False
force_header_patch = force_header_patch or _query_param_first("force_header", "0") == "1"
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
from insights_service import METRIC_ORDER, metric_stats_for, build_tournament_stats
from dashboard_service import FilterParams, apply_dashboard_filters

# ============================
# Exibição amigável
# ============================
def apply_friendly_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica formatações e traduções em um DataFrame para exibição amigável."""
    out = df.copy()

    def _translate_markets(frame: pd.DataFrame) -> pd.DataFrame:
        for col in ["bet_suggestion", "goal_bet_suggestion", "result_predicted"]:
            if col in frame.columns:
                frame[col] = frame[col].apply(market_label)
        return frame

    def _compute_final_score(frame: pd.DataFrame) -> pd.DataFrame:
        if {"status", "result_home", "result_away"}.issubset(frame.columns):
            status_norm = frame["status"].astype(str).map(norm_status_key)
            finished_mask = status_norm.isin(FINISHED_TOKENS)
            rh = pd.to_numeric(frame["result_home"], errors="coerce")
            ra = pd.to_numeric(frame["result_away"], errors="coerce")
            has_score = rh.notna() & ra.notna()
            frame["final_score"] = ""
            frame.loc[finished_mask & ~has_score, "final_score"] = "N/A"
            frame.loc[finished_mask & has_score, "final_score"] = (
                rh[finished_mask & has_score].astype(int).astype(str)
                + "-"
                + ra[finished_mask & has_score].astype(int).astype(str)
            )
        return frame

    def _apply_status_labels(frame: pd.DataFrame) -> pd.DataFrame:
        if "status" in frame.columns:
            frame["status"] = frame["status"].apply(status_label)
        if "tournament_id" in frame.columns:
            frame["tournament_id"] = frame["tournament_id"].apply(tournament_label)
        return frame

    def _apply_score_prediction(frame: pd.DataFrame) -> pd.DataFrame:
        if "score_predicted" in frame.columns:
            frame["score_predicted"] = frame["score_predicted"].apply(fmt_score_pred_text)
        return frame

    def _apply_btts_prediction(frame: pd.DataFrame) -> pd.DataFrame:
        if "btts_suggestion" in frame.columns:
            frame["btts_prediction"] = frame["btts_suggestion"].apply(market_label, default="-")
        return frame

    def _apply_guru_highlight(frame: pd.DataFrame) -> pd.DataFrame:
        if "guru_highlight" in frame.columns:
            scope_series = (
                frame["guru_highlight_scope"] if "guru_highlight_scope" in frame.columns else pd.Series("", index=frame.index)
            )
            frame["guru_highlight"] = [
                f"⭐ {scope}".strip() if bool(flag) else ""
                for flag, scope in zip(frame["guru_highlight"], scope_series)
            ]
        return frame

    for step in (
        _translate_markets,
        _compute_final_score,
        _apply_status_labels,
        _apply_score_prediction,
        _apply_btts_prediction,
        _apply_guru_highlight,
    ):
        out = step(out)

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

    df = load_data(content, months_back=_months_back_config())

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


        active_filters = 0
        if tournaments_sel and len(tournaments_sel) != len(tournament_opts):
            active_filters += 1
        model_unique = df["model"].nunique() if "model" in df.columns else 0
        if models_sel and (model_unique and len(models_sel) != model_unique):
            active_filters += 1
        if teams_sel:
            active_filters += 1
        if q_team:
            active_filters += 1
        if bet_sel or goal_sel:
            active_filters += 1
        if selected_date_range:
            active_filters += 1
        if guru_only:
            active_filters += 1

        params = FilterParams(
            tournaments_sel=tournaments_sel,
            models_sel=models_sel,
            teams_sel=teams_sel,
            bet_sel=bet_sel,
            goal_sel=goal_sel,
            selected_date_range=selected_date_range,
            sel_h=sel_h,
            sel_d=sel_d,
            sel_a=sel_a,
            q_team=q_team,
            guru_only=guru_only,
        )
        df_filtered, _, _, _ = apply_dashboard_filters(df, params)

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

            curr_df = df_ag if status_view.startswith("🗓️") else df_fin
            curr_label = "Agendados" if status_view.startswith("🗓️") else "Finalizados"

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

            # Recalcula o recorte exportável após possíveis ajustes automáticos em finalizados
            if status_view.startswith("🗓️"):
                curr_df = df_ag
                curr_label = "Agendados"
            else:
                curr_df = df_fin
                curr_label = "Finalizados"
            export_disabled = curr_df.empty
            # --------------------------------------------------------------------------

            # --- VISÃO POR STATUS (seleção acima) ---
            if status_view.startswith("🗓️"):
                if df_ag.empty:
                    st.info("Sem jogos agendados neste recorte.")
                else:
                    st.caption("Lista de jogos agendados")
                    st.write(f"Jogos futuros no recorte • {len(df_ag)} jogos • {'Visão lista' if use_list_view else 'Visão tabela'}")
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
                    with st.container(border=True):
                        hide_info, hide_toggle = st.columns([4, 1.3])
                        with hide_info:
                            st.caption("Lista de jogos finalizados")
                            st.write("Exibir ou ocultar rapidamente")
                            st.caption("Use quando quiser focar apenas nas métricas e gráficos de desempenho.")
                            st.caption(f"{len(df_fin)} jogos • {'Visão lista' if use_list_view else 'Visão tabela'}")
                        with hide_toggle:
                            hide_games = st.toggle(
                                "Ocultar lista de jogos",
                                key="pg_hide_fin_list",
                                value=st.session_state.get("pg_hide_fin_list", False),
                                help="Oculte a listagem para ver somente KPIs, gráficos e tabelas avançadas.",
                            )

                    if hide_games:
                        st.info("Lista oculta. Desative o toggle acima para reexibir os jogos.")
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

                    metric_order = METRIC_ORDER
                    overall_stats = metric_stats_for(overall_metrics, metric_order)
                    campeonatos_stats = build_tournament_stats(df_fin, metric_order)

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

                    st.caption("Gráfico de acertos")
                    st.subheader("Precisão por métrica")
                    st.caption("Compare modelos, mercados e a taxa de acerto consolidada.")

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
                            with st.container(border=True):
                                st.altair_chart(chart + text, use_container_width=True)
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
                        with st.container(border=True):
                            st.altair_chart(chart + text, use_container_width=True)

                                        # --- Gráficos de linha de acurácia por dia (nativo Altair, sem CDN externo) ---
                    accuracy_data = prepare_accuracy_chart_data(df_fin)
                    if not accuracy_data.empty:
                        st.markdown("### Desempenho diário por campeonato e métrica")
                        tournaments = sorted(accuracy_data["Campeonato"].dropna().unique().tolist())
                        for tourn in tournaments:
                            tourn_df = accuracy_data[accuracy_data["Campeonato"] == tourn]
                            if tourn_df.empty:
                                continue
                            with st.expander(f"{tourn}", expanded=False):
                                metric_options = sorted(tourn_df["Métrica"].dropna().unique().tolist())
                                metrics_sel_chart = st.multiselect(
                                    "Métricas",
                                    metric_options,
                                    default=metric_options,
                                    key=f"pg_daily_metrics_{tourn}",
                                )
                                plot_df = tourn_df[tourn_df["Métrica"].isin(metrics_sel_chart)] if metrics_sel_chart else tourn_df
                                chart = (
                                    alt.Chart(plot_df)
                                    .mark_line(point=True)
                                    .encode(
                                        x=alt.X("Data:T", title="Data"),
                                        y=alt.Y("Taxa de Acerto (%):Q", scale=alt.Scale(domain=[0, 100])),
                                        color=alt.Color("Modelo:N", scale=alt.Scale(range=chart_theme["palette"])),
                                        strokeDash=alt.StrokeDash("Métrica:N"),
                                        tooltip=[
                                            "Campeonato:N",
                                            "Modelo:N",
                                            "Métrica:N",
                                            alt.Tooltip("Taxa de Acerto (%):Q", format=".2f"),
                                            alt.Tooltip("Data:T", title="Data"),
                                        ],
                                    )
                                    .properties(height=260 if modo_mobile else 320)
                                    .interactive()
                                )
                                st.altair_chart(chart, use_container_width=True)
                    else:
                        st.info("Não há dados suficientes para gerar os gráficos de desempenho diário.")

                    # --- Tabela de Melhor Modelo por Campeonato e Mercado ---
                    best_model_data = get_best_model_by_market(df_fin.copy())
                    if not best_model_data.empty:
                        summary_pivot_table = create_summary_pivot_table(best_model_data)

                        st.caption("Modelos vencedores")
                        st.subheader("Sessão de Melhor Modelo")
                        st.caption("Compare o desempenho por campeonato e mercado no mesmo layout das demais tabelas glassy.")

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
        st.divider()

        fcol1, fcol2 = st.columns([3, 2])
        with fcol1:
            st.write(f"**Última atualização:** {last_update_dt.strftime('%d/%m/%Y %H:%M')}")
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
