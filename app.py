"""Módulo principal da aplicação Placar Guru."""
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

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


def render_custom_navigation():
    """Renderiza uma navegação customizada para renomear a página principal para 'Previsões'."""

    # `page_link` está disponível nas versões mais novas do Streamlit; evitamos quebrar builds antigas.
    if not hasattr(st.sidebar, "page_link"):
        return

    st.markdown(
        """
        <style>
        /* Esconde a navegação padrão para evitar duplicação de links */
        [data-testid="stSidebarNav"] { display: none; }
        /* Reduz o espaçamento superior quando a nav padrão está oculta */
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] > div:first-child { padding-top: 0.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("#### Navegação")
        st.page_link("app.py", label="Previsões", icon="🔮")
        st.page_link(
            "pages/2_Analise_de_Desempenho.py",
            label="Análise de Desempenho",
            icon="📊",
        )
        st.divider()


# Garante que o nome da página principal apareça como "Previsões" na navegação lateral customizada
render_custom_navigation()

# CSS para garantir que o header e o botão do menu (hambúrguer) apareçam
fix_header_and_sidebar_css = """
<style>
/* Garante que o header do Streamlit esteja sempre visível */
header[data-testid="stHeader"] {
    visibility: visible !important;
    display: flex !important;
    align-items: center;
    background: transparent !important;
    box-shadow: none !important;
    z-index: 1000 !important;
}

/* Garante que o ícone do menu (toggle do sidebar) apareça */
header [data-testid="baseButton-headerNoPadding"],
header [data-testid="stSidebarNavToggle"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
}

/* Se em algum lugar antigo tiver escondido o sidebar, força mostrar */
section[data-testid="stSidebar"] {
    display: block !important;
}
</style>
"""
# Aplica correção do header somente se estiver habilitada em secrets ou query string
try:
    force_header_patch = bool(st.secrets.get("force_header_patch", False))
except StreamlitSecretNotFoundError:
    force_header_patch = False
force_header_patch = force_header_patch or st.query_params.get("force_header", ["0"])[0] == "1"
if force_header_patch:
    st.markdown(fix_header_and_sidebar_css, unsafe_allow_html=True)


# Estado inicial: Light por padrão com anúncio único
st.session_state.setdefault("pg_dark_mode", False)
st.session_state.setdefault("pg_theme_announce", "")


def _sync_theme_toggle(source_key: str) -> None:
    st.session_state["pg_dark_mode"] = bool(st.session_state.get(source_key, False))
    st.session_state["pg_theme_announce"] = f"Tema {'escuro' if st.session_state['pg_dark_mode'] else 'claro'} ativado"
    st.session_state["pg_dark_mode_header"] = st.session_state["pg_dark_mode"]
    st.session_state["pg_dark_mode_sidebar"] = st.session_state["pg_dark_mode"]


dark_mode = bool(st.session_state.get("pg_dark_mode", False))
st.session_state.setdefault("pg_dark_mode_header", dark_mode)
st.session_state.setdefault("pg_dark_mode_sidebar", dark_mode)
st.session_state["pg_dark_mode_header"] = st.session_state["pg_dark_mode"]
st.session_state["pg_dark_mode_sidebar"] = st.session_state["pg_dark_mode"]

# --- Estilos mobile-first + cores e tema dos gráficos ---
inject_custom_css(dark_mode)
apply_altair_theme(dark_mode)
chart_theme = chart_tokens(dark_mode)

topbar_placeholder = st.empty()
viewport_width = detect_viewport_width()
modo_mobile = viewport_width < MOBILE_BREAKPOINT
st.session_state["pg_mobile_auto"] = modo_mobile
auto_view_label = f"Visual: {'mobile' if modo_mobile else 'desktop'} ({viewport_width}px)"

from reporting import generate_pdf_report
from ui_components import (
    filtros_ui,
    display_list_view,
    is_guru_highlight,
    guru_highlight_summary,
    render_glassy_table,
    render_app_header,
    render_chip,
)
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

    if "guru_highlight" in out.columns:
        scope_series = out["guru_highlight_scope"] if "guru_highlight_scope" in out.columns else pd.Series("", index=out.index)
        out["guru_highlight"] = [
            f"⭐ {scope}".strip() if bool(flag) else ""
            for flag, scope in zip(out["guru_highlight"], scope_series)
        ]

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
        q_team = flt.get("search_query", "")
        tournament_opts = flt.get("tournament_opts", [])
        min_date, max_date = flt.get("min_date"), flt.get("max_date")

        st.session_state.setdefault("pg_list_view_pref", modo_mobile)
        if modo_mobile:
            st.session_state["pg_list_view_pref"] = True
        else:
            st.session_state["pg_list_view_pref"] = st.checkbox(
                "Usar visualização em lista (mobile)",
                value=bool(st.session_state.get("pg_list_view_pref", False)),
                help="Mantém a listagem em cards mesmo no desktop quando preferir.",
            )

        st.session_state.setdefault("pg_table_density", DEFAULT_TABLE_DENSITY)
        density_toggle = st.toggle(
            "Compactar linhas da tabela",
            key="pg_density_toggle",
            value=st.session_state.get("pg_table_density") == "compact",
            help="Alterne para reduzir altura das linhas e ver mais jogos por tela.",
        )
        table_density = "compact" if density_toggle else "comfortable"
        st.session_state["pg_table_density"] = table_density

        use_list_view = bool(st.session_state.get("pg_list_view_pref", modo_mobile))

        shared_state = get_filter_state()
        defaults = flt.get("defaults", {})
        if modo_mobile:
            quick_summary = []
            if tournaments_sel:
                quick_summary.append(tournament_label(tournaments_sel[0]))
            if selected_date_range and isinstance(selected_date_range, (list, tuple)) and len(selected_date_range) == 2:
                quick_summary.append(f"{selected_date_range[0].strftime('%d/%m')}–{selected_date_range[1].strftime('%d/%m')}")
            quick_summary_txt = " · ".join(quick_summary) if quick_summary else "Sem filtros rápidos"

            with st.expander("Filtros rápidos (mobile)", expanded=True):
                st.markdown(
                    f"<p class='pg-mobile-toolbar__hint'>Concentre torneios, período e busca em um único bloco. Ativos: {quick_summary_txt}</p>",
                    unsafe_allow_html=True,
                )

                c1, c2 = st.columns(2)
                base_opts = ["Todos"] + tournament_opts
                quick_idx = 0
                if tournaments_sel and tournaments_sel[0] in tournament_opts:
                    quick_idx = base_opts.index(tournaments_sel[0])
                quick_tourn = c1.selectbox(
                    "Torneio (atalho)",
                    options=base_opts,
                    index=quick_idx,
                    label_visibility="collapsed",
                )
                range_opts = ["Todos", "Hoje", "Próx. 3 dias", "Últimos 3 dias"]
                quick_range_idx = 0
                if selected_date_range and isinstance(selected_date_range, (list, tuple)) and len(selected_date_range) == 2:
                    today = date.today()
                    if selected_date_range == (today, today):
                        quick_range_idx = 1
                    elif selected_date_range == (today, today + timedelta(days=3)):
                        quick_range_idx = 2
                    elif selected_date_range == (today - timedelta(days=3), today):
                        quick_range_idx = 3
                quick_range = c2.selectbox(
                    "Período (atalho)",
                    options=range_opts,
                    index=quick_range_idx,
                    label_visibility="collapsed",
                )
                q_team_input = st.text_input(
                    "Busca rápida por equipe",
                    key="pg_q_team_shared",
                    value=shared_state.search_query or "",
                    placeholder="Digite nome do time...",
                    label_visibility="collapsed",
                )

                if quick_tourn != "Todos":
                    tournaments_sel = [quick_tourn]
                    shared_state.tournaments_sel = tournaments_sel
                if quick_range != "Todos" and min_date and max_date:
                    today = date.today()
                    if quick_range == "Hoje":
                        selected_date_range = (today, today)
                    elif quick_range == "Próx. 3 dias":
                        selected_date_range = (today, today + timedelta(days=3))
                    elif quick_range == "Últimos 3 dias":
                        selected_date_range = (today - timedelta(days=3), today)
                    shared_state.selected_date_range = selected_date_range

                shared_state.search_query = q_team_input

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
        df_filtered = df_filtered.assign(
            guru_highlight_scope=df_filtered.apply(guru_highlight_summary, axis=1)
        )
        df_filtered["guru_highlight"] = df_filtered["guru_highlight_scope"].apply(bool)

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
            if st.session_state.get("pg_theme_announce"):
                live_messages.append(st.session_state.get("pg_theme_announce"))
                st.session_state["pg_theme_announce"] = ""
            header_html = render_app_header(live_messages=live_messages)
            with topbar_placeholder.container():
                brand_col, action_col = st.columns([4, 1.4])
                brand_col.markdown(header_html, unsafe_allow_html=True)
                with action_col:
                    st.toggle(
                        f"Alternar tema — {'escuro' if dark_mode else 'claro'}",
                        value=bool(st.session_state.get("pg_dark_mode_header", dark_mode)),
                        key="pg_dark_mode_header",
                        help="Altere o tema para avaliar contraste em dark/light.",
                        label_visibility="visible",
                        on_change=lambda: _sync_theme_toggle("pg_dark_mode_header"),
                    )

            export_data = generate_pdf_report(curr_df) if not export_disabled else None
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
                    if use_list_view:
                        display_list_view(df_ag)
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
                            display_list_view(df_fin)
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
