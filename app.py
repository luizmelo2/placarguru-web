
"""Módulo principal da aplicação Placar Guru."""
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
import streamlit.components.v1 as components
import pandas as pd
from datetime import datetime, date
from zoneinfo import ZoneInfo
import requests
from email.utils import parsedate_to_datetime

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
from reporting import generate_pdf_report
from ui_components import (
    filtros_ui,
    display_list_view,
    guru_highlight_flags,
    guru_highlight_summary,
    render_glassy_table,
    render_app_header,
)
from analysis import (
    prepare_accuracy_chart_data,
    get_best_model_by_market,
    create_summary_pivot_table,
    calculate_kpis,
)

st.set_page_config(
    layout="wide",
    page_title="Previsões",
    initial_sidebar_state="expanded",
)

def render_custom_navigation():
    """Renderiza a navegação customizada."""
    if hasattr(st.sidebar, "page_link"):
        st.markdown(
            """
            <style>
            [data-testid="stSidebarNav"] { display: none; }
            [data-testid="stSidebar"] [data-testid="stSidebarContent"] > div:first-child { padding-top: 0.25rem; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        with st.sidebar:
            st.markdown("#### Navegação")
            st.page_link("app.py", label="Previsões", icon="🔮")
            st.page_link("pages/2_Analise_de_Desempenho.py", label="Análise de Desempenho", icon="📊")
            st.divider()

def inject_topbar_branding():
    """Adiciona a marca Placar Guru no header."""
    st.markdown(
        """
        <style>
        header[data-testid="stHeader"] .pg-topbar-brand {
            display: inline-flex; align-items: baseline; gap: 6px; padding: 6px 12px;
            border-radius: 14px; border: 1px solid color-mix(in srgb, var(--stroke) 80%, var(--primary) 12%);
            background: color-mix(in srgb, var(--panel) 92%, var(--glass-strong)); box-shadow: 0 10px 28px rgba(0,0,0,0.08);
            font-weight: 800; font-size: 13px; letter-spacing: -0.01em; color: var(--text); margin-left: 8px; white-space: nowrap;
        }
        header[data-testid="stHeader"] .pg-topbar-brand span { color: var(--muted); font-weight: 700; }
        header[data-testid="stHeader"] [data-testid="stToolbarActions"] button[title*="Deploy"],
        header[data-testid="stHeader"] [data-testid="stToolbarActions"] .stDeployButton { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    components.html(
        """
        <script>
        const pgTopbarInterval = setInterval(() => {
          const doc = window.parent?.document;
          if (doc) {
            const header = doc.querySelector('header[data-testid="stHeader"]');
            if (header) {
              const toolbar = header.querySelector('[data-testid="stToolbar"]') || header;
              if (!header.querySelector('.pg-topbar-brand')) {
                const brand = doc.createElement('div');
                brand.className = 'pg-topbar-brand';
                brand.innerHTML = '<strong>Placar Guru</strong><span>/ Previsões Esportivas</span>';
                toolbar.appendChild(brand);
              } else {
                clearInterval(pgTopbarInterval);
              }
            }
          }
        }, 350);
        </script>
        """,
        height=0, width=0,
    )

def apply_friendly_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica formatações amigáveis para exibição."""
    out = df.copy()
    for col in ["bet_suggestion", "goal_bet_suggestion", "result_predicted"]:
        if col in out.columns:
            out[col] = out[col].apply(market_label)

    def _fmt_score(row):
        is_finished = norm_status_key(row.get("status", "")) in FINISHED_TOKENS
        has_score = pd.notna(row.get("result_home")) and pd.notna(row.get("result_away"))
        return f"{int(row['result_home'])}-{int(row['result_away'])}" if is_finished and has_score else ""

    if all(c in out.columns for c in ["status", "result_home", "result_away"]):
        out["final_score"] = out.apply(_fmt_score, axis=1)

    for col, func in {
        "status": status_label, "tournament_id": tournament_label,
        "score_predicted": fmt_score_pred_text
    }.items():
        if col in out.columns:
            out[col] = out[col].apply(func)

    if "guru_highlight" in out.columns:
        scope = out.get("guru_highlight_scope", "")
        out["guru_highlight"] = [f"⭐ {s}".strip() if f else "" for f, s in zip(out["guru_highlight"], scope)]

    return out.rename(columns=FRIENDLY_COLS)

def render_scheduled_view(df_ag, use_list_view, guru_only, table_density, modo_mobile):
    """Renderiza a visualização de jogos agendados."""
    if df_ag.empty:
        st.info("Sem jogos agendados neste recorte.")
    elif use_list_view:
        display_list_view(df_ag, hide_missing=guru_only)
    else:
        preset = "compact" if table_density == "compact" and modo_mobile else ("mobile" if modo_mobile else "desktop")
        cols = [c for c in TABLE_COLUMN_PRESETS.get(preset, []) if c in df_ag.columns]
        render_glassy_table(apply_friendly_for_display(df_ag[cols]), caption="Jogos agendados", density=table_density)

def render_finished_view(df_fin, use_list_view, guru_only, table_density, modo_mobile):
    """Renderiza a visualização de jogos finalizados e suas análises."""
    if df_fin.empty:
        st.info("Sem jogos finalizados neste recorte.")
        return

    if not st.toggle("Ocultar lista de jogos", key="pg_hide_fin_list", value=False):
        if use_list_view:
            display_list_view(df_fin, hide_missing=guru_only)
        else:
            preset = "compact" if table_density == "compact" and modo_mobile else ("mobile" if modo_mobile else "desktop")
            cols = [c for c in TABLE_COLUMN_PRESETS.get(preset, []) if c in df_fin.columns]
            render_glassy_table(apply_friendly_for_display(df_fin[cols]), caption="Jogos finalizados", density=table_density)

    metrics_df = calculate_kpis(df_fin, df_fin["model"].nunique() > 1)
    if not metrics_df.empty:
        st.markdown("### Análise de Desempenho")
        render_glassy_table(metrics_df, caption="Precisão por Métrica")

    best_model_data = get_best_model_by_market(df_fin.copy())
    if not best_model_data.empty:
        st.markdown("#### Melhores Modelos por Mercado")
        render_glassy_table(create_summary_pivot_table(best_model_data), caption="Resumo do Melhor Modelo por Mercado")

def main():
    """Função principal da aplicação."""
    render_custom_navigation()
    inject_custom_css(st.session_state.get("pg_dark_mode", False))
    apply_altair_theme(st.session_state.get("pg_dark_mode", False))
    inject_topbar_branding()

    try:
        content, _, last_mod = fetch_release_file(RELEASE_URL)
        last_update_dt = parsedate_to_datetime(last_mod).astimezone(ZoneInfo("America/Sao_Paulo")) if last_mod else datetime.now(tz=ZoneInfo("America/Sao_Paulo"))
        df = load_data(content)

        if df.empty:
            st.error("O arquivo `PrevisaoJogos.xlsx` está vazio.")
            return

        modo_mobile = detect_viewport_width() < MOBILE_BREAKPOINT
        flt = filtros_ui(df, modo_mobile)

        guru_scope_all = df.apply(guru_highlight_summary, axis=1)
        guru_flag_all = guru_scope_all.apply(bool)

        mask = pd.Series(True, index=df.index)
        if flt["tournaments_sel"]: mask &= df["tournament_id"].isin(flt["tournaments_sel"])
        if flt["models_sel"]: mask &= df["model"].isin(flt["models_sel"])
        if flt["teams_sel"]: mask &= (df["home"].isin(flt["teams_sel"]) | df["away"].isin(flt["teams_sel"]))
        if flt.get("search_query"): mask &= (df["home"].str.contains(flt["search_query"], case=False, na=False) | df["away"].str.contains(flt["search_query"], case=False, na=False))
        if flt["bet_sel"]: mask &= df["bet_suggestion"].isin(flt["bet_sel"])
        if flt["goal_sel"]: mask &= df["goal_bet_suggestion"].isin(flt["goal_sel"])
        if flt["selected_date_range"]:
            start, end = flt["selected_date_range"]
            mask &= df["date"].dt.date.between(start, end)
        if flt.get("guru_only"): mask &= guru_flag_all

        df_filtered = df[mask].assign(guru_highlight_scope=guru_scope_all[mask], guru_highlight=guru_flag_all[mask])

        status_view = st.radio("Ver jogos:", ["🗓️ Agendados", "✅ Finalizados"], horizontal=True)
        is_finished = df_filtered["status"].apply(norm_status_key) == "finished"
        df_ag, df_fin = df_filtered[~is_finished], df_filtered[is_finished]

        if status_view.startswith("✅"):
            render_finished_view(df_fin, st.session_state.get("pg_list_view_pref", modo_mobile), flt.get("guru_only", False), "compact", modo_mobile)
        else:
            render_scheduled_view(df_ag, st.session_state.get("pg_list_view_pref", modo_mobile), flt.get("guru_only", False), "compact", modo_mobile)

    except FileNotFoundError:
        st.error("`PrevisaoJogos.xlsx` não encontrado.")
    except requests.exceptions.RequestException as e:
        st.error(f"Erro de rede ao buscar dados: {e}")
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado no processamento dos dados: {e}")

if __name__ == "__main__":
    main()
