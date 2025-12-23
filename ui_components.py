
"""Módulo para componentes de UI reutilizáveis."""
import base64
import html
import mimetypes
import re
import textwrap
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Optional, List
from datetime import date, timedelta
import streamlit as st
import pandas as pd

from state import (
    get_filter_state,
    set_filter_state,
    reset_filters,
    build_filter_defaults,
    DEFAULT_TABLE_DENSITY,
)

from utils import (
    FRIENDLY_COLS, market_label, tournament_label, status_label,
    eval_result_pred_row, eval_score_pred_row, eval_bet_row,
    eval_goal_row, eval_btts_suggestion_row,
    get_prob_and_odd_for_market, fmt_score_pred_text,
    green_html, norm_status_key, FINISHED_TOKENS, _exists, _po, fmt_odd, fmt_prob,
    GOAL_MARKET_THRESHOLDS, MARKET_TO_ODDS_COLS, generate_sofascore_link
)

LOGO_DIR = Path(__file__).parent / "images"
SOFASCORE_ICON_PATH = LOGO_DIR / "sofascore_icon.svg"
DEFAULT_LOGO_PATH = LOGO_DIR / "default_team.svg"

HIGHLIGHT_PROB_THRESHOLD = 0.80
LOGO_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".webp"}


def _slugify_team(name: str) -> str:
    """Normaliza nomes de equipes para facilitar o match com o arquivo do escudo."""
    if not name:
        return ""
    normalized = unicodedata.normalize("NFKD", str(name))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_only).strip("_").lower()
    return re.sub(r"_+", "_", safe)


@lru_cache(maxsize=1)
def _team_logo_index() -> dict[str, Path]:
    """Indexa os arquivos de escudo disponíveis na pasta images/ por nome da equipe."""
    mapping: dict[str, Path] = {}
    if LOGO_DIR.exists():
        for path in LOGO_DIR.iterdir():
            if path.suffix.lower() not in LOGO_EXTENSIONS:
                continue
            stem = path.stem.split("_", 1)[-1]
            mapping[_slugify_team(stem)] = path
    return mapping


@lru_cache(maxsize=512)
def team_logo_data_uri(team_name: str) -> str:
    """Retorna um data URI base64 do escudo ou da imagem padrão."""
    logo_path = _team_logo_index().get(_slugify_team(team_name))
    if not logo_path or not logo_path.exists():
        logo_path = DEFAULT_LOGO_PATH if DEFAULT_LOGO_PATH.exists() else None
    if not logo_path:
        return ""

    mime = mimetypes.guess_type(logo_path.name)[0] or "image/png"
    encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@lru_cache(maxsize=1)
def _get_sofascore_icon_svg() -> str:
    """Carrega o conteúdo do SVG do ícone do Sofascore."""
    try:
        return SOFASCORE_ICON_PATH.read_text()
    except Exception:
        return ""


def _team_badge_html(team_name: str) -> str:
    """Monta o HTML acessível com escudo + nome da equipe."""
    safe_name = html.escape(team_name or "?")
    logo_src = team_logo_data_uri(team_name)
    logo_img = f"<img src='{logo_src}' alt='Escudo de {safe_name}' class='pg-team__logo' loading='lazy' />" if logo_src else ""
    return f"<span class='pg-team'>{logo_img}<span class='pg-team__name'>{safe_name}</span></span>"


def render_chip(text: str, tone: str = "ghost", aria_label: Optional[str] = None) -> str:
    """Renderiza um chip reutilizável com tom e rótulo acessível."""
    cls = "pg-chip" + (" ghost" if tone == "ghost" else "")
    aria = f" aria-label='{aria_label}'" if aria_label else ""
    return f"<span class='{cls}'{aria}>{text}</span>"


def render_status_badge(status: str) -> str:
    """Badge unificado de status para header/tabela."""
    label = status_label(status)
    prefix = "✅" if norm_status_key(status) in FINISHED_TOKENS else "🗓️"
    return f"{prefix} {label}"


def _prob_from_market(row: pd.Series, market_code: Optional[str]) -> Optional[float]:
    """Retorna a probabilidade associada a um mercado, se existir."""
    if pd.isna(market_code):
        return None
    cols = MARKET_TO_ODDS_COLS.get(str(market_code).strip())
    if not cols:
        return None
    try:
        return float(row.get(cols[0]))
    except (TypeError, ValueError):
        return None


def guru_highlight_flags(row: pd.Series) -> dict[str, bool]:
    """Retorna flags de destaque Guru por tipo de previsão."""
    markets = {
        "Resultado": "result_predicted",
        "Sugestão": "bet_suggestion",
        "Gols": "goal_bet_suggestion",
        "Ambos Marcam": "btts_suggestion",
    }
    return {label: (_prob_from_market(row, row.get(market)) or 0.0) >= HIGHLIGHT_PROB_THRESHOLD for label, market in markets.items()}


def guru_highlight_summary(row: pd.Series, sep: str = " · ") -> str:
    """Retorna uma string com as previsões que passaram do corte Guru."""
    return sep.join([label for label, is_on in guru_highlight_flags(row).items() if is_on])


def render_app_header(live_messages: Optional[list[str]] = None) -> str:
    """Header minimalista com apenas nome e slogan."""
    live_text = " | ".join(filter(None, live_messages or []))
    return f"""
    <div class="pg-header" role="banner">
      <div class="pg-header__brand" aria-label="Placar Guru">
        <div class="pg-logo" aria-label="Escudo do Placar Guru" role="img">
          <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
            <path class="pg-logo-shield" d="M12 10h40l-3.2 32.5L32 56 15.2 42.5 12 10Z" />
            <rect class="pg-logo-chart" x="18" y="30" width="6" height="14" rx="2" />
            <rect class="pg-logo-chart" x="26" y="26" width="6" height="18" rx="2" />
            <rect class="pg-logo-chart" x="34" y="34" width="6" height="10" rx="2" />
            <rect class="pg-logo-chart" x="42" y="22" width="6" height="22" rx="2" />
            <circle class="pg-logo-ball" cx="34.5" cy="21.5" r="8" />
            <circle class="pg-logo-glow" cx="34.5" cy="21.5" r="3.4" />
          </svg>
        </div>
        <div>
          <p class="pg-eyebrow">Placar Guru</p>
          <div class="pg-appname">Futebol + Data Science</div>
        </div>
      </div>
      <div class="pg-sr" aria-live="polite">{live_text}</div>
    </div>
    """


def render_glassy_table(df: pd.DataFrame, caption: Optional[str] = None, show_index: Optional[bool] = None, density: str = "comfortable"):
    """Renderiza uma tabela interativa com visual glassy."""
    if df is None or df.empty:
        st.info("Sem dados para exibir.")
        return

    df_to_render = df.copy()
    guru_key = next((c for c in ("guru_highlight", "Sugestão Guru") if c in df_to_render.columns), None)
    guru_scope_key = next((c for c in ("guru_highlight_scope", "Sugestão Guru (detalhe)") if c in df_to_render.columns), None)

    def _guru_cell(idx, value):
        if not value: return "—"
        scope = str(df_to_render.at[idx, guru_scope_key]).strip() if guru_scope_key and idx in df_to_render.index else ""
        return f"⭐ {scope}" if scope else "⭐"

    if guru_key:
        df_to_render["Guru"] = [ _guru_cell(idx, v) for idx, v in df_to_render[guru_key].items()]

    status_col = next((c for c in ("Status", "status") if c in df_to_render.columns), None)
    if status_col:
        df_to_render["Status (badge)"] = df_to_render[status_col].apply(render_status_badge)

    if show_index is None:
        show_index = not isinstance(df_to_render.index, pd.RangeIndex) or bool(df_to_render.index.name)
    if show_index and not df_to_render.index.name:
        df_to_render.index.name = ""

    st.data_editor(
        df_to_render, use_container_width=True, hide_index=not show_index, disabled=True,
        column_config={
            "Guru": st.column_config.Column(label="Sugestão Guru", width="small"),
            "Status (badge)": st.column_config.TextColumn(label="Status", width="small"),
        }
    )

def _render_filtros_campeonatos(container, current_selection: list, all_options: list) -> list:
    """Renderiza os filtros de campeonatos."""
    csel_all, cclear = container.columns(2)
    tournaments_sel = current_selection
    if csel_all.button("Selecionar Todos", use_container_width=True, key="btn_sel_all_tourn"): tournaments_sel = all_options
    if cclear.button("Limpar", use_container_width=True, key="btn_clear_tourn"): tournaments_sel = []
    return container.multiselect(
        "Selecione campeonatos", all_options, key="sel_tournaments", default=tournaments_sel,
        format_func=tournament_label, placeholder="Escolha um ou mais campeonatos..."
    )

def _render_filtros_modelos(container, model_opts: list, default_models: list):
    """Renderiza o filtro de seleção de modelos."""
    return container.multiselect(
        FRIENDLY_COLS["model"], model_opts, default=default_models,
        placeholder="Selecione um ou mais modelos..."
    )

def _render_filtros_equipes(container, team_opts: list, modo_mobile: bool, search_query: str, default_teams: Optional[list] = None):
    """Renderiza os filtros de equipes e a busca rápida."""
    col_sel, col_input = container.columns([1, 1.2] if not modo_mobile else [1, 1])
    teams_sel = col_sel.multiselect("Equipe (Casa ou Visitante)", team_opts, default=default_teams or [], placeholder="Escolha uma ou mais equipes...")
    q_team = col_input.text_input("🔍 Buscar equipe", placeholder="Digite nome da equipe...", key="pg_q_team_shared", value=search_query)
    return teams_sel, q_team

def _render_filtros_sugestoes(container, bet_opts: list, goal_opts: list, defaults: Optional[dict] = None):
    """Renderiza os filtros de sugestões de aposta."""
    defaults = defaults or {}
    c1, c2 = container.columns(2)
    bet_sel = c1.multiselect(FRIENDLY_COLS["bet_suggestion"], bet_opts, default=defaults.get("bet_sel", []), format_func=market_label, placeholder="Ex.: Vencedor, Dupla chance...")
    goal_sel = c2.multiselect(FRIENDLY_COLS["goal_bet_suggestion"], goal_opts, default=defaults.get("goal_sel", []), format_func=market_label, placeholder="Ex.: Over/Under, BTTS...")
    guru_only = container.toggle("Apenas Sugestão Guru", key="pg_guru_only", value=bool(defaults.get("guru_only", False)), help="Filtra jogos com probabilidade >= 80%.")
    return bet_sel, goal_sel, guru_only

def _render_filtros_periodo(container, min_date: Optional[date], max_date: Optional[date], current_range: tuple = ()):
    """Renderiza o filtro de período com botões de atalho."""
    if not (min_date and max_date): return ()

    today = date.today()
    btn_cols = container.columns(5)
    if btn_cols[0].button("Hoje", use_container_width=True): current_range = (today, today)
    if btn_cols[1].button("Próx. 3 dias", use_container_width=True): current_range = (today, today + timedelta(days=3))
    if btn_cols[2].button("Últimos 3 dias", use_container_width=True): current_range = (today - timedelta(days=3), today)
    if btn_cols[3].button("Semana", use_container_width=True):
        start = today - timedelta(days=today.weekday())
        current_range = (start, start + timedelta(days=6))
    if btn_cols[4].button("Limpar", use_container_width=True): current_range = ()

    return container.date_input("Período", value=current_range or (min_date, max_date), min_value=min_date, max_value=max_date)

def _render_filtros_odds(container, df: pd.DataFrame, defaults: Optional[dict] = None):
    """Renderiza os sliders de filtro de odds."""
    defaults = defaults or {}
    sel_h, sel_d, sel_a = defaults.get("sel_h"), defaults.get("sel_d"), defaults.get("sel_a")

    def _range(col):
        return (float(df[col].min()), float(df[col].max())) if col in df and not df[col].dropna().empty else (0.0, 1.0)

    with container.expander("Odds", expanded=False):
        sel_h = st.slider(FRIENDLY_COLS["odds_H"], *_range("odds_H"), value=sel_h)
        sel_d = st.slider(FRIENDLY_COLS["odds_D"], *_range("odds_D"), value=sel_d)
        sel_a = st.slider(FRIENDLY_COLS["odds_A"], *_range("odds_A"), value=sel_a)
    return sel_h, sel_d, sel_a

def filtros_ui(df: pd.DataFrame, modo_mobile: bool) -> dict:
    """Renderiza a interface de filtros principal."""
    defaults, opts = build_filter_defaults(df, modo_mobile)
    state = get_filter_state(defaults)

    with st.sidebar:
        st.markdown("### Filtros")
        if st.button("Limpar filtros", use_container_width=True):
            state = reset_filters(defaults)

        state.tournaments_sel = _render_filtros_campeonatos(st.container(), state.tournaments_sel, opts["tourn_opts"])
        state.models_sel = _render_filtros_modelos(st.container(), opts["model_opts"], state.models_sel)
        state.teams_sel, state.search_query = _render_filtros_equipes(st.container(), opts["team_opts"], modo_mobile, state.search_query, state.teams_sel)
        state.bet_sel, state.goal_sel, state.guru_only = _render_filtros_sugestoes(st.container(), opts["bet_opts"], opts["goal_opts"], asdict(state))
        state.selected_date_range = _render_filtros_periodo(st.container(), opts["min_date"], opts["max_date"], state.selected_date_range)
        state.sel_h, state.sel_d, state.sel_a = _render_filtros_odds(st.container(), df, asdict(state))

    set_filter_state(state)
    return {**asdict(state), **opts, "defaults": defaults}

def _prepare_display_data(row: pd.Series, hide_missing: bool = False) -> dict:
    """Prepara todos os dados necessários para a exibição de uma linha."""
    home_name, away_name = str(row.get("home", "?")), str(row.get("away", "?"))

    def _get_badge(hit_status): return "✅" if hit_status else ("❌" if hit_status is False else "")
    missing_label = "—" if hide_missing else "Sem previsão"

    return {
        "status_txt": status_label(row.get("status", "N/A")),
        "badge_res": _get_badge(eval_result_pred_row(row)),
        "badge_score": _get_badge(eval_score_pred_row(row)),
        "badge_bet": _get_badge(eval_bet_row(row)),
        "badge_goal": _get_badge(eval_goal_row(row)),
        "badge_btts_pred": _get_badge(eval_btts_suggestion_row(row)),
        "result_txt": f"{market_label(row.get('result_predicted'), default=missing_label)} {get_prob_and_odd_for_market(row, row.get('result_predicted'))}",
        "score_txt": fmt_score_pred_text(row.get('score_predicted'), default=missing_label),
        "aposta_txt": f"{market_label(row.get('bet_suggestion'), default=missing_label)} {get_prob_and_odd_for_market(row, row.get('bet_suggestion'))}",
        "gols_txt": f"{market_label(row.get('goal_bet_suggestion'), default=missing_label)} {get_prob_and_odd_for_market(row, row.get('goal_bet_suggestion'))}",
        "btts_pred_txt": f"{market_label(row.get('btts_suggestion'), default=missing_label)} {get_prob_and_odd_for_market(row, row.get('btts_suggestion'))}",
        "cap_line": f"{tournament_label(row.get('tournament_id'))} • Modelo {row.get('model','—')}",
        "is_finished": norm_status_key(row.get("status", "")) in FINISHED_TOKENS,
        "final_score": f"{int(row.get('result_home', 0))}-{int(row.get('result_away', 0))}" if pd.notna(row.get("result_home")) else "—",
        "highlight": bool(guru_highlight_summary(row)),
        "highlight_scope": guru_highlight_summary(row),
        "match_title_html": f"{_team_badge_html(home_name)}<span class='pg-vs'>vs</span>{_team_badge_html(away_name)}",
        "kickoff": row["date"].strftime("%d/%m %H:%M") if pd.notna(row.get("date")) else "N/A",
        "sofascore_link": generate_sofascore_link(home_name, away_name),
    }

def _build_details_html(row: pd.Series, data: dict, df: pd.DataFrame) -> str:
    """Monta o HTML do bloco de "Detalhes" dentro do card."""
    under_lines, over_lines = [], []
    for v in GOAL_MARKET_THRESHOLDS:
        under_lines.append(f"<li><strong>Under {v}:</strong> {_po(row, f'prob_under_{v.replace('.','_')}', f'odds_match_goals_{v}_under')}</li>")
        over_lines.append(f"<li><strong>Over {v}:</strong> {_po(row, f'prob_over_{v.replace('.','_')}', f'odds_match_goals_{v}_over')}</li>")

    btts_html = f"""
    <div class="pg-details-block">
      <div class="pg-details-subtitle">BTTS (Prob. — Odd)</div>
      <ul class="pg-details-list">
        <li><strong>Ambos marcam — Sim:</strong> {_po(row, 'prob_btts_yes', 'odds_btts_yes')}</li>
        <li><strong>Ambos marcam — Não:</strong> {_po(row, 'prob_btts_no', 'odds_btts_no')}</li>
      </ul>
    </div>
    """ if "prob_btts_yes" in df.columns else ""

    return f"""
    <details class="pg-details">
      <summary><span class="pg-details-title">Detalhes & Odds</span></summary>
      <div class="pg-details-body">
        <div class="pg-details-block">
          <div class="pg-details-subtitle">Sugestões e 1x2</div>
          <ul class="pg-details-list">
            <li><strong>Sugestão:</strong> {green_html(data['aposta_txt'])} {data['badge_bet']}</li>
            <li><strong>Sugestão de Gols:</strong> {green_html(data['gols_txt'])} {data['badge_goal']}</li>
            <li><strong>Odds 1x2:</strong> {green_html(fmt_odd(row.get('odds_H')))} / {green_html(fmt_odd(row.get('odds_D')))} / {green_html(fmt_odd(row.get('odds_A')))}</li>
            <li><strong>Prob. (H/D/A):</strong> {green_html(fmt_prob(row.get('prob_H')))} / {green_html(fmt_prob(row.get('prob_D')))} / {green_html(fmt_prob(row.get('prob_A')))}</li>
          </ul>
        </div>
        <div class="pg-details-block">
          <div class="pg-details-subtitle">Over/Under (Prob. — Odd)</div>
          <div class="pg-details-two-cols">
            <ul class="pg-details-list">{"".join(under_lines)}</ul>
            <ul class="pg-details-list">{"".join(over_lines)}</ul>
          </div>
        </div>
        {btts_html}
      </div>
    </details>
    """

def display_list_view(df: pd.DataFrame, hide_missing: bool = False):
    """Renderiza uma lista de jogos em formato de cards para visualização mobile."""
    for _, row in df.iterrows():
        data = _prepare_display_data(row, hide_missing)
        details_html = _build_details_html(row, data, df)
        sofascore_html = f'<a href="{data["sofascore_link"]}" target="_blank" rel="noopener noreferrer" class="pg-sofascore-link">{_get_sofascore_icon_svg()}</a>' if data.get("sofascore_link") else ""

        highlight_label = f"<span class='badge' style='background: var(--neon); color:#0f172a;'>Sugestão Guru{' — ' + data['highlight_scope'] if data['highlight_scope'] else ''}</span>" if data["highlight"] else ""
        final_score_badge = f"<span class='badge badge-finished'>Placar: {data['final_score']}</span>" if data["is_finished"] else ""
        hit_badges = "".join([f"<span class='badge {'badge-ok' if data[key]=='✅' else 'badge-bad'}'>{data[key]} {label}</span>" for label, key in [("Resultado", "badge_res"), ("Placar", "badge_score"), ("Sugestão", "badge_bet"), ("Gols", "badge_goal"), ("BTTS", "badge_btts_pred")] if data.get(key)])

        st.markdown(f"""
            <div class="pg-card {'neon' if data['highlight'] else ''}">
              <div class="pg-meta">{data['cap_line']}</div>
              <div class="pg-matchup">{data['match_title_html']}{sofascore_html}<span class="badge">{data['kickoff']}</span>{highlight_label}</div>
              <div class="pg-grid">
                <div class="pg-pill"><div class="label">🎯 Resultado</div><div class="value">{green_html(data['result_txt'])}</div></div>
                <div class="pg-pill"><div class="label">💡 Sugestão</div><div class="value">{green_html(data['aposta_txt'])}</div></div>
                <div class="pg-pill"><div class="label">⚽ Gols</div><div class="value">{green_html(data['gols_txt'])}</div></div>
                <div class="pg-pill"><div class="label">🥅 Ambos marcam</div><div class="value">{green_html(data['btts_pred_txt'])}</div></div>
                <div class="pg-pill"><div class="label">📊 Placar Previsto</div><div class="value">{green_html(data['score_txt'])}</div></div>
              </div>
              <div class="pg-match-foot">{final_score_badge}{hit_badges}</div>
              {details_html}
            </div>
        """, unsafe_allow_html=True)

def filtros_analise_ui(df: pd.DataFrame) -> dict:
    """Renderiza a interface de filtros para a página de análise de desempenho."""
    st.sidebar.header("Parâmetros da Análise")
    prob_min = st.sidebar.slider("Probabilidade Mínima (%)", 0, 100, 65, 1, "%d%%") / 100.0
    odd_min = st.sidebar.slider("Odd Mínima", 1.0, 5.0, 1.3, 0.01)

    st.sidebar.header("Filtros de Jogos")
    tourn_opts = sorted(df["tournament_id"].dropna().unique()) if "tournament_id" in df.columns else []
    model_opts = sorted(df["model"].dropna().unique()) if "model" in df.columns else []

    models_sel = st.sidebar.multiselect(FRIENDLY_COLS["model"], model_opts, default=model_opts)
    tournaments_sel = st.sidebar.multiselect(FRIENDLY_COLS["tournament_id"], tourn_opts, default=tourn_opts, format_func=tournament_label)

    selected_date_range = ()
    if "date" in df.columns and df["date"].notna().any():
        min_date, max_date = df["date"].min().date(), df["date"].max().date()
        selected_date_range = st.sidebar.date_input("Período", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    return dict(prob_min=prob_min, odd_min=odd_min, tournaments_sel=tournaments_sel, models_sel=models_sel, selected_date_range=selected_date_range)
