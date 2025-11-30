"""Módulo para componentes de UI reutilizáveis."""
import textwrap
import streamlit as st
import pandas as pd
from typing import Optional, List
from datetime import date, timedelta

from utils import (
    FRIENDLY_COLS, market_label, tournament_label, status_label,
    eval_result_pred_row, eval_score_pred_row, eval_bet_row,
    eval_goal_row, eval_btts_suggestion_row, evaluate_market,
    get_prob_and_odd_for_market, fmt_score_pred_text,
    green_html, norm_status_key, FINISHED_TOKENS, _exists, _po, fmt_odd, fmt_prob,
GOAL_MARKET_THRESHOLDS, MARKET_TO_ODDS_COLS
)


HIGHLIGHT_PROB_THRESHOLD = 0.60
HIGHLIGHT_ODD_THRESHOLD = 1.20


def render_glassy_table(df: pd.DataFrame, caption: Optional[str] = None, show_index: Optional[bool] = None):
    """Renderiza uma tabela interativa com visual glassy e realce de Sugestão Guru.

    show_index: força a exibição do índice. Quando None, ativa para índices nomeados
    ou não numéricos para preservar colunas como "Campeonato"/"Mercado de Aposta".
    """

    if df is None or df.empty:
        st.info("Sem dados para exibir.")
        return

    df_to_render = df.copy()
    df_to_render["Guru"] = df_to_render.get("guru_highlight", False).apply(
        lambda v: "🌟 Destaque" if bool(v) else "—"
    )
    if "status" in df_to_render.columns:
        df_to_render["Status (badge)"] = df_to_render["status"].apply(lambda v: f"✅ {v}" if "inaliz" in str(v).lower() else f"🗓️ {v}")
    if show_index is None:
        show_index = not isinstance(df_to_render.index, pd.RangeIndex) or bool(df_to_render.index.name)

    if show_index and not df_to_render.index.name:
        df_to_render.index.name = ""

    column_config = {
        "Guru": st.column_config.Column(
            label="Sugestão Guru",
            help="Aplica quando prob. ≥ 60% e odd > 1.20.",
            width="small",
        ),
        "status": st.column_config.TextColumn(
            label="Status",
            help="Mostra se o jogo está agendado ou finalizado.",
            width="small",
        ),
        "Status (badge)": st.column_config.TextColumn(
            label="Status", help="Badge com o estágio do jogo.", width="small",
        ),
    }

    with st.container():
        st.markdown('<div class="pg-table-card pg-table-card--interactive">', unsafe_allow_html=True)
        st.data_editor(
            df_to_render,
            use_container_width=True,
            hide_index=not show_index,
            disabled=True,
            column_config=column_config,
        )
        legend = "⭐ Sugestão Guru: Prob ≥ 60% e Odd > 1.20."
        if caption:
            legend = f"{caption} · {legend}"
        st.markdown(f"<div class='pg-table-caption'>{legend}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def is_guru_highlight(row: pd.Series) -> bool:
    """Aplica a regra de destaque (prob > 60% e odd > 1.20) usando a sugestão de aposta."""
    market_code = row.get("bet_suggestion")
    if pd.isna(market_code):
        return False

    cols = MARKET_TO_ODDS_COLS.get(str(market_code).strip())
    if not cols:
        return False

    prob = row.get(cols[0])
    odd = row.get(cols[1])

    try:
        prob_val = float(prob)
        odd_val = float(odd)
    except Exception:
        return False

    if pd.isna(prob_val) or pd.isna(odd_val):
        return False

    return prob_val >= HIGHLIGHT_PROB_THRESHOLD and odd_val > HIGHLIGHT_ODD_THRESHOLD

def _render_filtros_modelos(container, model_opts: list, default_models: list, modo_mobile: bool):
    """Renderiza o filtro de seleção de modelos."""
    col = container.columns(1)[0] if modo_mobile else container.columns(2)[0]
    return col.multiselect(FRIENDLY_COLS["model"], model_opts, default=default_models)

def _render_filtros_equipes(container, team_opts: list, modo_mobile: bool, tournaments_sel: Optional[List]):
    """Renderiza os filtros de equipes e a busca rápida."""
    c1, c2 = container.columns(2)
    with c1:
        # Apenas para alinhar com o seletor de equipes
        st.write(f"**{len(tournaments_sel or []):d} torneios selecionados**")

    teams_sel = c2.multiselect(
        "Equipe (Casa ou Visitante)", team_opts,
        default=[] if modo_mobile else team_opts
    )
    q_team = container.text_input(
        "🔍 Buscar equipe (Casa/Visitante)",
        placeholder="Digite parte do nome da equipe...",
        key="pg_q_team_shared",
        value=st.session_state.get("pg_q_team_shared", ""),
    )
    return teams_sel, q_team

def _render_filtros_sugestoes(container, bet_opts: list, goal_opts: list):
    """Renderiza os filtros de sugestões de aposta."""
    c1, c2 = container.columns(2)
    bet_sel = c1.multiselect(
        FRIENDLY_COLS["bet_suggestion"], bet_opts, default=[], format_func=market_label
    )
    goal_sel = c2.multiselect(
        FRIENDLY_COLS["goal_bet_suggestion"], goal_opts, default=[], format_func=market_label
    )
    return bet_sel, goal_sel

def _render_filtros_periodo(container, min_date: Optional[date], max_date: Optional[date]):
    """Renderiza o filtro de período com botões de atalho."""
    selected_date_range = ()
    with container.expander("Período", expanded=False):
        if min_date and max_date:
            today = date.today()
            btn_cols = st.columns(5)
            if btn_cols[0].button("Hoje"):
                selected_date_range = (today, today)
            if btn_cols[1].button("Próx. 3 dias"):
                selected_date_range = (today, today + timedelta(days=3))
            if btn_cols[2].button("Últimos 3 dias"):
                selected_date_range = (today - timedelta(days=3), today)
            if btn_cols[3].button("Semana"):
                start = today - timedelta(days=today.weekday())
                selected_date_range = (start, start + timedelta(days=6))
            if btn_cols[4].button("Limpar"):
                selected_date_range = ()

            if not selected_date_range:
                selected_date_range = st.date_input(
                    "Período (intervalo)", value=(min_date, max_date),
                    min_value=min_date, max_value=max_date
                )
    return selected_date_range

def _render_filtros_odds(container, df: pd.DataFrame):
    """Renderiza os sliders de filtro de odds."""
    def _range(series: pd.Series, default=(0.0, 1.0)):
        """Calcula o range (min, max) de uma série, com um valor padrão."""
        s = series.dropna()
        return (float(s.min()), float(s.max())) if not s.empty else default

    sel_h, sel_d, sel_a = (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
    with container.expander("Odds", expanded=False):
        if "odds_H" in df.columns:
            min_h, max_h = _range(df["odds_H"])
            sel_h = st.slider(FRIENDLY_COLS["odds_H"], min_h, max_h, (min_h, max_h))
        if "odds_D" in df.columns:
            min_d, max_d = _range(df["odds_D"])
            sel_d = st.slider(FRIENDLY_COLS["odds_D"], min_d, max_d, (min_d, max_d))
        if "odds_A" in df.columns:
            min_a, max_a = _range(df["odds_A"])
            sel_a = st.slider(FRIENDLY_COLS["odds_A"], min_a, max_a, (min_a, max_a))
    return sel_h, sel_d, sel_a


def filtros_ui(
    df: pd.DataFrame, modo_mobile: bool,
) -> dict:
    """Renderiza a interface de filtros principal e retorna as seleções do usuário."""
    # Mantém o controle dos filtros no menu lateral esquerdo, oculto por padrão
    st.session_state.setdefault("pg_filters_open", False)
    st.session_state.setdefault("pg_filters_cache", {})
    # --- 1. Extração de Opções ---
    model_opts = sorted(df["model"].dropna().unique()) if "model" in df.columns else []
    tourn_opts = sorted(df["tournament_id"].dropna().unique()) if "tournament_id" in df.columns else []
    team_opts = sorted(pd.concat([df["home"], df["away"]]).dropna().astype(str).unique()) if _exists(df, "home", "away") else []
    bet_opts = sorted(df["bet_suggestion"].dropna().unique()) if "bet_suggestion" in df.columns else []
    goal_opts = sorted(df["goal_bet_suggestion"].dropna().unique()) if "goal_bet_suggestion" in df.columns else []

    def _odds_default(col: str, default: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
        """Retorna o range padrão para a coluna de odds quando o filtro está oculto."""
        if col in df.columns:
            s = df[col].dropna()
            if not s.empty:
                return (float(s.min()), float(s.max()))
        return default

    # --- 2. Lógica de Defaults ---
    default_models = []
    if model_opts:
        url_models = [v.strip().lower() for v in st.session_state.get("model_init_raw", [])]
        wanted = [m for m in model_opts if str(m).strip().lower() in url_models]
        if not wanted:
            wanted = [m for m in model_opts if str(m).strip().lower() == "combo"]
        default_models = wanted or model_opts

    min_date = df["date"].min().date() if "date" in df and df["date"].notna().any() else None
    max_date = df["date"].max().date() if "date" in df and df["date"].notna().any() else None

    # controle de campeonatos no sidebar
    st.session_state.setdefault("sel_tournaments", list(tourn_opts))
    # mantém apenas válidos
    valid_sel = [t for t in st.session_state.sel_tournaments if t in tourn_opts]
    if valid_sel != st.session_state.sel_tournaments:
        st.session_state.sel_tournaments = valid_sel
    if not st.session_state.sel_tournaments and tourn_opts:
        st.session_state.sel_tournaments = list(tourn_opts)
    tournaments_sel = list(st.session_state.sel_tournaments)

    # --- 3. Renderização da UI (menu lateral esquerdo) ---
    with st.sidebar:
        st.markdown("<div class='pg-filter-shell'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="pg-filter-header">
              <div>
                <p class="pg-eyebrow">Filtros principais</p>
                <h4 style="margin:0;">Refine torneios, modelos e odds</h4>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='pg-filter-toggle-label'>Ocultar/mostrar filtros</div>", unsafe_allow_html=True)
        st.toggle(
            "Exibir filtros",
            key="pg_filters_open",
            value=st.session_state.get("pg_filters_open", False),
        )

        if st.session_state.get("pg_filters_open", False):
            st.markdown("<div class='pg-filter-section'><p class='pg-eyebrow'>Campeonatos</p>", unsafe_allow_html=True)
            csel_all, cclear = st.columns(2)
            with csel_all:
                if st.button("Selecionar Todos", use_container_width=True, key="btn_sel_all_tourn"):
                    st.session_state.sel_tournaments = list(tourn_opts)
                    tournaments_sel = list(tourn_opts)
            with cclear:
                if st.button("Limpar", use_container_width=True, key="btn_clear_tourn"):
                    st.session_state.sel_tournaments = []
                    tournaments_sel = []

            st.multiselect(
                label="Selecione campeonatos",
                options=tourn_opts,
                key="sel_tournaments",
                format_func=tournament_label,
                placeholder="Escolha um ou mais campeonatos...",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            models_sel = _render_filtros_modelos(st, model_opts, default_models, modo_mobile)
            teams_sel, q_team = _render_filtros_equipes(
                st, team_opts, modo_mobile, tournaments_sel
            )
            bet_sel, goal_sel = _render_filtros_sugestoes(st, bet_opts, goal_opts)
            selected_date_range = _render_filtros_periodo(st, min_date, max_date)
            sel_h, sel_d, sel_a = _render_filtros_odds(st, df)

            st.session_state.pg_filters_cache = {
                "tournaments_sel": tournaments_sel,
                "models_sel": models_sel,
                "teams_sel": teams_sel,
                "q_team": q_team,
                "bet_sel": bet_sel,
                "goal_sel": goal_sel,
                "selected_date_range": selected_date_range,
                "sel_h": sel_h,
                "sel_d": sel_d,
                "sel_a": sel_a,
            }
        else:
            cache = st.session_state.get("pg_filters_cache", {})
            tournaments_sel = cache.get("tournaments_sel", tournaments_sel)
            models_sel = cache.get("models_sel", default_models)
            teams_sel = cache.get("teams_sel", [])
            q_team = cache.get("q_team", "")
            bet_sel = cache.get("bet_sel", [])
            goal_sel = cache.get("goal_sel", [])
            selected_date_range = cache.get("selected_date_range", ())
            sel_h = cache.get("sel_h")
            sel_d = cache.get("sel_d")
            sel_a = cache.get("sel_a")
            if sel_h is None:
                sel_h = _odds_default("odds_H")
            if sel_d is None:
                sel_d = _odds_default("odds_D")
            if sel_a is None:
                sel_a = _odds_default("odds_A")
            st.markdown("<div class='pg-chip ghost'>Filtros ocultos. Use o toggle acima para ajustar.</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # --- 4. Sincronização e Retorno ---
    try:
        st.query_params["model"] = models_sel or []
    except Exception:
        pass  # Pode falhar em alguns contextos de execução

    return {
        "tournaments_sel": tournaments_sel,
        "models_sel": models_sel,
        "teams_sel": teams_sel,
        "bet_sel": bet_sel,
        "goal_sel": goal_sel,
        "selected_date_range": selected_date_range,
        "sel_h": sel_h, "sel_d": sel_d, "sel_a": sel_a,
        "q_team": q_team,
        "tournament_opts": tourn_opts,
        "min_date": min_date,
        "max_date": max_date,
    }


def _prepare_display_data(row: pd.Series) -> dict:
    """Prepara todos os dados necessários para a exibição de uma linha."""
    dt_txt = row["date"].strftime("%d/%m %H:%M") if ("date" in row.index and pd.notna(row["date"])) else "N/A"

    market_code = row.get("bet_suggestion")
    prob_val = odd_val = None
    if pd.notna(market_code):
        cols = MARKET_TO_ODDS_COLS.get(str(market_code).strip())
        if cols:
            prob_val = row.get(cols[0])
            odd_val = row.get(cols[1])
    highlight = is_guru_highlight(row)

    # Avaliações de acerto
    hit_res = eval_result_pred_row(row)
    hit_score = eval_score_pred_row(row)
    hit_bet = eval_bet_row(row)
    hit_goal = eval_goal_row(row)
    btts_pred = row.get("btts_suggestion")
    hit_btts_pred = eval_btts_suggestion_row(row)

    def _get_badge(hit_status):
        return "✅" if hit_status is True else ("❌" if hit_status is False else "")

    # Textos e odds
    result_txt = f"{market_label(row.get('result_predicted'))} {get_prob_and_odd_for_market(row, row.get('result_predicted'))}"
    score_txt = fmt_score_pred_text(row.get('score_predicted'))
    aposta_txt = f"{market_label(row.get('bet_suggestion'))} {get_prob_and_odd_for_market(row, row.get('bet_suggestion'))}"
    gols_txt = f"{market_label(row.get('goal_bet_suggestion'))} {get_prob_and_odd_for_market(row, row.get('goal_bet_suggestion'))}"
    btts_pred_txt = f"{market_label(btts_pred, default='-')} {get_prob_and_odd_for_market(row, btts_pred)}"

    return {
        "title": f"{dt_txt} • {row.get('home', '?')} vs {row.get('away', '?')}",
        "status_txt": status_label(row.get("status", "N/A")),
        "badge_res": _get_badge(hit_res),
        "badge_score": _get_badge(hit_score),
        "badge_bet": _get_badge(hit_bet),
        "badge_goal": _get_badge(hit_goal),
        "badge_btts_pred": _get_badge(hit_btts_pred),
        "result_txt": result_txt,
        "score_txt": score_txt,
        "aposta_txt": aposta_txt,
        "gols_txt": gols_txt,
        "btts_pred_txt": btts_pred_txt,
        "cap_line": f"{tournament_label(row.get('tournament_id'))} • Modelo {row.get('model','—')}",
        "is_finished": norm_status_key(row.get("status", "")) in FINISHED_TOKENS,
        "final_score": f"{int(row.get('result_home', 0))}-{int(row.get('result_away', 0))}" if pd.notna(row.get("result_home")) else "—",
        "highlight": highlight,
        "suggested_prob": prob_val,
        "suggested_odd": odd_val,
        "match_title": f"{row.get('home','?')} vs {row.get('away','?')}",
        "kickoff": dt_txt,
    }


def _compact_html(html: str) -> str:
    """Remove indentação e quebras de linha para evitar renderização como texto Markdown.

    Streamlit pode exibir tags literalmente se a string HTML começar com espaços/linhas
    vazias, pois o Markdown interpreta como bloco de código. Este helper normaliza
    o HTML em uma única linha, preservando a legibilidade no app.
    """
    return " ".join(
        line.strip()
        for line in textwrap.dedent(html).splitlines()
        if line.strip()
    )

def _build_over_under_lists(row: pd.Series, df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Gera listas com os mercados under/over disponíveis para renderização em HTML."""

    under_lines, over_lines = [], []

    for v in GOAL_MARKET_THRESHOLDS:
        prob_key_under = f"prob_under_{str(v).replace('.', '_')}"
        odd_key_under = f"odds_match_goals_{v}_under"
        if _exists(df, prob_key_under):
            under_lines.append(f"<li><strong>Under {v}:</strong> {_po(row, prob_key_under, odd_key_under)}</li>")

        prob_key_over = f"prob_over_{str(v).replace('.', '_')}"
        odd_key_over = f"odds_match_goals_{v}_over"
        if _exists(df, prob_key_over):
            over_lines.append(f"<li><strong>Over {v}:</strong> {_po(row, prob_key_over, odd_key_over)}</li>")

    return under_lines, over_lines


def _build_details_html(row: pd.Series, data: dict, df: pd.DataFrame) -> str:
    """Monta o HTML do bloco de "Detalhes" dentro do card, evitando expanders externos."""

    under_lines, over_lines = _build_over_under_lists(row, df)
    under_html = "".join(under_lines)
    over_html = "".join(over_lines)

    btts_html = ""
    if _exists(df, "prob_btts_yes", "prob_btts_no"):
        btts_html = """
        <div class="pg-details-block">
          <div class="pg-details-subtitle">BTTS (Prob. — Odd)</div>
          <ul class="pg-details-list">
            <li><strong>Ambos marcam — Sim:</strong> {btts_yes}</li>
            <li><strong>Ambos marcam — Não:</strong> {btts_no}</li>
          </ul>
        </div>
        """.format(
            btts_yes=_po(row, "prob_btts_yes", "odds_btts_yes"),
            btts_no=_po(row, "prob_btts_no", "odds_btts_no"),
        )

    details_html = f"""
    <details class="pg-details">
      <summary>
        <span class="pg-details-title">Detalhes, Probabilidades & Odds</span>
        <span class="pg-details-hint">Toque para abrir os mercados 1x2, O/U e BTTS</span>
      </summary>
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
            <ul class="pg-details-list">{under_html}</ul>
            <ul class="pg-details-list">{over_html}</ul>
          </div>
        </div>
        {btts_html}
      </div>
    </details>
    """

    return _compact_html(details_html)

def display_list_view(df: pd.DataFrame):
    """Renderiza uma lista de jogos em formato de cards para visualização mobile."""
    for _, row in df.iterrows():
        data = _prepare_display_data(row)

        with st.container():
            badge_class = "badge-finished" if data["is_finished"] else "badge-wait"
            highlight_label = (
                "<span class=\"badge\" style=\"background: var(--neon); color:#0f172a; border-color: var(--neon);\">Sugestão Guru</span>"
                if data["highlight"]
                else ""
            )
            final_score_badge = (
                f"<span class=\"badge badge-finished\">Placar Final {data['final_score']}</span>"
                if data["is_finished"]
                else ""
            )
            prob_odd_badge = ""
            if data["suggested_prob"] is not None:
                prob_odd_badge = (
                    f"<span class=\"badge\" style=\"background:color-mix(in srgb, var(--panel) 90%, transparent); border-color:var(--stroke);\">"
                    f"Prob: {fmt_prob(data['suggested_prob']) if data['suggested_prob'] is not None else 'N/A'} • "
                    f"Odd: {fmt_odd(data['suggested_odd']) if data['suggested_odd'] is not None else 'N/A'}"
                    "</span>"
                )

            hit_badges = []
            for label, key in [
                ("Resultado", "badge_res"),
                ("Placar", "badge_score"),
                ("Sugestão", "badge_bet"),
                ("Gols", "badge_goal"),
                ("BTTS", "badge_btts_pred"),
            ]:
                icon = data.get(key)
                if icon:
                    cls = "badge-ok" if icon == "✅" else "badge-bad"
                    hit_badges.append(f"<span class='badge {cls}'>{icon} {label}</span>")
            hit_html = " ".join(hit_badges)

            details_html = _build_details_html(row, data, df)

            card_html = _compact_html(
                f"""
                <div class="pg-card {'neon' if data['highlight'] else ''}">
                  <div style="display:flex; align-items:center; justify-content:space-between; gap:10px;">
                    <div>
                      <div class="pg-meta">{data['cap_line']}</div>
                      <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                        <div style="font-weight:700; font-size:1.05rem;">{data['match_title']}</div>
                        <span class="badge">{data['kickoff']}</span>
                        {highlight_label}
                      </div>
                    </div>
                    <span class="badge {badge_class}">{data['status_txt']}</span>
                  </div>

                  <div class="pg-grid" style="margin-top:10px;">
                    <div class="pg-pill">
                      <div class="label">🎯 Resultado</div>
                      <div class="value">{green_html(data['result_txt'])}</div>
                    </div>
                    <div class="pg-pill">
                      <div class="label">💡 Sugestão</div>
                      <div class="value">{green_html(data['aposta_txt'])}</div>
                      <div class="text-muted" style="font-size:12px;">Prob≥60% & Odd>1.20 ativa o destaque</div>
                    </div>
                    <div class="pg-pill">
                      <div class="label">⚽ Gols</div>
                      <div class="value">{green_html(data['gols_txt'])}</div>
                    </div>
                    <div class="pg-pill">
                      <div class="label">🥅 Ambos marcam</div>
                      <div class="value">{green_html(data['btts_pred_txt'])}</div>
                    </div>
                    <div class="pg-pill">
                      <div class="label">📊 Placar Previsto</div>
                      <div class="value">{green_html(data['score_txt'])}</div>
                    </div>
                  </div>

                  <div style="display:flex; align-items:center; gap:10px; margin-top:10px; flex-wrap:wrap;">
                    {final_score_badge}
                    {prob_odd_badge}
                    {hit_html}
                  </div>

                  {details_html}
                </div>
                """
            )

            st.markdown(card_html, unsafe_allow_html=True)
            st.write("")

def filtros_analise_ui(df: pd.DataFrame) -> dict:
    """Renderiza a interface de filtros para a página de análise de desempenho."""
    st.sidebar.header("Parâmetros da Análise")
    prob_min = st.sidebar.slider("Probabilidade Mínima (%)", 0, 100, 65, 1, "%d%%") / 100.0
    odd_min = st.sidebar.slider("Odd Mínima", 1.0, 5.0, 1.3, 0.01)

    st.sidebar.header("Filtros de Jogos")
    tourn_opts = sorted(df["tournament_id"].dropna().unique().tolist()) if "tournament_id" in df.columns else []
    model_opts = sorted(df["model"].dropna().unique()) if "model" in df.columns else []

    models_sel = st.sidebar.multiselect(FRIENDLY_COLS["model"], model_opts, default=model_opts)
    tournaments_sel = st.sidebar.multiselect(FRIENDLY_COLS["tournament_id"], tourn_opts, default=tourn_opts, format_func=tournament_label)

    selected_date_range = ()
    if "date" in df.columns and df["date"].notna().any():
        min_date, max_date = df["date"].dropna().min().date(), df["date"].dropna().max().date()
        selected_date_range = st.sidebar.date_input("Período (intervalo)", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    return dict(prob_min=prob_min, odd_min=odd_min, tournaments_sel=tournaments_sel, models_sel=models_sel, selected_date_range=selected_date_range)
