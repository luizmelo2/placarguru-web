"""Módulo para componentes de UI reutilizáveis."""
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
    GOAL_MARKET_THRESHOLDS
)

def _render_filtros_modelos(container, model_opts: list, default_models: list, modo_mobile: bool):
    """Renderiza o filtro de seleção de modelos."""
    col = container.columns(1)[0] if modo_mobile else container.columns(2)[0]
    return col.multiselect(FRIENDLY_COLS["model"], model_opts, default=default_models)

def _render_filtros_equipes(container, team_opts: list, modo_mobile: bool, tournaments_sel_external: Optional[List]):
    """Renderiza os filtros de equipes e a busca rápida."""
    c1, c2 = container.columns(2)
    with c1:
        # Apenas para alinhar com o seletor de equipes
        st.write(f"**{len(tournaments_sel_external or []):d} torneios selecionados**")

    teams_sel = c2.multiselect(
        "Equipe (Casa ou Visitante)", team_opts,
        default=[] if modo_mobile else team_opts
    )
    q_team = container.text_input(
        "🔍 Buscar equipe (Casa/Visitante)",
        placeholder="Digite parte do nome da equipe..."
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
    tournaments_sel_external: Optional[List] = None
) -> dict:
    """Renderiza a interface de filtros principal e retorna as seleções do usuário."""
    # --- 1. Extração de Opções ---
    model_opts = sorted(df["model"].dropna().unique()) if "model" in df.columns else []
    team_opts = sorted(pd.concat([df["home"], df["away"]]).dropna().astype(str).unique()) if _exists(df, "home", "away") else []
    bet_opts = sorted(df["bet_suggestion"].dropna().unique()) if "bet_suggestion" in df.columns else []
    goal_opts = sorted(df["goal_bet_suggestion"].dropna().unique()) if "goal_bet_suggestion" in df.columns else []

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

    # --- 3. Renderização da UI ---
    target = st.sidebar if not modo_mobile else st
    container = target.expander("🔎 Filtros", expanded=not modo_mobile)

    models_sel = _render_filtros_modelos(container, model_opts, default_models, modo_mobile)
    teams_sel, q_team = _render_filtros_equipes(
        container, team_opts, modo_mobile, tournaments_sel_external
    )
    bet_sel, goal_sel = _render_filtros_sugestoes(container, bet_opts, goal_opts)
    selected_date_range = _render_filtros_periodo(container, min_date, max_date)
    sel_h, sel_d, sel_a = _render_filtros_odds(container, df)

    # --- 4. Sincronização e Retorno ---
    try:
        st.query_params["model"] = models_sel or []
    except Exception:
        pass  # Pode falhar em alguns contextos de execução

    return {
        "tournaments_sel": tournaments_sel_external or [],
        "models_sel": models_sel,
        "teams_sel": teams_sel,
        "bet_sel": bet_sel,
        "goal_sel": goal_sel,
        "selected_date_range": selected_date_range,
        "sel_h": sel_h, "sel_d": sel_d, "sel_a": sel_a,
        "q_team": q_team
    }


def _prepare_display_data(row: pd.Series) -> dict:
    """Prepara todos os dados necessários para a exibição de uma linha."""
    dt_txt = row["date"].strftime("%d/%m %H:%M") if ("date" in row.index and pd.notna(row["date"])) else "N/A"

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
        "final_score": f"{int(row.get('result_home', 0))}-{int(row.get('result_away', 0))}" if pd.notna(row.get("result_home")) else "—"
    }

def _render_over_under_section(row: pd.Series, df: pd.DataFrame):
    """Renderiza a seção de 'Over/Under' dentro do expander."""
    st.markdown("---")
    st.markdown("**Over/Under (Prob. — Odd)**")

    under_lines = [
        f"- **Under {v}:** {_po(row, f'prob_under_{v}', f'odds_match_goals_{v}_under')}"
        for v in GOAL_MARKET_THRESHOLDS if _exists(df, f"prob_under_{v.replace('.', '_')}")
    ]
    if under_lines:
        st.markdown("\n".join(under_lines), unsafe_allow_html=True)

    over_lines = [
        f"- **Over {v}:** {_po(row, f'prob_over_{v}', f'odds_match_goals_{v}_over')}"
        for v in GOAL_MARKET_THRESHOLDS if _exists(df, f"prob_over_{v.replace('.', '_')}")
    ]
    if over_lines:
        st.markdown("\n".join(over_lines), unsafe_allow_html=True)

def _render_expander_details(row: pd.Series, data: dict, df: pd.DataFrame):
    """Renderiza o conteúdo dentro do st.expander para a visualização em lista."""
    with st.expander("Detalhes, Probabilidades & Odds"):
        # Seção 1: Sugestões e Probabilidades 1x2
        st.markdown(
            f"""
            - **Sugestão:** {green_html(data["aposta_txt"])} {data["badge_bet"]}
            - **Sugestão de Gols:** {green_html(data["gols_txt"])} {data["badge_goal"]}
            - **Odds 1x2:** {green_html(fmt_odd(row.get('odds_H')))} / \
                {green_html(fmt_odd(row.get('odds_D')))} / \
                {green_html(fmt_odd(row.get('odds_A')))}
            - **Prob. (H/D/A):** {green_html(fmt_prob(row.get('prob_H')))} / \
                {green_html(fmt_prob(row.get('prob_D')))} / \
                {green_html(fmt_prob(row.get('prob_A')))}
            """,
            unsafe_allow_html=True
        )

        # Seção 2: Over/Under
        _render_over_under_section(row, df)

        # Seção 3: BTTS
        if _exists(df, "prob_btts_yes", "prob_btts_no"):
            st.markdown("---")
            st.markdown("**BTTS (Prob. — Odd)**")
            st.markdown(f"- **Ambos marcam — Sim:** {_po(row, 'prob_btts_yes', 'odds_btts_yes')}", unsafe_allow_html=True)
            st.markdown(f"- **Ambos marcam — Não:** {_po(row, 'prob_btts_no', 'odds_btts_no')}", unsafe_allow_html=True)

def display_list_view(df: pd.DataFrame):
    """Renderiza uma lista de jogos em formato de cards para visualização mobile."""
    for _, row in df.iterrows():
        data = _prepare_display_data(row)

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{data['title']}**")
                st.caption(data['cap_line'])
                st.markdown(
                    f'''
                    <div class="info-grid">
                        <div><span class="text-label">{data["badge_res"]} 🎯 Resultado:</span> {green_html(data["result_txt"])}</div>
                        <div><span class="text-label">{data["badge_bet"]} 💡 Sugestão Aposta:</span> {green_html(data["aposta_txt"])}</div>
                        <div><span class="text-label">{data["badge_goal"]} ⚽ Sugestão Gols:</span> {green_html(data["gols_txt"])}</div>
                        <div><span class="text-label">{data["badge_btts_pred"]} 🥅 Ambos Marcam:</span> {green_html(data["btts_pred_txt"])}</div>
                        <div><span class="text-label">{data["badge_score"]} 📊 Placar Previsto:</span> {green_html(data["score_txt"])}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            with c2:
                badge_class = "badge-finished" if data["is_finished"] else "badge-wait"
                st.markdown(f'<span class="badge {badge_class}">{data["status_txt"]}</span>', unsafe_allow_html=True)
                if data["is_finished"]:
                    st.markdown(f"**Placar Final:** {data['final_score']}")

            _render_expander_details(row, data, df)

            st.markdown('</div>', unsafe_allow_html=True)
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
