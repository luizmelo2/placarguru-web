import streamlit as st
import pandas as pd
from typing import Optional, List
from datetime import date, timedelta

from utils import (
    FRIENDLY_COLS, market_label, tournament_label, status_label,
    eval_result_pred_row, eval_score_pred_row, eval_bet_row,
    eval_goal_row, predict_btts_from_prob, evaluate_market,
    get_prob_and_odd_for_market, fmt_score_pred_text,
    green_html, norm_status_key, FINISHED_TOKENS, _exists, _po, fmt_odd, fmt_prob
)

# ========= Badge de confiança (opcional no caption) =========
def conf_badge(row):
    vals = [row.get("prob_H"), row.get("prob_D"), row.get("prob_A") ]
    if any(pd.isna(v) for v in vals): return ""
    try:
        conf = max(vals) * 100.0
    except Exception:
        return ""
    if pd.isna(conf): return ""
    if conf >= 70: return "🟢 Conf. Alta"
    if conf >= 55: return "🟡 Conf. Média"
    return "🟠 Conf. Baixa"

def filtros_ui(df: pd.DataFrame, MODO_MOBILE: bool, tournaments_sel_external: Optional[List]=None) -> dict:
    model_opts  = sorted(df["model"].dropna().unique()) if "model" in df.columns else []

    if {"home", "away"}.issubset(df.columns):
        team_opts = pd.concat([df["home"], df["away"]], ignore_index=True).dropna()
        team_opts = sorted(team_opts.astype(str).unique())
    else:
        team_opts = []

    bet_opts  = sorted(df["bet_suggestion"].dropna().unique()) if "bet_suggestion" in df.columns else []
    goal_opts = sorted(df["goal_bet_suggestion"].dropna().unique()) if "goal_bet_suggestion" in df.columns else []

    # DEFAULTS: URL (?model=) ou "Combo"
    if model_opts:
        url_models_lower = [v.strip().lower() for v in st.session_state.model_init_raw]
        wanted_models = [m for m in model_opts if str(m).strip().lower() in url_models_lower]
        if not wanted_models:
            wanted_models = [m for m in model_opts if str(m).strip().lower() == "combo"]
        models_default = wanted_models or model_opts
    else:
        models_default = []

    # datas min/max
    if "date" in df.columns and df["date"].notna().any():
        min_date = df["date"].dropna().min().date()
        max_date = df["date"].dropna().max().date()
    else:
        min_date = max_date = None

    target = st.sidebar if not MODO_MOBILE else st
    container = target.expander("🔎 Filtros", expanded=not MODO_MOBILE)

    with container:
        # Modelos
        c1 = st.columns(1)[0] if MODO_MOBILE else st.columns(2)[0]
        with c1:
            models_sel = st.multiselect(FRIENDLY_COLS["model"], model_opts, default=models_default)

        # Times (o seletor de torneios veio do topo)
        c3, c4 = st.columns(2)
        with c3:
            tournaments_sel = tournaments_sel_external or []
        with c4:
            teams_sel = st.multiselect("Equipe (Casa ou Visitante)", team_opts, default=[] if MODO_MOBILE else team_opts)

        # Busca rápida por equipe
        q_team = st.text_input("🔍 Buscar equipe (Casa/Visitante)", placeholder="Digite parte do nome da equipe...")

        # Sugestões
        c5, c6 = st.columns(2)
        with c5:
            bet_sel = st.multiselect(FRIENDLY_COLS["bet_suggestion"], bet_opts, default=[], format_func=market_label)
        with c6:
            goal_sel = st.multiselect(FRIENDLY_COLS["goal_bet_suggestion"], goal_opts, default=[], format_func=market_label)

        # Período
        with st.expander("Período", expanded=False):
            selected_date_range = ()
            if min_date:
                today = date.today()
                cc1, cc2, cc3, cc4, cc5  = st.columns(5)
                with cc1:
                    if st.button("Hoje"): selected_date_range = (today, today)
                with cc2:
                    if st.button("Próx. 3 dias"): selected_date_range = (today, today + timedelta(days=3))
                with cc3:
                    if st.button("Últimos 3 dias"): selected_date_range = (today - timedelta(days=3), today)
                with cc4:
                    if st.button("Semana"):
                        start = today - timedelta(days=today.weekday())
                        end = start + timedelta(days=6)
                        selected_date_range = (start, end)
                with cc5:
                    if st.button("Limpar"): selected_date_range = ()

                if not selected_date_range:
                    selected_date_range = st.date_input(
                        "Período (intervalo)", value=(min_date, max_date),
                        min_value=min_date, max_value=max_date
                    )

            def _range(series: pd.Series, default=(0.0, 1.0)):
                s = series.dropna()
                return (float(s.min()), float(s.max())) if not s.empty else default

        # Odds
        with st.expander("Odds", expanded=False):
            selH = selD = selA = (0.0, 1.0)
            if "odds_H" in df.columns:
                minH, maxH = _range(df["odds_H"])
                selH = st.slider(FRIENDLY_COLS["odds_H"], minH, maxH, (minH, maxH))
            if "odds_D" in df.columns:
                minD, maxD = _range(df["odds_D"])
                selD = st.slider(FRIENDLY_COLS["odds_D"], minD, maxD, (minD, maxD))
            if "odds_A" in df.columns:
                minA, maxA = _range(df["odds_A"])
                selA = st.slider(FRIENDLY_COLS["odds_A"], minA, maxA, (minA, maxA))

    # Reflete estado na URL — apenas modelo
    try:
        st.query_params.update({"model": models_sel or []})
    except Exception:
        pass

    return dict(
        tournaments_sel=tournaments_sel, models_sel=models_sel, teams_sel=teams_sel,
        bet_sel=bet_sel, goal_sel=goal_sel, selected_date_range=selected_date_range,
        selH=selH, selD=selD, selA=selA, q_team=q_team
    )

def _prepare_display_data(row: pd.Series) -> dict:
    """Prepara todos os dados necessários para a exibição de uma linha."""
    dt_txt = row["date"].strftime("%d/%m %H:%M") if ("date" in row.index and pd.notna(row["date"])) else "N/A"

    # Avaliações de acerto
    hit_res = eval_result_pred_row(row)
    hit_score = eval_score_pred_row(row)
    hit_bet = eval_bet_row(row)
    hit_goal = eval_goal_row(row)
    btts_pred = predict_btts_from_prob(row)
    hit_btts_pred = evaluate_market(btts_pred, row.get("result_home"), row.get("result_away"))

    def _get_badge(hit_status):
        return "✅" if hit_status is True else ("❌" if hit_status is False else "")

    # Textos e odds
    result_txt = f"{market_label(row.get('result_predicted'))} {get_prob_and_odd_for_market(row, row.get('result_predicted'))}"
    score_txt = fmt_score_pred_text(row.get('score_predicted'))
    aposta_txt = f"{market_label(row.get('bet_suggestion'))} {get_prob_and_odd_for_market(row, row.get('bet_suggestion'))}"
    gols_txt = f"{market_label(row.get('goal_bet_suggestion'))} {get_prob_and_odd_for_market(row, row.get('goal_bet_suggestion'))}"
    btts_pred_txt = f"{market_label(btts_pred, default='-')} {get_prob_and_odd_for_market(row, btts_pred)}"

    return {
        "title": f"{dt_txt} • {row.get('home','?')} vs {row.get('away','?')}",
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

def _render_expander_details(row: pd.Series, data: dict, df: pd.DataFrame):
    """Renderiza o conteúdo dentro do st.expander para a visualização em lista."""
    with st.expander("Detalhes, Probabilidades & Odds"):
        # Seção 1: Sugestões e Probabilidades 1x2
        st.markdown(
            f"""
            - **Sugestão:** {green_html(data["aposta_txt"])} {data["badge_bet"]}
            - **Sugestão de Gols:** {green_html(data["gols_txt"])} {data["badge_goal"]}
            - **Odds 1x2:** {green_html(fmt_odd(row.get('odds_H')))} / {green_html(fmt_odd(row.get('odds_D')))} / {green_html(fmt_odd(row.get('odds_A')))}
            - **Prob. (H/D/A):** {green_html(fmt_prob(row.get('prob_H')))} / {green_html(fmt_prob(row.get('prob_D')))} / {green_html(fmt_prob(row.get('prob_A')))}
            """,
            unsafe_allow_html=True
        )

        # Seção 2: Over/Under
        st.markdown("---")
        st.markdown("**Over/Under (Prob. — Odd)**")

        under_lines = [
            f"- **Under {v}:** {_po(row, f'prob_under_{v}', f'odds_match_goals_{v}_under')}"
            for v in ["0.5", "1.5", "2.5", "3.5"] if _exists(df, f"prob_under_{v.replace('.', '_')}")
        ]
        if under_lines:
            st.markdown("\n".join(under_lines), unsafe_allow_html=True)

        over_lines = [
            f"- **Over {v}:** {_po(row, f'prob_over_{v}', f'odds_match_goals_{v}_over')}"
            for v in ["0.5", "1.5", "2.5", "3.5"] if _exists(df, f"prob_over_{v.replace('.', '_')}")
        ]
        if over_lines:
            st.markdown("\n".join(over_lines), unsafe_allow_html=True)

        # Seção 3: BTTS
        if _exists(df, "prob_btts_yes", "prob_btts_no"):
            st.markdown("---")
            st.markdown("**BTTS (Prob. — Odd)**")
            st.markdown(f"- **Ambos marcam — Sim:** {_po(row, 'prob_btts_yes', 'odds_btts_yes')}", unsafe_allow_html=True)
            st.markdown(f"- **Ambos marcam — Não:** {_po(row, 'prob_btts_no', 'odds_btts_no')}", unsafe_allow_html=True)

def display_list_view(df: pd.DataFrame):
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
