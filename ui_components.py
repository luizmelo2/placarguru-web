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

def display_list_view(df: pd.DataFrame):
    for _, row in df.iterrows():
        dt_txt = row["date"].strftime("%d/%m %H:%M") if ("date" in df.columns and pd.notna(row["date"])) else "N/A"
        title = f"{dt_txt} • {row.get('home','?')} vs {row.get('away','?')}"
        status_txt = status_label(row.get("status","N/A"))

        # badges (resultado e placar)
        hit_res   = eval_result_pred_row(row)
        hit_score = eval_score_pred_row(row)
        badge_res   = "✅" if hit_res is True else ("❌" if hit_res is False else "")
        badge_score = "✅" if hit_score is True else ("❌" if hit_score is False else "")

        # sugestões + avaliação (com fallback amigável)
        aposta_txt = market_label(row.get('bet_suggestion'))
        gols_txt   = market_label(row.get('goal_bet_suggestion'))

        hit_bet  = eval_bet_row(row)
        hit_goal = eval_goal_row(row)
        badge_bet  = "✅" if hit_bet is True else ("❌" if hit_bet is False else "")
        badge_goal = "✅" if hit_goal is True else ("❌" if hit_goal is False else "")

        # Nova previsão "Ambos Marcam"
        btts_pred = predict_btts_from_prob(row)
        hit_btts_pred = evaluate_market(btts_pred, row.get("result_home"), row.get("result_away"))
        badge_btts_pred = "✅" if hit_btts_pred is True else ("❌" if hit_btts_pred is False else "")

        # previsões com fallback amigável e odds
        result_txt = f"{market_label(row.get('result_predicted'))} {get_prob_and_odd_for_market(row, row.get('result_predicted'))}"
        score_txt  = fmt_score_pred_text(row.get('score_predicted'))
        aposta_txt = f"{market_label(row.get('bet_suggestion'))} {get_prob_and_odd_for_market(row, row.get('bet_suggestion'))}"
        gols_txt   = f"{market_label(row.get('goal_bet_suggestion'))} {get_prob_and_odd_for_market(row, row.get('goal_bet_suggestion'))}"
        btts_pred_txt = f"{market_label(btts_pred, default='-')} {get_prob_and_odd_for_market(row, btts_pred)}"

        # confiança AO LADO da previsão (e NÃO no caption)
        conf_txt = conf_badge(row)  # ex.: "🟢 Confiança: Alta"
        conf_html = f'<span class="conf-inline">({conf_txt})</span>' if conf_txt else ""

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{title}**")
                # caption SEM confiança
                cap_line = f"{tournament_label(row.get('tournament_id'))} • Modelo {row.get('model','—')}"
                st.caption(cap_line)

                # Grid com Confiança ao lado da PREV.
                st.markdown(
                    f'''
                    <div class="info-grid">
                        <div><span class="text-label">{badge_res} 🎯 Resultado:</span> {green_html(result_txt)} </div>
                        <div><span class="text-label">{badge_bet} 💡 Sugestão Aposta:</span> {green_html(aposta_txt)} </div>
                        <div><span class="text-label">{badge_goal} ⚽ Sugestão Gols:</span> {green_html(gols_txt)}</div>
                        <div><span class="text-label">{badge_btts_pred} 🥅 Ambos Marcam:</span> {green_html(btts_pred_txt)}</div>
                        <div><span class="text-label">{badge_score} 📊 Placar Previsto:</span> {green_html(score_txt)} </div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            with c2:
                badge_class = "badge-finished" if norm_status_key(row.get("status","")) in FINISHED_TOKENS else "badge-wait"
                st.markdown(f'<span class="badge {badge_class}">{status_txt}</span>', unsafe_allow_html=True)
                if norm_status_key(row.get("status","")) in FINISHED_TOKENS:
                    rh, ra = row.get("result_home"), row.get("result_away")
                    final_txt = f"{int(rh)}-{int(ra)}" if pd.notna(rh) and pd.notna(ra) else "—"
                    st.markdown(f"**Placar Final:** {final_txt}")

            # Detalhes com Prob/Odds (valores verdes)
            with st.expander("Detalhes, Probabilidades & Odds"):
                st.markdown(
                    f'''
                    - **Sugestão:** {green_html(aposta_txt)} {badge_bet}
                    - **Sugestão de Gols:** {green_html(gols_txt)} {badge_goal}
                    - **Odds 1x2:** {green_html(fmt_odd(row.get('odds_H')))} / {green_html(fmt_odd(row.get('odds_D')))} / {green_html(fmt_odd(row.get('odds_A')))}
                    - **Prob. (H/D/A):** {green_html(fmt_prob(row.get('prob_H')))} / {green_html(fmt_prob(row.get('prob_D')))} / {green_html(fmt_prob(row.get('prob_A')))}
                    ''',
                    unsafe_allow_html=True
                )

                st.markdown("---")
                st.markdown("**Over/Under (Prob. — Odd)**")

                under_lines = []
                if _exists(df, "prob_under_0_5"): under_lines.append(f"- **Under 0.5:** {_po(row, 'prob_under_0_5', 'odds_match_goals_0.5_under')}")
                if _exists(df, "prob_under_1_5"): under_lines.append(f"- **Under 1.5:** {_po(row, 'prob_under_1_5', 'odds_match_goals_1.5_under')}")
                if _exists(df, "prob_under_2_5"): under_lines.append(f"- **Under 2.5:** {_po(row, 'prob_under_2_5', 'odds_match_goals_2.5_under')}")
                if _exists(df, "prob_under_3_5"): under_lines.append(f"- **Under 3.5:** {_po(row, 'prob_under_3_5', 'odds_match_goals_3.5_under')}")
                if under_lines:
                    st.markdown("\n".join(under_lines), unsafe_allow_html=True)

                over_lines = []
                if _exists(df, "prob_over_0_5"): over_lines.append(f"- **Over 0.5:** {_po(row, 'prob_over_0_5', 'odds_match_goals_0.5_over')}")
                if _exists(df, "prob_over_1_5"): over_lines.append(f"- **Over 1.5:** {_po(row, 'prob_over_1_5', 'odds_match_goals_1.5_over')}")
                if _exists(df, "prob_over_2_5"): over_lines.append(f"- **Over 2.5:** {_po(row, 'prob_over_2_5', 'odds_match_goals_2.5_over')}")
                if _exists(df, "prob_over_3_5"): over_lines.append(f"- **Over 3.5:** {_po(row, 'prob_over_3_5', 'odds_match_goals_3.5_over')}")
                if over_lines:
                    st.markdown("\n".join(over_lines), unsafe_allow_html=True)

                if _exists(df, "prob_btts_yes") or _exists(df, "prob_btts_no"):
                    st.markdown("---")
                    st.markdown("**BTTS (Prob. — Odd)**")
                    if _exists(df, "prob_btts_yes"):
                        st.markdown(f"- **Ambos marcam — Sim:** { _po(row, 'prob_btts_yes', 'odds_btts_yes') }", unsafe_allow_html=True)
                    if _exists(df, "prob_btts_no"):
                        st.markdown(f"- **Ambos marcam — Não:** { _po(row, 'prob_btts_no', 'odds_btts_no') }", unsafe_allow_html=True)

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
