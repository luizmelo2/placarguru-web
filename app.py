import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import timedelta, date, datetime
from typing import Any, Tuple, Optional, List
import os
from zoneinfo import ZoneInfo  # Python 3.9+

# --- novos imports para baixar a release e controlar cache/tempo ---
import requests
import tempfile, time
from io import BytesIO
from email.utils import parsedate_to_datetime
from fpdf import FPDF, XPos, YPos

# Importa funções e constantes do utils.py
from utils import (
    _exists, fetch_release_file, RELEASE_URL, load_data, FRIENDLY_COLS,
    tournament_label, market_label, norm_status_key, fmt_score_pred_text,
    status_label, eval_result_pred_row, eval_score_pred_row, eval_bet_row,
    eval_goal_row, eval_sugestao_combo_row, green_html, FINISHED_TOKENS, fmt_odd, fmt_prob, _po,
    normalize_pred_code, evaluate_market, parse_score_pred, get_prob_and_odd_for_market,
    predict_btts_from_prob, safe_btts_code_from_row
)

# ============================
# Configuração da página
# ============================
st.set_page_config(
    layout="wide",
    page_title="Placar Guru",
    initial_sidebar_state="collapsed",
)

# Toggle manual de modo mobile (controle explícito para layout responsivo)
col_m1, col_m2 = st.columns([1, 4])
with col_m1:
    MODO_MOBILE = st.toggle("📱 Mobile", value=True)
with col_m2:
    st.title("Placar Guru")

# --- Estilos mobile-first + cores ---
st.markdown('''
<style>
html, body, .stApp { font-size: 16px; }
@media (max-width: 768px) {
  html, body, .stApp { font-size: 17px; }
  section[data-testid="stSidebar"] { display:none !important; }
}

/* Títulos menores */
h1 { font-size: 1.6rem; line-height: 1.2; margin-bottom: .25rem; }
h2 { font-size: 1.25rem; line-height: 1.25; margin: .5rem 0 .25rem 0; }
h3 { font-size: 1.10rem; line-height: 1.3; margin: .35rem 0 .15rem 0; }
@media (max-width: 768px) {
  h1 { font-size: 1.35rem; }
  h2 { font-size: 1.15rem; }
  h3 { font-size: 1.00rem; }
}

/* Containers e componentes */
.block-container { padding-top: 0.5rem !important; }
.card {
  border: 1px solid #1f2937; border-radius: 14px; padding: 12px;
  background: #0b0b0b; box-shadow: 0 1px 8px rgba(0,0,0,.2);
}
.badge { padding: 2px 8px; border-radius: 999px; font-weight: 600; }
.badge-ok { background:#14532d; color:#d1fae5; }
.badge-bad { background:#7f1d1d; color:#fee2e2; }
.badge-wait { background:#334155; color:#e2e8f0; }
.badge-finished { background: #1e3a8a; color: #dbeafe; } /* Royal Blue */
button, .stButton>button { border-radius: 12px; padding: 10px 14px; font-weight: 600; }
div[data-testid="stExpander"] summary { padding: 10px 12px; font-size: 1.05rem; font-weight: 700; }
.stDataFrame { overflow-x: auto; }

/* Tipografia/cores dentro dos cards */
.text-label { color:#9CA3AF; font-weight:600; }   /* rótulos (Prev., Placar, etc.) */
.text-muted { color:#9CA3AF; }
.text-strong { color:#E5E7EB; font-weight:700; }

/* ÚNICA cor destaque (verde) para previsão, placar, sugestões, probabilidades e odds */
.accent-green { color:#22C55E; font-weight:700; }
.text-odds { color: #9CA3AF; font-size: 0.9em; margin-left: 0.5rem; }

.value-soft { color:#E5E7EB; }
.info-line { margin-top:.25rem; }
.info-line > .sep { color:#6B7280; margin: 0 .35rem; }

/* Grid de detalhes */
.info-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.1rem 0.8rem;
  margin-top: 0.25rem;
}
@media (max-width: 768px) {
  .info-grid {
    grid-template-columns: 1fr;
    gap: 0.2rem;
  }
}

/* Caixa do filtro (C2) */
.tourn-box {
  border: 1px solid #1f2937; border-radius: 12px; padding: 12px;
  background: #0c0c0c; margin-bottom: 8px;
}
.tourn-title { font-weight:800; font-size:1.05rem; }

/* Confiança ao lado da Previsão */
.conf-inline { margin-left: .4rem; color:#9CA3AF; font-weight:600; white-space:nowrap; }
</style>
''', unsafe_allow_html=True)

# ============================
# Exibição amigável
# ============================
def apply_friendly_for_display(df: pd.DataFrame) -> pd.DataFrame:
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
    if "prob_btts_yes" in out.columns and "prob_btts_no" in out.columns:
        out["btts_prediction"] = out.apply(predict_btts_from_prob, axis=1).apply(market_label, default="-")


    return out.rename(columns=FRIENDLY_COLS)

# ========= Badge de confiança (opcional no caption) =========
def conf_badge(row):
    vals = [row.get("prob_H"), row.get("prob_D"), row.get("prob_A") ]
    if any(pd.isna(v) for v in vals): return ""
    try:
        conf = max(vals) * 100.0
    except Exception:
        return ""
    if np.isnan(conf): return ""
    if conf >= 70: return "🟢 Conf. Alta"
    if conf >= 55: return "🟡 Conf. Média"
    return "🟠 Conf. Baixa"

# ============================
# UI de Filtros (sem Status)
# ============================
def filtros_ui(df: pd.DataFrame, tournaments_sel_external: Optional[List]=None) -> dict:
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

# ============================
# Cards (lista) — valores verdes (inclui Prob/Odds)
# ============================
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

def generate_pdf_report(df: pd.DataFrame):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)

    title = f"Relatório de Jogos - {datetime.now().strftime('%d/%m/%Y')}"
    pdf.cell(0, 10, title, border=0, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("helvetica", "B", 8)

    # Cabeçalho da tabela
    headers = ["Data", "Casa", "Visitante", "Prev.", "Placar", "Sugestão", "Gols"]
    col_widths = [22, 35, 35, 15, 20, 30, 30]
    for h, w in zip(headers, col_widths):
        pdf.cell(w, 8, h, border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.ln()

    # Linhas
    pdf.set_font("helvetica", "", 8)
    for _, row in df.iterrows():
        data_txt = row["date"].strftime("%d/%m %H:%M") if pd.notna(row.get("date")) else ""
        pdf.cell(col_widths[0], 8, str(data_txt)[:16], border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_widths[1], 8, str(row.get("home", ""))[:18], border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_widths[2], 8, str(row.get("away", ""))[:18], border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_widths[3], 8, str(market_label(row.get("result_predicted")))[:10], border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_widths[4], 8, str(fmt_score_pred_text(row.get("score_predicted")))[:10], border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_widths[5], 8, str(market_label(row.get("bet_suggestion")))[:18], border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.cell(col_widths[6], 8, str(market_label(row.get("goal_bet_suggestion")))[:18], border=1, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    out = pdf.output()
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    else:
        return out.encode("latin-1", "ignore")

# ----------------------------
# Função de acurácia (reuso)
# ----------------------------
def compute_acc2(ok_mask: pd.Series, bad_mask: pd.Series):
    total = int((ok_mask | bad_mask).sum())
    correct = int(ok_mask.sum())
    acc = (correct / total * 100.0) if total > 0 else np.nan
    return acc, correct, total

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
        flt = filtros_ui(df, tournaments_sel_external=top_tournaments_sel)
        tournaments_sel, models_sel, teams_sel = flt["tournaments_sel"], flt["models_sel"], flt["teams_sel"]
        bet_sel, goal_sel = flt["bet_sel"], flt["goal_sel"]
        selected_date_range, selH, selD, selA = flt["selected_date_range"], flt["selH"], flt["selD"], flt["selA"]
        q_team = flt["q_team"]

        use_list_view = True if MODO_MOBILE else st.sidebar.checkbox("Usar visualização em lista (mobile)", value=False)

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
            final_mask &= ((df["odds_H"] >= selH[0]) & (df["odds_H"] <= selH[1])) | (df["odds_H"].isna())
        if "odds_D" in df.columns:
            final_mask &= ((df["odds_D"] >= selD[0]) & (df["odds_D"] <= selD[1])) | (df["odds_D"].isna())
        if "odds_A" in df.columns:
            final_mask &= ((df["odds_A"] >= selA[0]) & (df["odds_A"] <= selA[1])) | (df["odds_A"].isna())

        df_filtered = df[final_mask]

        # Abas Agendados x Finalizados (KPIs só em Finalizados)
        if df_filtered.empty:
            st.warning("Nenhum dado corresponde aos filtros atuais.")
        else:
            status_norm_all = df_filtered["status"].astype(str).map(norm_status_key) if "status" in df_filtered.columns else pd.Series("", index=df_filtered.index)
            df_ag  = df_filtered[status_norm_all != "finished"]
            df_fin = df_filtered[status_norm_all == "finished"]

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
                df_fin = df_fin[df_fin["date"].dt.date.between(_start, _end) | df_fin["date"].isna()]

            if "date" in df_fin.columns:
                df_fin = df_fin.sort_values("date", ascending=False, na_position="last")
            # --------------------------------------------------------------------------

            tab_ag, tab_fin = st.tabs(["🗓️ Agendados", "✅ Finalizados"])

            # --- ABA AGENDADOS (sem KPIs) ---
            with tab_ag:
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
                        st.dataframe(
                            apply_friendly_for_display(df_ag[
                                [c for c in ["date","home","away","tournament_id","model","status",
                                             "result_predicted","score_predicted","bet_suggestion","goal_bet_suggestion", "btts_prediction",
                                             "odds_H","odds_D","odds_A","result_home","result_away"] if c in df_ag.columns]
                            ]),
                            use_container_width=True, hide_index=True
                        )

            # --- ABA FINALIZADOS (com KPIs e gráfico) ---
            with tab_fin:
                if df_fin.empty:
                    st.info("Sem jogos finalizados neste recorte.")
                else:
                    if not st.checkbox("Ocultar lista de jogos", value=False):
                        if use_list_view:
                            display_list_view(df_fin)
                        else:
                            st.dataframe(
                                apply_friendly_for_display(df_fin[
                                    [c for c in ["date","home","away","tournament_id","model","status",
                                                 "result_predicted","score_predicted","bet_suggestion","goal_bet_suggestion", "btts_prediction",
                                                 "odds_H","odds_D","odds_A","result_home","result_away"] if c in df_fin.columns]
                                ]),
                                use_container_width=True, hide_index=True
                            )

                    # --- Função para preparar dados para o novo gráfico de acurácia diária ---
                    def prepare_accuracy_chart_data(df: pd.DataFrame) -> pd.DataFrame:
                        def _evaluate_row(row):
                            correct = 0
                            total = 0

                            markets_to_check = [
                                eval_result_pred_row(row),
                                eval_bet_row(row),
                                eval_goal_row(row),
                                eval_score_pred_row(row),
                            ]

                            btts_pred = predict_btts_from_prob(row)
                            markets_to_check.append(evaluate_market(btts_pred, row.get("result_home"), row.get("result_away")))

                            for result in markets_to_check:
                                if result is not None:
                                    total += 1
                                    if result:
                                        correct += 1

                            return pd.Series([correct, total], index=['correct_count', 'total_count'])

                        if df.empty or 'date' not in df.columns:
                            return pd.DataFrame()

                        counts = df.apply(_evaluate_row, axis=1)
                        df_counts = pd.concat([df[['date', 'tournament_id']], counts], axis=1)
                        df_counts.dropna(subset=['date', 'tournament_id'], inplace=True)

                        df_counts['day'] = df_counts['date'].dt.date
                        daily_agg = df_counts.groupby(['day', 'tournament_id']).sum(numeric_only=True).reset_index()

                        daily_agg['accuracy_rate'] = daily_agg.apply(
                            lambda row: (row['correct_count'] / row['total_count'] * 100) if row['total_count'] > 0 else 0,
                            axis=1
                        )

                        daily_agg['tournament_id'] = daily_agg['tournament_id'].apply(tournament_label)
                        chart_df = daily_agg.rename(columns={
                            'day': 'Data',
                            'tournament_id': 'Campeonato',
                            'accuracy_rate': 'Taxa de Acerto (%)'
                        })

                        return chart_df[['Data', 'Campeonato', 'Taxa de Acerto (%)']]

                    def get_best_model_by_market(df: pd.DataFrame) -> pd.DataFrame:
                        """
                        Calcula o melhor modelo por campeonato e mercado de aposta com base na taxa de acerto.
                        """
                        if df.empty or 'model' not in df.columns or 'tournament_id' not in df.columns:
                            return pd.DataFrame()

                        # 1. Calcula o acerto para cada mercado de interesse
                        df['hit_result'] = df.apply(eval_result_pred_row, axis=1)
                        df['hit_bet'] = df.apply(eval_bet_row, axis=1)
                        df['hit_goal'] = df.apply(eval_goal_row, axis=1)
                        df['hit_btts'] = df.apply(lambda row: evaluate_market(predict_btts_from_prob(row), row.get("result_home"), row.get("result_away")), axis=1)

                        # Converte True/False para 1/0 para agregação, mantendo None como NaN
                        for col in ['hit_result', 'hit_bet', 'hit_goal', 'hit_btts']:
                            df[col] = df[col].apply(lambda x: 1 if x is True else (0 if x is False else np.nan))

                        # 2. Reestrutura os dados para formato longo (tidy)
                        id_vars = ['tournament_id', 'model']
                        value_vars = ['hit_result', 'hit_bet', 'hit_goal', 'hit_btts']
                        df_melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='mercado', value_name='acerto')
                        df_melted.dropna(subset=['acerto'], inplace=True)

                        # 3. Agrupa para calcular acertos e totais
                        agg = df_melted.groupby(['tournament_id', 'model', 'mercado']).agg(
                            total_acertos=('acerto', 'sum'),
                            total_jogos=('acerto', 'count')
                        ).reset_index()

                        # 4. Calcula a taxa de acerto
                        agg['taxa_acerto'] = (agg['total_acertos'] / agg['total_jogos'] * 100).round(2)

                        # 5. Encontra o melhor modelo para cada campeonato e mercado
                        best_model_idx = agg.groupby(['tournament_id', 'mercado'])['taxa_acerto'].idxmax()
                        df_best = agg.loc[best_model_idx].copy()

                        # 6. Formata o DataFrame final
                        df_best['mercado'] = df_best['mercado'].map({
                            'hit_result': 'Resultado Final',
                            'hit_bet': 'Sugestão de Aposta',
                            'hit_goal': 'Sugestão de Gols',
                            'hit_btts': 'Ambos Marcam (Prob)'
                        })
                        df_best['tournament_id'] = df_best['tournament_id'].apply(tournament_label)

                        df_final = df_best[['tournament_id', 'mercado', 'model', 'taxa_acerto', 'total_jogos']]
                        df_final = df_final.rename(columns={
                            'tournament_id': 'Campeonato',
                            'mercado': 'Mercado de Aposta',
                            'model': 'Melhor Modelo',
                            'taxa_acerto': 'Taxa de Acerto (%)',
                            'total_jogos': 'Total de Jogos Avaliados'
                        })

                        return df_final.sort_values(by=['Campeonato', 'Mercado de Aposta']).reset_index(drop=True)

                    # ---------- KPIs e gráfico por modelo (apenas finalizados) ----------
                    rh = df_fin.get("result_home", pd.Series(index=df_fin.index, dtype="float"))
                    ra = df_fin.get("result_away", pd.Series(index=df_fin.index, dtype="float"))
                    mask_valid = rh.notna() & ra.notna()

                    # Códigos reais H/D/A
                    real_code = pd.Series(index=df_fin.index, dtype="object")
                    real_code.loc[mask_valid & (rh > ra)] = "H"
                    real_code.loc[mask_valid & (rh == ra)] = "D"
                    real_code.loc[mask_valid & (rh < ra)] = "A"

                    def compute_acc(ok_mask: pd.Series, bad_mask: pd.Series):
                        total = int((ok_mask | bad_mask).sum())
                        correct = int(ok_mask.sum())
                        acc = (correct / total * 100.0) if total > 0 else np.nan
                        return acc, correct, total

                    selected_models = list(df_fin["model"].dropna().unique()) if "model" in df_fin.columns else []
                    multi_model = len(selected_models) > 1

                    if multi_model:
                        rows = []
                        for m in selected_models:
                            sub = df_fin[df_fin["model"] == m]
                            if sub.empty:
                                continue

                            rh_s = sub.get("result_home", pd.Series(index=sub.index, dtype="float"))
                            ra_s = sub.get("result_away", pd.Series(index=sub.index, dtype="float"))
                            mv_s = rh_s.notna() & ra_s.notna()

                            real_s = pd.Series(index=sub.index, dtype="object")
                            real_s.loc[mv_s & (rh_s > ra_s)] = "H"
                            real_s.loc[mv_s & (rh_s == ra_s)] = "D"
                            real_s.loc[mv_s & (rh_s < ra_s)] = "A"

                            # Resultado Previsto
                            pred_eval_s = sub.apply(eval_result_pred_row, axis=1)
                            pred_correct_s = pred_eval_s == True
                            pred_wrong_s   = pred_eval_s == False

                            # Sugestão de Aposta
                            bet_codes_s = sub.get("bet_suggestion", pd.Series(index=sub.index, dtype="object"))
                            bet_eval_s = pd.Series(index=sub.index, dtype="object")
                            for idx in sub.index:
                                bet_eval_s.loc[idx] = evaluate_market(bet_codes_s.loc[idx], rh_s.loc[idx], ra_s.loc[idx]) if mv_s.loc[idx] else None
                            bet_correct_s = bet_eval_s == True
                            bet_wrong_s   = bet_eval_s == False

                            # Sugestão de Gols
                            goal_codes_s = sub.get("goal_bet_suggestion", pd.Series(index=sub.index, dtype="object"))
                            goal_eval_s = pd.Series(index=sub.index, dtype="object")
                            for idx in sub.index:
                                goal_eval_s.loc[idx] = evaluate_market(goal_codes_s.loc[idx], ra_s.loc[idx], rh_s.loc[idx]) if mv_s.loc[idx] else None
                            goal_correct_s = goal_eval_s == True
                            goal_wrong_s   = goal_eval_s == False

                            # Separate BTTS from goal suggestions
                            btts_mask_s = goal_codes_s.astype(str).str.lower().isin(['btts_yes', 'btts_no'])
                            btts_correct_s = goal_correct_s & btts_mask_s
                            btts_wrong_s = goal_wrong_s & btts_mask_s

                            # Exclude BTTS from the general "goal" metric
                            goal_correct_s_no_btts = goal_correct_s & ~btts_mask_s
                            goal_wrong_s_no_btts   = goal_wrong_s & ~btts_mask_s

                            # Placar Previsto
                            score_eval_s = pd.Series(index=sub.index, dtype="object")
                            if "score_predicted" in sub.columns:
                                for idx in sub.index:
                                    if mv_s.loc[idx]:
                                        ph, pa = parse_score_pred(sub.at[idx, "score_predicted"])
                                        if ph is None or pa is None:
                                            score_eval_s.loc[idx] = None
                                        else:
                                            try:
                                                score_eval_s.loc[idx] = (int(rh_s.loc[idx]) == int(ph)) and (int(ra_s.loc[idx]) == int(pa))
                                            except Exception:
                                                score_eval_s.loc[idx] = None
                                    else:
                                        score_eval_s.loc[idx] = None
                            score_correct_s = score_eval_s == True
                            score_wrong_s   = score_eval_s == False

                            # Sugestão Combo
                            combo_eval_s = sub.apply(eval_sugestao_combo_row, axis=1)
                            combo_correct_s = combo_eval_s == True
                            combo_wrong_s   = combo_eval_s == False

                            # Métricas por modelo
                            acc_pred, c_pred, t_pred = compute_acc(pred_correct_s, pred_wrong_s)
                            acc_bet,  c_bet,  t_bet  = compute_acc(bet_correct_s,  bet_wrong_s)
                            acc_goal, c_goal, t_goal = compute_acc(goal_correct_s_no_btts, goal_wrong_s_no_btts)
                            acc_btts, c_btts, t_btts = compute_acc(btts_correct_s, btts_wrong_s)
                            acc_score, c_score, t_score = compute_acc(score_correct_s, score_wrong_s)
                            acc_combo, c_combo, t_combo = compute_acc(combo_correct_s, combo_wrong_s)

                            # Nova métrica BTTS (Prob)
                            btts_pred_s = sub.apply(predict_btts_from_prob, axis=1)
                            btts_pred_eval_s = pd.Series(index=sub.index, dtype="object")
                            for idx in sub.index:
                                btts_pred_eval_s.loc[idx] = evaluate_market(btts_pred_s.loc[idx], rh_s.loc[idx], ra_s.loc[idx]) if mv_s.loc[idx] else None
                            btts_pred_correct_s = btts_pred_eval_s == True
                            btts_pred_wrong_s   = btts_pred_eval_s == False
                            acc_btts_pred, c_btts_pred, t_btts_pred = compute_acc(btts_pred_correct_s, btts_pred_wrong_s)


                            rows += [
                                {"Modelo": m, "Métrica": "Resultado",            "Acerto (%)": 0 if np.isnan(acc_pred) else round(acc_pred,1), "Acertos": c_pred,  "Total Avaliado": t_pred},
                                {"Modelo": m, "Métrica": "Sugestão de Aposta",   "Acerto (%)": 0 if np.isnan(acc_bet)  else round(acc_bet,1),  "Acertos": c_bet,   "Total Avaliado": t_bet},
                                {"Modelo": m, "Métrica": "Sugestão Combo",       "Acerto (%)": 0 if np.isnan(acc_combo) else round(acc_combo,1), "Acertos": c_combo, "Total Avaliado": t_combo},
                                {"Modelo": m, "Métrica": "Sugestão de Gols",     "Acerto (%)": 0 if np.isnan(acc_goal) else round(acc_goal,1), "Acertos": c_goal,  "Total Avaliado": t_goal},
                                {"Modelo": m, "Métrica": "Ambos Marcam (Sugestão)", "Acerto (%)": 0 if np.isnan(acc_btts) else round(acc_btts,1), "Acertos": c_btts,  "Total Avaliado": t_btts},
                                {"Modelo": m, "Métrica": "Ambos Marcam (Prob)",  "Acerto (%)": 0 if np.isnan(acc_btts_pred) else round(acc_btts_pred,1), "Acertos": c_btts_pred, "Total Avaliado": t_btts_pred},
                                {"Modelo": m, "Métrica": "Placar Previsto",      "Acerto (%)": 0 if np.isnan(acc_score) else round(acc_score,1), "Acertos": c_score, "Total Avaliado": t_score},
                            ]

                        metrics_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Modelo","Métrica","Acerto (%)","Acertos","Total Avaliado"])

                        st.subheader("Percentual de acerto por modelo (apenas finalizados)")
                        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

                        # Gráfico de barras agrupadas por modelo
                        if not metrics_df.empty:
                            chart = (
                                alt.Chart(metrics_df)
                                .mark_bar()
                                .encode(
                                    x=alt.X('Métrica:N', title=''),
                                    y=alt.Y('Acerto (%):Q', scale=alt.Scale(domain=[0,100])),
                                    color='Modelo:N',
                                    xOffset='Modelo:N',
                                    tooltip=['Modelo:N','Métrica:N','Acertos:Q','Total Avaliado:Q', alt.Tooltip('Acerto (%):Q', format='.1f')]
                                )
                                .properties(height=240 if MODO_MOBILE else 280)
                            )
                            text = (
                                alt.Chart(metrics_df)
                                .mark_text(dy=-8)
                                .encode(
                                    x='Métrica:N',
                                    y='Acerto (%):Q',
                                    detail='Modelo:N',
                                    text=alt.Text('Acerto (%):Q', format='.1f'),
                                    color='Modelo:N'
                                )
                            )
                            st.altair_chart(chart + text, use_container_width=True)

                    else:
                        # Um único modelo/seleção: comportamento agregado
                        pred_eval = df_fin.apply(eval_result_pred_row, axis=1)
                        pred_correct = pred_eval == True
                        pred_wrong   = pred_eval == False

                        bet_codes = df_fin.get("bet_suggestion", pd.Series(index=df_fin.index, dtype="object"))
                        bet_eval = pd.Series(index=df_fin.index, dtype="object")
                        for idx in df_fin.index:
                            bet_eval.loc[idx] = evaluate_market(bet_codes.loc[idx], rh.loc[idx], ra.loc[idx]) if mask_valid.loc[idx] else None
                        bet_correct = bet_eval == True
                        bet_wrong   = bet_eval == False

                        goal_codes = df_fin.get("goal_bet_suggestion", pd.Series(index=df_fin.index, dtype="object"))
                        goal_eval = pd.Series(index=df_fin.index, dtype="object")
                        for idx in df_fin.index:
                            goal_eval.loc[idx] = evaluate_market(goal_codes.loc[idx], rh.loc[idx], ra.loc[idx]) if mask_valid.loc[idx] else None
                        goal_correct = goal_eval == True
                        goal_wrong   = goal_eval == False

                        # BTTS separado
                        btts_mask = goal_codes.astype(str).str.lower().isin(['btts_yes', 'btts_no'])
                        btts_correct = goal_correct & btts_mask
                        btts_wrong = goal_wrong & btts_mask

                        # Gols (excluindo BTTS)
                        goal_correct_no_btts = goal_correct & ~btts_mask
                        goal_wrong_no_btts   = goal_wrong & ~btts_mask

                        # métricas
                        acc_pred, c_pred, t_pred = compute_acc2(pred_correct, pred_wrong)
                        acc_bet,  c_bet,  t_bet  = compute_acc2(bet_correct,  bet_wrong)
                        acc_btts, c_btts, t_btts = compute_acc2(btts_correct, btts_wrong)
                        acc_goal, c_goal, t_goal = compute_acc2(goal_correct_no_btts, goal_wrong_no_btts)

                        # Sugestão Combo (modelo único)
                        combo_eval = df_fin.apply(eval_sugestao_combo_row, axis=1)
                        combo_correct = combo_eval == True
                        combo_wrong   = combo_eval == False
                        acc_combo, c_combo, t_combo = compute_acc2(combo_correct, combo_wrong)

                        # Nova métrica BTTS (Prob) - modelo único
                        btts_pred = df_fin.apply(predict_btts_from_prob, axis=1)
                        btts_pred_eval = pd.Series(index=df_fin.index, dtype="object")
                        for idx in df_fin.index:
                            btts_pred_eval.loc[idx] = evaluate_market(btts_pred.loc[idx], rh.loc[idx], ra.loc[idx]) if mask_valid.loc[idx] else None
                        btts_pred_correct = btts_pred_eval == True
                        btts_pred_wrong   = btts_pred_eval == False
                        acc_btts_pred, c_btts_pred, t_btts_pred = compute_acc2(btts_pred_correct, btts_pred_wrong)

                        st.subheader("Percentual de acerto (apenas finalizados)")
                        k1, k2, k3, k4, k5 = (st.container(), st.container(), st.container(), st.container(), st.container()) if MODO_MOBILE else st.columns(5)
                        k1.metric("Resultado", f"{0 if np.isnan(acc_pred) else round(acc_pred,1)}%", f"{c_pred}/{t_pred}")
                        k2.metric("Sugestão de Aposta", f"{0 if np.isnan(acc_bet) else round(acc_bet,1)}%", f"{c_bet}/{t_bet}")
                        k3.metric("Sugestão Combo", f"{0 if np.isnan(acc_combo) else round(acc_combo,1)}%", f"{c_combo}/{t_combo}")
                        k4.metric("Sugestão de Gols", f"{0 if np.isnan(acc_goal) else round(acc_goal,1)}%", f"{c_goal}/{t_goal}")
                        #k5.metric("Ambos Marcam (Sugestão)", f"{0 if np.isnan(acc_btts) else round(acc_btts,1)}%", f"{c_btts}/{t_btts}")
                        k5.metric("Ambos Marcam ", f"{0 if np.isnan(acc_btts_pred) else round(acc_btts_pred,1)}%", f"{c_btts_pred}/{t_btts_pred}")


                        metrics_df = pd.DataFrame({
                            "Métrica": ["Resultado", "Sugestão de Aposta", "Sugestão Combo", "Sugestão de Gols", "Ambos Marcam (Prob)"],
                            "Acerto (%)": [
                                0 if np.isnan(acc_pred) else round(acc_pred, 1),
                                0 if np.isnan(acc_bet) else round(acc_bet, 1),
                                0 if np.isnan(acc_combo) else round(acc_combo, 1),
                                0 if np.isnan(acc_goal) else round(acc_goal, 1),
                                0 if np.isnan(acc_btts_pred) else round(acc_btts_pred, 1),
                            ],
                            "Acertos": [c_pred, c_bet, c_combo, c_goal, c_btts_pred],
                            "Total Avaliado": [t_pred, t_bet, t_combo, t_goal, t_btts_pred],
                        })

                        chart = alt.Chart(metrics_df).mark_bar().encode(
                            x=alt.X('Métrica:N', title=''),
                            y=alt.Y('Acerto (%):Q', scale=alt.Scale(domain=[0, 100])),
                            tooltip=['Métrica:N', 'Acertos:Q', 'Total Avaliado:Q', alt.Tooltip('Acerto (%):Q', format='.1f')]
                        ).properties(height=220 if MODO_MOBILE else 260)
                        text = alt.Chart(metrics_df).mark_text(dy=-8).encode(
                            x='Métrica:N',
                            y='Acerto (%):Q',
                            text=alt.Text('Acerto (%):Q', format='.1f')
                        )
                        st.altair_chart(chart + text, use_container_width=True)

                    # --- Gráfico de linha de acurácia por dia/campeonato ---
                    st.subheader("Acurácia Diária por Campeonato")
                    accuracy_data = prepare_accuracy_chart_data(df_fin)
                    if not accuracy_data.empty:
                        line_chart = alt.Chart(accuracy_data).mark_line(point=True).encode(
                            x=alt.X('Data:T', title='Data'),
                            y=alt.Y('Taxa de Acerto (%):Q', scale=alt.Scale(domain=[0, 100]), title='Taxa de Acerto (%)'),
                            color='Campeonato:N',
                            tooltip=['Data:T', 'Campeonato:N', alt.Tooltip('Taxa de Acerto (%):Q', format='.1f')]
                        ).properties(
                            height=300,
                            title='Taxa de Acerto Diária por Campeonato'
                        )
                        st.altair_chart(line_chart, use_container_width=True)
                    else:
                        st.info("Não há dados suficientes para gerar o gráfico de acurácia diária.")

                    # --- Tabela de Melhor Modelo por Campeonato e Mercado ---
                    st.subheader("Melhor Modelo por Campeonato e Mercado")
                    best_model_data = get_best_model_by_market(df_fin.copy())
                    if not best_model_data.empty:
                        st.dataframe(best_model_data, use_container_width=True, hide_index=True)
                    else:
                        st.info("Não há dados suficientes para gerar a tabela de melhores modelos.")

        # --- Rodapé: Última Atualização (da release/servidor GitHub) ---
        st.markdown(
            '''
            <hr style="border: 0; border-top: 1px solid #1f2937; margin: 1rem 0 0.5rem 0;" />
            <div style="color:#9CA3AF; font-size:0.95rem;">
              <strong>Última atualização:</strong> %s
            </div>
            ''' % last_update_dt.strftime("%d/%m/%Y %H:%M"),
            unsafe_allow_html=True
        )
        # Botão para forçar atualização (limpa o cache de dados e re-executa o app)
        if st.button("🔄 Atualizar agora"):
            st.cache_data.clear()
            st.rerun()

except FileNotFoundError:
    st.error("FATAL: `PrevisaoJogos.xlsx` não encontrado.")
except Exception as e:
    st.error(f"Erro inesperado: {e}")
