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
from utils import (_exists, fetch_release_file, RELEASE_URL, load_data, FRIENDLY_COLS, tournament_label, market_label, norm_status_key,
                   fmt_score_pred_text, status_label, eval_result_pred_row, eval_score_pred_row, eval_bet_row, eval_goal_row, green_html,
                   FINISHED_TOKENS, fmt_odd, fmt_prob, _po, normalize_pred_code, evaluate_market, parse_score_pred)

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
h1 {
  font-size: 1.6rem;       /* desktop */
  line-height: 1.2;
  margin-bottom: .25rem;
}
h2 {
  font-size: 1.25rem;      /* header */
  line-height: 1.25;
  margin: .5rem 0 .25rem 0;
}
h3 {
  font-size: 1.10rem;      /* subheader */
  line-height: 1.3;
  margin: .35rem 0 .15rem 0;
}
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
button, .stButton>button { border-radius: 12px; padding: 10px 14px; font-weight: 600; }
div[data-testid="stExpander"] summary { padding: 10px 12px; font-size: 1.05rem; font-weight: 700; }
.stDataFrame { overflow-x: auto; }

/* Tipografia/cores dentro dos cards */
.text-label { color:#9CA3AF; font-weight:600; }   /* rótulos (Prev., Placar, etc.) */
.text-muted { color:#9CA3AF; }
.text-strong { color:#E5E7EB; font-weight:700; }

/* ÚNICA cor destaque (verde) para previsão, placar, sugestões, probabilidades e odds */
.accent-green { color:#22C55E; font-weight:700; }

.value-soft { color:#E5E7EB; }
.info-line { margin-top:.25rem; }
.info-line > .sep { color:#6B7280; margin: 0 .35rem; }

/* Novo layout em grid para detalhes do jogo */
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
    if conf >= 65: return "🟢 Confiança: Alta"
    if conf >= 55: return "🟡 Confiança: Média"
    return "🟠 Confiança: Baixa"

# ============================
# UI de Filtros (sem Status)
# ============================
def filtros_ui(df: pd.DataFrame) -> dict:
    tourn_opts  = sorted(df["tournament_id"].dropna().unique().tolist()) if "tournament_id" in df.columns else []
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

        # Torneios e Times
        c3, c4 = st.columns(2)
        with c3:
            tournaments_sel = st.multiselect(FRIENDLY_COLS["tournament_id"], tourn_opts, default=tourn_opts, format_func=tournament_label)
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
                minH, maxH = _range(df["odds_H"]);
                selH = st.slider(FRIENDLY_COLS["odds_H"], minH, maxH, (minH, maxH))
            if "odds_D" in df.columns:
                minD, maxD = _range(df["odds_D"]);
                selD = st.slider(FRIENDLY_COLS["odds_D"], minD, maxD, (minD, maxD))
            if "odds_A" in df.columns:
                minA, maxA = _range(df["odds_A"]);
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
        badge_res   = "✅" if hit_res is True else ("❌" if hit_res is False else "⏳")
        badge_score = "✅" if hit_score is True else ("❌" if hit_score is False else "⏳")

        # sugestões + avaliação (com fallback amigável)
        aposta_txt = market_label(row.get('bet_suggestion'))
        gols_txt   = market_label(row.get('goal_bet_suggestion'))

        hit_bet  = eval_bet_row(row)
        hit_goal = eval_goal_row(row)
        badge_bet  = "✅" if hit_bet is True else ("❌" if hit_bet is False else "⏳")
        badge_goal = "✅" if hit_goal is True else ("❌" if hit_goal is False else "⏳")

        # previsões com fallback amigável
        result_txt = market_label(row.get('result_predicted'))
        score_txt  = fmt_score_pred_text(row.get('score_predicted'))

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{title}**")
                conf_txt = conf_badge(row)
                cap_line = f"{tournament_label(row.get('tournament_id'))} • Modelo {row.get('model','—')}"
                if conf_txt: cap_line += f" • {conf_txt}"
                st.caption(cap_line)

                # Layout em grid para Previsão e Sugestões
                st.markdown(
                    f'''
                    <div class="info-grid">
                        <div><span class="text-label">Prev.:</span> {green_html(result_txt)} {badge_res}</div>
                        <div><span class="text-label">Placar:</span> {green_html(score_txt)} {badge_score}</div>
                        <div><span class="text-label">💡 Sugestão:</span> {green_html(aposta_txt)} {badge_bet}</div>
                        <div><span class="text-label">⚽ Gols:</span> {green_html(gols_txt)} {badge_goal}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

            with c2:
                st.markdown(f'<span class="badge badge-wait">{status_txt}</span>', unsafe_allow_html=True)
                if norm_status_key(row.get("status","")) in FINISHED_TOKENS:
                    rh, ra = row.get("result_home"), row.get("result_away")
                    final_txt = f"{int(rh)}-{int(ra)}" if pd.notna(rh) and pd.notna(ra) else "—"
                    st.markdown(f"**Final:** {final_txt}")

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

    # Helper para sanitizar texto para PDF (codificação cp1252)
    def V(text):
        return str(text).encode("cp1252", "replace").decode("cp1252")

    pdf.set_font("helvetica", "B", 16)

    title = f"Relatório de Jogos - {datetime.now().strftime('%d/%m/%Y')}"
    pdf.cell(0, 10, V(title), border=0, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("helvetica", "B", 8)

    # Cabeçalho da tabela
    headers = ["Data", "Casa", "Visitante", "Prev.", "Placar", "Sugestão", "Gols"]
    col_widths = [22, 35, 35, 15, 20, 30, 30]
    for h, w in zip(headers, col_widths):
        pdf.cell(
            w, 8, V(h),
            border=1, align="C",
            new_x=XPos.RIGHT, new_y=YPos.TOP
        )
    # quebra de linha depois do último:
    pdf.ln()

    # Linhas
    pdf.set_font("helvetica", "", 8)
    for _, row in df.iterrows():
        data_txt = row["date"].strftime("%d/%m %H:%M") if pd.notna(row.get("date")) else ""
        pdf.cell(col_widths[0], 8, V(data_txt)[:16],
                 border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)

        pdf.cell(col_widths[1], 8, V(row.get("home", ""))[:18],
                 border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)

        pdf.cell(col_widths[2], 8, V(row.get("away", ""))[:18],
                 border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)

        pdf.cell(col_widths[3], 8, V(market_label(row.get("result_predicted")))[:10],
                 border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)

        pdf.cell(col_widths[4], 8, V(fmt_score_pred_text(row.get("score_predicted")))[:10],
                 border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)

        pdf.cell(col_widths[5], 8, V(market_label(row.get("bet_suggestion")))[:18],
                 border=1, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)

        pdf.cell(col_widths[6], 8, V(market_label(row.get("goal_bet_suggestion")))[:18],
                 border=1, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    # --- retorno compatível com fpdf/fpdf2 ---
    out = pdf.output()
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)                     # já é bytes/bytearray
    else:
        return out.encode("latin-1", "ignore")  # versões que retornam str

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
        flt = filtros_ui(df)
        tournaments_sel, models_sel, teams_sel = flt["tournaments_sel"], flt["models_sel"], flt["teams_sel"]
        bet_sel, goal_sel = flt["bet_sel"], flt["goal_sel"]
        selected_date_range, selH, selD, selA = flt["selected_date_range"], flt["selH"], flt["selD"], flt["selA"]
        q_team = flt["q_team"]

        use_list_view = True if MODO_MOBILE else st.sidebar.checkbox("Usar visualização em lista (mobile)", value=False)

        # Máscara combinada (sem status)
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
                                             "result_predicted","score_predicted","bet_suggestion","goal_bet_suggestion",
                                             "odds_H","odds_D","odds_A","result_home","result_away"] if c in df_ag.columns]
                            ]),
                            use_container_width=True, hide_index=True
                        )

            # --- ABA FINALIZADOS (com KPIs e gráfico) ---
            with tab_fin:
                if df_fin.empty:
                    st.info("Sem jogos finalizados neste recorte.")
                else:
                    if use_list_view:
                        display_list_view(df_fin)
                    else:
                        st.dataframe(
                            apply_friendly_for_display(df_fin[
                                [c for c in ["date","home","away","tournament_id","model","status",
                                             "result_predicted","score_predicted","bet_suggestion","goal_bet_suggestion",
                                             "odds_H","odds_D","odds_A","result_home","result_away"] if c in df_fin.columns]
                            ]),
                            use_container_width=True, hide_index=True
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
                            pred_code_s = normalize_pred_code(sub.get("result_predicted", pd.Series(index=sub.index, dtype="object")))
                            pred_correct_s = mv_s & (pred_code_s == real_s)
                            pred_wrong_s   = mv_s & pred_code_s.notna() & real_s.notna() & (pred_code_s != real_s)

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
                                goal_eval_s.loc[idx] = evaluate_market(goal_codes_s.loc[idx], rh_s.loc[idx], ra_s.loc[idx]) if mv_s.loc[idx] else None
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

                            # Métricas por modelo
                            acc_pred, c_pred, t_pred = compute_acc(pred_correct_s, pred_wrong_s)
                            acc_bet,  c_bet,  t_bet  = compute_acc(bet_correct_s,  bet_wrong_s)
                            acc_goal, c_goal, t_goal = compute_acc(goal_correct_s_no_btts, goal_wrong_s_no_btts)
                            acc_btts, c_btts, t_btts = compute_acc(btts_correct_s, btts_wrong_s)
                            acc_score, c_score, t_score = compute_acc(score_correct_s, score_wrong_s)

                            rows += [
                                {"Modelo": m, "Métrica": "Resultado",            "Acerto (%)": 0 if np.isnan(acc_pred) else round(acc_pred,1), "Acertos": c_pred,  "Total Avaliado": t_pred},
                                {"Modelo": m, "Métrica": "Sugestão de Aposta",   "Acerto (%)": 0 if np.isnan(acc_bet)  else round(acc_bet,1),  "Acertos": c_bet,   "Total Avaliado": t_bet},
                                {"Modelo": m, "Métrica": "Sugestão de Gols",     "Acerto (%)": 0 if np.isnan(acc_goal) else round(acc_goal,1), "Acertos": c_goal,  "Total Avaliado": t_goal},
                                {"Modelo": m, "Métrica": "Ambos Marcam",         "Acerto (%)": 0 if np.isnan(acc_btts) else round(acc_btts,1), "Acertos": c_btts,  "Total Avaliado": t_btts},
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
                        # Um único modelo/seleção: mantém comportamento agregado
                        pred_code = normalize_pred_code(df_fin.get("result_predicted", pd.Series(index=df_fin.index, dtype="object")))
                        pred_correct = mask_valid & (pred_code == real_code)
                        pred_wrong   = mask_valid & pred_code.notna() & real_code.notna() & (pred_code != real_code)

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

                        score_eval = pd.Series(index=df_fin.index, dtype="object")
                        if "score_predicted" in df_fin.columns:
                            for idx in df_fin.index:
                                if mask_valid.loc[idx]:
                                    ph, pa = parse_score_pred(df_fin.at[idx, "score_predicted"])
                                    if ph is None or pa is None:
                                        score_eval.loc[idx] = None
                                    else:
                                        try:
                                            score_eval.loc[idx] = (int(rh.loc[idx]) == int(ph)) and (int(ra.loc[idx]) == int(pa))
                                        except Exception:
                                            score_eval.loc[idx] = None
                                else:
                                    score_eval.loc[idx] = None
                        score_correct = score_eval == True
                        score_wrong   = score_eval == False

                        def compute_acc2(ok_mask: pd.Series, bad_mask: pd.Series):
                            total = int((ok_mask | bad_mask).sum())
                            correct = int(ok_mask.sum())
                            acc = (correct / total * 100.0) if total > 0 else np.nan
                            return acc, correct, total

                        acc_pred, c_pred, t_pred = compute_acc2(pred_correct, pred_wrong)
                        acc_bet,  c_bet,  t_bet  = compute_acc2(bet_correct,  bet_wrong)

                        # NEW: Separate BTTS from goal suggestions
                        btts_mask = goal_codes.astype(str).str.lower().isin(['btts_yes', 'btts_no'])
                        btts_correct = goal_correct & btts_mask
                        btts_wrong = goal_wrong & btts_mask
                        acc_btts, c_btts, t_btts = compute_acc2(btts_correct, btts_wrong)

                        # Exclude BTTS from the general "goal" metric
                        goal_correct_no_btts = goal_correct & ~btts_mask
                        goal_wrong_no_btts   = goal_wrong & ~btts_mask
                        acc_goal, c_goal, t_goal = compute_acc2(goal_correct_no_btts, goal_wrong_no_btts)

                        st.subheader("Percentual de acerto (apenas finalizados)")
                        k1, k2, k3, k4 = (st.container(), st.container(), st.container(), st.container()) if MODO_MOBILE else st.columns(4)
                        k1.metric("Resultado", f"{0 if np.isnan(acc_pred) else round(acc_pred,1)}%", f"{c_pred}/{t_pred}")
                        k2.metric("Sugestão de Aposta", f"{0 if np.isnan(acc_bet) else round(acc_bet,1)}%", f"{c_bet}/{t_bet}")
                        k3.metric("Sugestão de Gols", f"{0 if np.isnan(acc_goal) else round(acc_goal,1)}%", f"{c_goal}/{t_goal}")
                        k4.metric("Ambos Marcam", f"{0 if np.isnan(acc_btts) else round(acc_btts,1)}%", f"{c_btts}/{t_btts}")

                        metrics_df = pd.DataFrame({
                            "Métrica": ["Resultado", "Sugestão de Aposta", "Sugestão de Gols", "Ambos Marcam"],
                            "Acerto (%)": [
                                0 if np.isnan(acc_pred) else round(acc_pred, 1),
                                0 if np.isnan(acc_bet) else round(acc_bet, 1),
                                0 if np.isnan(acc_goal) else round(acc_goal, 1),
                                0 if np.isnan(acc_btts) else round(acc_btts, 1),
                            ],
                            "Acertos": [c_pred, c_bet, c_goal, c_btts],
                            "Total Avaliado": [t_pred, t_bet, t_goal, t_btts],
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
