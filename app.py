import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
from typing import Any, Tuple, Optional

# ============================
# Configuração da página
# ============================
st.set_page_config(
    layout="wide",
    page_title="Análise de Previsões de Futebol",
    initial_sidebar_state="collapsed",  # colapsa sidebar por padrão (melhor no celular)
)

# Toggle manual de modo mobile (controle explícito para layout responsivo)
col_m1, col_m2 = st.columns([1, 4])
with col_m1:
    MODO_MOBILE = st.toggle("📱 Mobile", value=True)
with col_m2:
    st.title("Análise de Previsões de Futebol")

# --- Estilos mobile-first ---
st.markdown("""
<style>
/* tipografia e espaçamento base */
html, body, .stApp { font-size: 16px; }
@media (max-width: 768px) {
  html, body, .stApp { font-size: 17px; } /* texto um pouco maior */
  section[data-testid="stSidebar"] { display:none !important; } /* esconde sidebar no mobile */
}

/* container base */
.block-container { padding-top: 0.5rem !important; }

/* cards de partidas */
.card {
  border: 1px solid #1f2937; border-radius: 14px; padding: 12px;
  background: #0b0b0b; box-shadow: 0 1px 8px rgba(0,0,0,.2);
}
.badge { padding: 2px 8px; border-radius: 999px; font-weight: 600; }
.badge-ok { background:#14532d; color:#d1fae5; }
.badge-bad { background:#7f1d1d; color:#fee2e2; }
.badge-wait { background:#334155; color:#e2e8f0; }

/* botões “tocáveis” */
button, .stButton>button {
  border-radius: 12px; padding: 10px 14px; font-weight: 600;
}

/* expanders com header maior */
div[data-testid="stExpander"] summary {
  padding: 10px 12px; font-size: 1.05rem; font-weight: 700;
}

/* dataframe: overflow horizontal com scroll */
.stDataFrame { overflow-x: auto; }
</style>
""", unsafe_allow_html=True)

# ============================
# Dicionários Amigáveis
# ============================
FRIENDLY_COLS = {
    "status": "Status",
    "tournament_id": "Torneio",
    "model": "Modelo",
    "date": "Data/Hora",
    "home": "Casa",
    "away": "Visitante",
    "result_predicted": "Resultado Previsto",
    "score_predicted": "Placar Previsto",
    "bet_suggestion": "Sugestão de Aposta",
    "goal_bet_suggestion": "Sugestão de Gols",
    "odds_H": "Odd Casa",
    "odds_D": "Odd Empate",
    "odds_A": "Odd Visitante",
    "prob_H": "Prob. Casa",
    "prob_D": "Prob. Empate",
    "prob_A": "Prob. Visitante",
    # Over/Under
    "over_0_5": "Mais de 0.5 gols",
    "over_1_5": "Mais de 1.5 gols",
    "over_2_5": "Mais de 2.5 gols",
    "over_3_5": "Mais de 3.5 gols",
    "under_0_5": "Menos de 0.5 gols",
    "under_1_5": "Menos de 1.5 gols",
    "under_2_5": "Menos de 2.5 gols",
    "under_3_5": "Menos de 3.5 gols",
    # BTTS
    "btts_yes": "Ambos Marcam Sim",
    "btts_no": "Ambos Marcam Não",
    "final_score": "Resultado Final",
}

FRIENDLY_MARKETS = {
    # 1x2
    "H": "Casa",
    "D": "Empate",
    "A": "Visitante",
    # Over/Under
    "over_0_5": "Mais de 0.5 gols",
    "over_1_5": "Mais de 1.5 gols",
    "over_2_5": "Mais de 2.5 gols",
    "over_3_5": "Mais de 3.5 gols",
    "under_0_5": "Menos de 0.5 gols",
    "under_1_5": "Menos de 1.5 gols",
    "under_2_5": "Menos de 2.5 gols",
    "under_3_5": "Menos de 3.5 gols",
    # BTTS
    "btts_yes": "Ambos Marcam Sim",
    "btts_no": "Ambos Marcam Não",
}

# Amigáveis de torneio (ajuste conforme sua base)
FRIENDLY_TOURNAMENTS = {
    325: "Brasileirão Série A", "325": "Brasileirão Série A",
    390: "Brasileirão Série B", "390": "Brasileirão Série B",
    17:  "Premier League (Inglês)", "17": "Premier League (Inglês)",
    8:   "La Liga (Espanhol)", "8": "La Liga (Espanhol)",
    23:  "Italiano (Séria A)", "23": "Italiano (Séria A)",
    35:  "Bundesliga (Alemão)", "35": "Bundesliga (Alemão)",
}

# ============================
# Status (apenas finished)
# ============================
FINISHED_TOKENS = {"finished"}  # somente 'finished' conta como finalizado

FRIENDLY_STATUS_MAP = {
    "finished": "Finalizado",
    "nostarted": "Agendado",
    "not_started": "Agendado",
    "notstarted": "Agendado",
}

# Normalizador p/ resultado previsto
PRED_NORMALIZER = {
    "H": "H", "D": "D", "A": "A",
    "CASA": "H", "EMPATE": "D", "VISITANTE": "A",
    "HOME": "H", "DRAW": "D", "AWAY": "A",
}

# ============================
# Helpers
# ============================
def market_label(v: Any) -> Any:
    return FRIENDLY_MARKETS.get(v, v)

def _canon_tourn_key(x: Any):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, float):
        return int(x) if float(x).is_integer() else x
    try:
        s = str(x).strip()
        return int(s)
    except Exception:
        return str(x).strip()

def tournament_label(x: Any) -> str:
    k = _canon_tourn_key(x)
    if k in FRIENDLY_TOURNAMENTS:
        return FRIENDLY_TOURNAMENTS[k]
    ks = str(k) if k is not None else None
    if ks in FRIENDLY_TOURNAMENTS:
        return FRIENDLY_TOURNAMENTS[ks]
    return f"Torneio {x}"

def _norm_status_key(s: Any) -> str:
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")

def status_label(s: Any) -> str:
    return FRIENDLY_STATUS_MAP.get(_norm_status_key(s), str(s))

def normalize_pred_code(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="object")
    s = series.astype(str).str.strip().str.upper()
    return s.map(lambda x: PRED_NORMALIZER.get(x, np.nan))

def _parse_threshold(token: str) -> Optional[float]:
    if token is None:
        return None
    t = str(token).replace("_", ".").strip()
    try:
        return float(t)
    except Exception:
        return None

def evaluate_market(code: Any, rh: Any, ra: Any) -> Optional[bool]:
    """True se o 'code' bate com placar final; False se erra; None se não avaliável."""
    if pd.isna(code) or pd.isna(rh) or pd.isna(ra):
        return None
    s = str(code).strip().lower()

    # 1x2
    if s in ("h", "casa", "home"):
        return rh > ra
    if s in ("d", "empate", "draw"):
        return rh == ra
    if s in ("a", "visitante", "away"):
        return rh < ra

    # Over/Under
    if s.startswith("over_"):
        th = _parse_threshold(s.split("over_", 1)[1])
        return None if th is None else (float(rh) + float(ra)) > th
    if s.startswith("under_"):
        th = _parse_threshold(s.split("under_", 1)[1])
        return None if th is None else (float(rh) + float(ra)) < th

    # BTTS
    if s == "btts_yes":
        return (float(rh) > 0) and (float(ra) > 0)
    if s == "btts_no":
        return (float(rh) == 0) or (float(ra) == 0)

    return None

# Odds: 2 casas decimais
def fmt_odd(x):
    try:
        v = float(x)
        if pd.isna(v):
            return "N/A"
        return f"{v:.2f}"
    except Exception:
        return "N/A"

def fmt_prob(x):
    try:
        v = float(x)
        if pd.isna(v):
            return "N/A"
        v = v*100
        return f"{v:.2f}%"
    except Exception:
        return "N/A"

# ------ Placar Previsto: parser e avaliação ------
def parse_score_pred(x: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    Extrai (home, away) do campo score_predicted.
    Aceita: '2-1', '2x1', lista [2,1], dict {'home':2,'away':1} etc.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return (None, None)

    # dict-like
    if isinstance(x, dict):
        for hk, ak in (("home", "away"), ("h", "a")):
            if hk in x and ak in x:
                try:
                    return int(x[hk]), int(x[ak])
                except Exception:
                    return (None, None)

    # list/tuple
    if isinstance(x, (list, tuple)) and len(x) == 2:
        try:
            return int(x[0]), int(x[1])
        except Exception:
            return (None, None)

    # string: regex "n algo n"
    s = str(x)
    m = re.search(r"(\d+)\D+(\d+)", s)
    if m:
        try:
            return int(m.group(1)), int(m.group(2))
        except Exception:
            return (None, None)

    return (None, None)

def eval_result_pred_row(row) -> Optional[bool]:
    """Acerto/erro do Resultado Previsto (H/D/A) para usar na lista (cards)."""
    status_val = _norm_status_key(row.get("status", ""))
    if status_val not in FINISHED_TOKENS:
        return None
    rh, ra = row.get("result_home"), row.get("result_away")
    if pd.isna(rh) or pd.isna(ra):
        return None
    real = "H" if rh > ra else ("D" if rh == ra else "A")
    pred_raw = row.get("result_predicted")
    pred_str = str(pred_raw).strip().upper()
    pred = PRED_NORMALIZER.get(pred_str, np.nan)
    if pd.isna(pred):
        return None
    return pred == real

def eval_score_pred_row(row) -> Optional[bool]:
    """Acerto/erro do Placar Previsto para usar na lista (cards)."""
    status_val = _norm_status_key(row.get("status", ""))
    if status_val not in FINISHED_TOKENS:
        return None
    rh, ra = row.get("result_home"), row.get("result_away")
    if pd.isna(rh) or pd.isna(ra):
        return None
    ph, pa = parse_score_pred(row.get("score_predicted"))
    if ph is None or pa is None:
        return None
    try:
        return (int(rh) == int(ph)) and (int(ra) == int(pa))
    except Exception:
        return None

# ===== helpers para exibir probabilidade + odd e checagem de colunas =====
def _po(row, prob_key: str, odd_key: str) -> str:
    """Formata 'Prob - Odd' com segurança."""
    return f"{fmt_prob(row.get(prob_key))} - Odd: {fmt_odd(row.get(odd_key))}"

def _exists(df: pd.DataFrame, *cols) -> bool:
    return all(c in df.columns for c in cols)

# ============================
# Carregamento e normalização
# ============================
@st.cache_data
def load_data():
    file_path = "PrevisaoJogos.xlsx"
    df = pd.read_excel(file_path)

    # Tipos
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["odds_H", "odds_D", "odds_A"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Converte placares para numérico
    for col in ["result_home", "result_away"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normaliza sugestões que possam vir como dict {"market": "..."}
    def _only_market(x):
        if isinstance(x, dict):
            return x.get("market")
        return x

    for col in ["bet_suggestion", "goal_bet_suggestion"]:
        if col in df.columns:
            df[col] = df[col].apply(_only_market)

    return df

# ============================
# Exibição amigável (sem afetar filtros)
# ============================
def apply_friendly_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Tradução de mercados
    for col in ["bet_suggestion", "goal_bet_suggestion", "result_predicted"]:
        if col in out.columns:
            out[col] = out[col].map(FRIENDLY_MARKETS).fillna(out[col])

    # Resultado Final
    def _fmt_score(row):
        status_raw = _norm_status_key(row.get("status", ""))
        if status_raw in FINISHED_TOKENS:
            rh = row.get("result_home"); ra = row.get("result_away")
            if pd.notna(rh) and pd.notna(ra):
                try:
                    return f"{int(rh)}-{int(ra)}"
                except Exception:
                    return f"{rh}-{ra}"
            return "N/A"
        return ""
    if {"status", "result_home", "result_away"}.issubset(out.columns):
        out["final_score"] = out.apply(_fmt_score, axis=1)

    # Status amigável
    if "status" in out.columns:
        out["status"] = out["status"].apply(status_label)

    # Torneio amigável
    if "tournament_id" in out.columns:
        out["tournament_id"] = out["tournament_id"].apply(tournament_label)

    # Renomeia headers
    out = out.rename(columns=FRIENDLY_COLS)
    return out

# ============================
# UI de Filtros (sidebar no desktop / gaveta no mobile)
# ============================
def filtros_ui(df: pd.DataFrame) -> dict:
    # helpers de options
    status_opts = sorted(df["status"].dropna().unique()) if "status" in df.columns else []
    tourn_opts  = sorted(df["tournament_id"].dropna().unique().tolist()) if "tournament_id" in df.columns else []
    model_opts  = sorted(df["model"].dropna().unique()) if "model" in df.columns else []

    if {"home", "away"}.issubset(df.columns):
        team_opts = pd.concat([df["home"], df["away"]], ignore_index=True).dropna()
        team_opts = sorted(team_opts.astype(str).unique())
    else:
        team_opts = []

    bet_opts   = sorted(df["bet_suggestion"].dropna().unique()) if "bet_suggestion" in df.columns else []
    goal_opts  = sorted(df["goal_bet_suggestion"].dropna().unique()) if "goal_bet_suggestion" in df.columns else []

    # datas
    if "date" in df.columns and df["date"].notna().any():
        min_date = df["date"].dropna().min().date()
        max_date = df["date"].dropna().max().date()
    else:
        min_date = max_date = None

    target = st.sidebar if not MODO_MOBILE else st
    container = target.expander("🔎 Filtros", expanded=not MODO_MOBILE)

    with container:
        # linha de filtros rápidos (chips)
        c1, c2 = st.columns(2) if MODO_MOBILE else st.columns(4)
        with c1:
            status_sel = st.multiselect(FRIENDLY_COLS["status"], status_opts, default=status_opts, format_func=status_label)
        with c2:
            models_sel = st.multiselect(FRIENDLY_COLS["model"], model_opts, default=model_opts)

        c3, c4 = st.columns(2)
        with c3:
            tournaments_sel = st.multiselect(FRIENDLY_COLS["tournament_id"], tourn_opts, default=tourn_opts, format_func=tournament_label)
        with c4:
            teams_sel = st.multiselect("Equipe (Casa ou Visitante)", team_opts, default=[] if MODO_MOBILE else team_opts)

        c5, c6 = st.columns(2)
        with c5:
            bet_sel = st.multiselect(
                FRIENDLY_COLS["bet_suggestion"],
                bet_opts,
                default=[],
                format_func=market_label,  # ⬅️ nomes amigáveis
            )
        with c6:
            goal_sel = st.multiselect(
                FRIENDLY_COLS["goal_bet_suggestion"],
                goal_opts,
                default=[],
                format_func=market_label,  # ⬅️ nomes amigáveis
            )

        # período e odds num segundo nível
        with st.expander("Ajustes finos (período e odds)", expanded=False):
            if min_date:
                selected_date_range = st.date_input("Período", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            else:
                selected_date_range = ()

            def _range(series: pd.Series, default=(0.0, 1.0)):
                s = series.dropna()
                return (float(s.min()), float(s.max())) if not s.empty else default

            selH = selD = selA = (0.0, 1.0)
            if "odds_H" in df.columns:
                minH, maxH = _range(df["odds_H"]); selH = st.slider(FRIENDLY_COLS["odds_H"], minH, maxH, (minH, maxH))
            if "odds_D" in df.columns:
                minD, maxD = _range(df["odds_D"]); selD = st.slider(FRIENDLY_COLS["odds_D"], minD, maxD, (minD, maxD))
            if "odds_A" in df.columns:
                minA, maxA = _range(df["odds_A"]); selA = st.slider(FRIENDLY_COLS["odds_A"], minA, maxA, (minA, maxA))

    return dict(
        status_sel=status_sel, tournaments_sel=tournaments_sel, models_sel=models_sel, teams_sel=teams_sel,
        bet_sel=bet_sel, goal_sel=goal_sel, selected_date_range=selected_date_range, selH=selH, selD=selD, selA=selA
    )

# ============================
# Cards (lista mobile-first) — com probabilidades de gols + odds
# ============================
def display_list_view(df: pd.DataFrame):
    for _, row in df.iterrows():
        dt_txt = row["date"].strftime("%d/%m %H:%M") if ("date" in df.columns and pd.notna(row["date"])) else "N/A"
        title = f"{row.get('home','?')} vs {row.get('away','?')}"
        status_txt = status_label(row.get("status","N/A"))

        # badges
        hit_res = eval_result_pred_row(row)
        hit_score = eval_score_pred_row(row)
        badge_res   = "✅" if hit_res is True else ("❌" if hit_res is False else "⏳")
        badge_score = "✅" if hit_score is True else ("❌" if hit_score is False else "⏳")

        # Sugestões principais
        aposta_txt = market_label(row.get('bet_suggestion', '—'))
        gols_txt = market_label(row.get('goal_bet_suggestion', '—'))

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2 = st.columns([3, 1])
            with c1:
                # título e cabeçalho
                st.markdown(f"**{title}**")
                st.caption(f"{dt_txt} • {tournament_label(row.get('tournament_id'))} • {row.get('model','—')}")

                # Previsão principal
                st.markdown(
                    f"**Prev.:** {market_label(row.get('result_predicted'))} {badge_res}  "
                    f"•  **Placar:** {row.get('score_predicted','—')} {badge_score}"
                )

                # 💡 Sugestão de Aposta e ⚽ Sugestão de Gols com destaque
                st.markdown(
                    f"**💡 {FRIENDLY_COLS['bet_suggestion']}:** {aposta_txt}\n\n"
                    f"**⚽ {FRIENDLY_COLS['goal_bet_suggestion']}:** {gols_txt}"
                )

            with c2:
                # status e resultado final
                st.markdown(f'<span class="badge badge-wait">{status_txt}</span>', unsafe_allow_html=True)
                if _norm_status_key(row.get("status","")) in FINISHED_TOKENS:
                    rh, ra = row.get("result_home"), row.get("result_away")
                    final_txt = f"{int(rh)}-{int(ra)}" if pd.notna(rh) and pd.notna(ra) else "—"
                    st.markdown(f"**Final:** {final_txt}")

            # expander com detalhes completos
            with st.expander("Detalhes, Probabilidades & Odds"):
                # Sugestões e 1x2
                st.markdown(
                    f"- **Sugestão:** {aposta_txt}\n"
                    f"- **Sugestão de Gols:** {gols_txt}\n"
                    f"- **Odds 1x2:** {fmt_odd(row.get('odds_H'))} / {fmt_odd(row.get('odds_D'))} / {fmt_odd(row.get('odds_A'))}\n"
                    f"- **Prob. (H/D/A):** {fmt_prob(row.get('prob_H'))} / {fmt_prob(row.get('prob_D'))} / {fmt_prob(row.get('prob_A'))}"
                )

                # Probabilidades de Gols — Over/Under + BTTS (com odds)
                st.markdown("---")
                st.markdown("**Over/Under (Prob. — Odd)**")

                # UNDER
                under_lines = []
                if _exists(df, "prob_under_0_5"): under_lines.append(f"- **Under 0.5:** {_po(row, 'prob_under_0_5', 'odds_match_goals_0.5_under')}")
                if _exists(df, "prob_under_1_5"): under_lines.append(f"- **Under 1.5:** {_po(row, 'prob_under_1_5', 'odds_match_goals_1.5_under')}")
                if _exists(df, "prob_under_2_5"): under_lines.append(f"- **Under 2.5:** {_po(row, 'prob_under_2_5', 'odds_match_goals_2.5_under')}")
                if _exists(df, "prob_under_3_5"): under_lines.append(f"- **Under 3.5:** {_po(row, 'prob_under_3_5', 'odds_match_goals_3.5_under')}")
                if under_lines:
                    st.markdown("\n".join(under_lines))

                # OVER
                over_lines = []
                if _exists(df, "prob_over_0_5"): over_lines.append(f"- **Over 0.5:** {_po(row, 'prob_over_0_5', 'odds_match_goals_0.5_over')}")
                if _exists(df, "prob_over_1_5"): over_lines.append(f"- **Over 1.5:** {_po(row, 'prob_over_1_5', 'odds_match_goals_1.5_over')}")
                if _exists(df, "prob_over_2_5"): over_lines.append(f"- **Over 2.5:** {_po(row, 'prob_over_2_5', 'odds_match_goals_2.5_over')}")
                if _exists(df, "prob_over_3_5"): over_lines.append(f"- **Over 3.5:** {_po(row, 'prob_over_3_5', 'odds_match_goals_3.5_over')}")
                if over_lines:
                    st.markdown("\n".join(over_lines))

                # BTTS
                if _exists(df, "prob_btts_yes") or _exists(df, "prob_btts_no"):
                    st.markdown("---")
                    st.markdown("**BTTS (Prob. — Odd)**")
                    if _exists(df, "prob_btts_yes"):
                        st.markdown(f"- **Ambos marcam — Sim:** {_po(row, 'prob_btts_yes', 'odds_btts_yes')}")
                    if _exists(df, "prob_btts_no"):
                        st.markdown(f"- **Ambos marcam — Não:** {_po(row, 'prob_btts_no', 'odds_btts_no')}")

            st.markdown('</div>', unsafe_allow_html=True)
            st.write("")  # espaçamento



# ============================
# App principal
# ============================
try:
    df = load_data()
    if df.empty:
        st.error("O arquivo `PrevisaoJogos.xlsx` está vazio ou não pôde ser lido.")
    else:
        # -------- Filtros (mobile: gaveta no topo / desktop: sidebar) --------
        flt = filtros_ui(df)
        status_sel, tournaments_sel, models_sel = flt["status_sel"], flt["tournaments_sel"], flt["models_sel"]
        teams_sel, bet_sel, goal_sel = flt["teams_sel"], flt["bet_sel"], flt["goal_sel"]
        selected_date_range, selH, selD, selA = flt["selected_date_range"], flt["selH"], flt["selD"], flt["selA"]

        # Visualização: lista é padrão no mobile
        use_list_view = True if MODO_MOBILE else st.sidebar.checkbox("Usar visualização em lista (mobile)", value=False)

        # -------- Filtros combinados --------
        final_mask = pd.Series(True, index=df.index)
        if status_sel and "status" in df.columns:
            final_mask &= df["status"].isin(status_sel)
        if tournaments_sel and "tournament_id" in df.columns:
            final_mask &= df["tournament_id"].isin(tournaments_sel)
        if models_sel and "model" in df.columns:
            final_mask &= df["model"].isin(models_sel)
        if teams_sel and {"home", "away"}.issubset(df.columns):
            home_str = df["home"].astype(str); away_str = df["away"].astype(str)
            final_mask &= (home_str.isin(teams_sel) | away_str.isin(teams_sel))
        if bet_sel and "bet_suggestion" in df.columns:
            final_mask &= df["bet_suggestion"].isin(bet_sel)
        if goal_sel and "goal_bet_suggestion" in df.columns:
            final_mask &= df["goal_bet_suggestion"].isin(goal_sel)
        if selected_date_range and len(selected_date_range) == 2 and "date" in df.columns:
            start_date, end_date = selected_date_range
            final_mask &= (df["date"].dt.date.between(start_date, end_date)) | (df["date"].isna())
        if "odds_H" in df.columns:
            final_mask &= ((df["odds_H"] >= selH[0]) & (df["odds_H"] <= selH[1])) | (df["odds_H"].isna())
        if "odds_D" in df.columns:
            final_mask &= ((df["odds_D"] >= selD[0]) & (df["odds_D"] <= selD[1])) | (df["odds_D"].isna())
        if "odds_A" in df.columns:
            final_mask &= ((df["odds_A"] >= selA[0]) & (df["odds_A"] <= selA[1])) | (df["odds_A"].isna())

        df_filtered = df[final_mask]

        # ============================
        # Exibição
        # ============================
        st.header("Predições Filtradas")
        if df_filtered.empty:
            st.warning("Nenhum dado corresponde aos filtros atuais.")
        else:
            # ---------- Lógica de acerto/erro (para KPIs e destaques) ----------
            status_norm = df_filtered["status"].astype(str).map(_norm_status_key) if "status" in df_filtered.columns else pd.Series("", index=df_filtered.index)
            is_finished = status_norm.isin(FINISHED_TOKENS) if "status" in df_filtered.columns else pd.Series(False, index=df_filtered.index)

            rh = df_filtered.get("result_home", pd.Series(index=df_filtered.index, dtype="float"))
            ra = df_filtered.get("result_away", pd.Series(index=df_filtered.index, dtype="float"))

            mask_valid = is_finished & rh.notna() & ra.notna()

            # Real (H/D/A)
            real_code = pd.Series(index=df_filtered.index, dtype="object")
            real_code.loc[mask_valid & (rh > ra)] = "H"
            real_code.loc[mask_valid & (rh == ra)] = "D"
            real_code.loc[mask_valid & (rh < ra)] = "A"

            # Previsto (H/D/A) normalizado
            pred_code = normalize_pred_code(df_filtered.get("result_predicted", pd.Series(index=df_filtered.index, dtype="object")))
            pred_correct = mask_valid & (pred_code == real_code)
            pred_wrong   = mask_valid & pred_code.notna() & real_code.notna() & (pred_code != real_code)

            # Sugestão de Aposta
            bet_codes = df_filtered.get("bet_suggestion", pd.Series(index=df_filtered.index, dtype="object"))
            bet_eval = pd.Series(index=df_filtered.index, dtype="object")
            for idx in df_filtered.index:
                bet_eval.loc[idx] = evaluate_market(bet_codes.loc[idx], rh.loc[idx], ra.loc[idx]) if mask_valid.loc[idx] else None
            bet_correct = bet_eval == True
            bet_wrong   = bet_eval == False

            # Sugestão de Gols
            goal_codes = df_filtered.get("goal_bet_suggestion", pd.Series(index=df_filtered.index, dtype="object"))
            goal_eval = pd.Series(index=df_filtered.index, dtype="object")
            for idx in df_filtered.index:
                goal_eval.loc[idx] = evaluate_market(goal_codes.loc[idx], rh.loc[idx], ra.loc[idx]) if mask_valid.loc[idx] else None
            goal_correct = goal_eval == True
            goal_wrong   = goal_eval == False

            # Placar Previsto (exato)
            score_eval = pd.Series(index=df_filtered.index, dtype="object")
            if "score_predicted" in df_filtered.columns:
                for idx in df_filtered.index:
                    if mask_valid.loc[idx]:
                        ph, pa = parse_score_pred(df_filtered.at[idx, "score_predicted"])
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

            # ---------- LISTA (cards) ou TABELA ----------
            if use_list_view:
                display_list_view(df_filtered)
            else:
                base_cols = [
                    "date", "home", "away", "tournament_id", "model",
                    "status",
                    "result_predicted", "score_predicted",
                    "result_home", "result_away",  # gerar Resultado Final
                    "bet_suggestion", "goal_bet_suggestion",
                    "odds_H", "odds_D", "odds_A"
                ]
                base_cols = [c for c in base_cols if c in df_filtered.columns]
                display_df_full = apply_friendly_for_display(df_filtered[base_cols])

                # remove placar bruto
                for raw_col in ["result_home", "result_away"]:
                    if raw_col in display_df_full.columns:
                        display_df_full = display_df_full.drop(columns=[raw_col])

                desired_order = [
                    "Data/Hora", "Casa", "Visitante", "Torneio", "Modelo",
                    "Status",
                    "Resultado Final",
                    "Resultado Previsto",
                    "Sugestão de Aposta", "Sugestão de Gols",
                    "Placar Previsto",
                    "Odd Casa", "Odd Empate", "Odd Visitante"
                ]
                display_df = display_df_full[[c for c in desired_order if c in display_df_full.columns]]

                # Dataframe simples e responsivo (sem Styler pesado)
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                )

            # ---------- KPIs e gráfico ----------
            def compute_acc(ok_mask: pd.Series, bad_mask: pd.Series):
                total = int((ok_mask | bad_mask).sum())
                correct = int(ok_mask.sum())
                acc = (correct / total * 100.0) if total > 0 else np.nan
                return acc, correct, total

            acc_pred, c_pred, t_pred = compute_acc(pred_correct, pred_wrong)
            acc_bet,  c_bet,  t_bet  = compute_acc(bet_correct,  bet_wrong)
            acc_goal, c_goal, t_goal = compute_acc(goal_correct, goal_wrong)

            st.subheader("Percentual de acerto (jogos finalizados)")
            if MODO_MOBILE:
                k1 = st.container()
                k2 = st.container()
                k3 = st.container()
            else:
                k1, k2, k3 = st.columns(3)

            k1.metric("Resultado", f"{0 if np.isnan(acc_pred) else round(acc_pred,1)}%", f"{c_pred}/{t_pred}")
            k2.metric("Sugestão de Aposta", f"{0 if np.isnan(acc_bet) else round(acc_bet,1)}%", f"{c_bet}/{t_bet}")
            k3.metric("Sugestão de Gols", f"{0 if np.isnan(acc_goal) else round(acc_goal,1)}%", f"{c_goal}/{t_goal}")

            metrics_df = pd.DataFrame({
                "Métrica": ["Resultado", "Sugestão de Aposta", "Sugestão de Gols"],
                "Acerto (%)": [
                    0 if np.isnan(acc_pred) else round(acc_pred, 1),
                    0 if np.isnan(acc_bet) else round(acc_bet, 1),
                    0 if np.isnan(acc_goal) else round(acc_goal, 1),
                ],
                "Acertos": [c_pred, c_bet, c_goal],
                "Total Avaliado": [t_pred, t_bet, t_goal],
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

except FileNotFoundError:
    st.error("FATAL: `PrevisaoJogos.xlsx` não encontrado.")
except Exception as e:
    st.error(f"Erro inesperado: {e}")
