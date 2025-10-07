import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
import urllib.parse
from datetime import date
from typing import Any, Tuple, Optional

# ============================
# Persistência via Query Params
# ============================
def _csv_encode(items):
    if not items:
        return ""
    return ",".join([str(x) for x in items])

def _csv_decode(s):
    if not s:
        return []
    return [x for x in s.split(",") if x != ""]

def _float_tuple_decode(s, default=(0.0, 1.0)):
    try:
        a, b = s.split(",")
        return (float(a), float(b))
    except Exception:
        return default

def _date_range_decode(ds, de):
    try:
        if not ds or not de:
            return ()
        y1, m1, d1 = [int(x) for x in ds.split("-")]
        y2, m2, d2 = [int(x) for x in de.split("-")]
        return (date(y1, m1, d1), date(y2, m2, d2))
    except Exception:
        return ()

def _get_query_params():
    # Compat: versões novas (st.query_params) e antigas (experimental_get_query_params)
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()

def _set_query_params(params: dict):
    try:
        st.query_params.clear()
        st.query_params.update(params)
    except Exception:
        st.experimental_set_query_params(**params)

def read_filters_from_url():
    qs = _get_query_params()
    return dict(
        status_sel=_csv_decode(qs.get("st", [""])[0]),
        tournaments_sel=_csv_decode(qs.get("tn", [""])[0]),
        models_sel=_csv_decode(qs.get("md", [""])[0]),
        teams_sel=_csv_decode(qs.get("tm", [""])[0]),
        bet_sel=_csv_decode(qs.get("bs", [""])[0]),
        goal_sel=_csv_decode(qs.get("gs", [""])[0]),
        selected_date_range=_date_range_decode(qs.get("ds", [""])[0], qs.get("de", [""])[0]),
        selH=_float_tuple_decode(qs.get("oh", [""])[0]),
        selD=_float_tuple_decode(qs.get("od", [""])[0]),
        selA=_float_tuple_decode(qs.get("oa", [""])[0]),
        use_list_view=(qs.get("lv", [""])[0] == "1"),
        mobile=(qs.get("mb", [""])[0] == "1"),
    )

def write_filters_to_url(flt: dict, MODO_MOBILE: bool):
    params = {
        "st": _csv_encode(flt["status_sel"]),
        "tn": _csv_encode(flt["tournaments_sel"]),
        "md": _csv_encode(flt["models_sel"]),
        "tm": _csv_encode(flt["teams_sel"]),
        "bs": _csv_encode(flt["bet_sel"]),
        "gs": _csv_encode(flt["goal_sel"]),
        "ds": flt["selected_date_range"][0].isoformat() if flt["selected_date_range"] else "",
        "de": flt["selected_date_range"][1].isoformat() if flt["selected_date_range"] else "",
        "oh": f"{flt['selH'][0]},{flt['selH'][1]}",
        "od": f"{flt['selD'][0]},{flt['selD'][1]}",
        "oa": f"{flt['selA'][0]},{flt['selA'][1]}",
        "lv": "1" if st.session_state.get("use_list_view", False) else "0",
        "mb": "1" if MODO_MOBILE else "0",
    }
    params = {k: v for k, v in params.items() if v not in ("", ",")}
    _set_query_params(params)

# ============================
# Configuração da página
# ============================
st.set_page_config(
    layout="wide",
    page_title="Análise de Previsões de Futebol",
    initial_sidebar_state="collapsed",
)

# Lê prefs da URL ANTES do toggle
prefs = read_filters_from_url()

# Toggle manual de modo mobile (padrão vindo da URL, senão True)
col_m1, col_m2 = st.columns([1, 4])
with col_m1:
    MODO_MOBILE = st.toggle("📱 Mobile", value=prefs.get("mobile", True))
with col_m2:
    st.title("Análise de Previsões de Futebol")

# --- Estilos mobile-first ---
st.markdown("""
<style>
/* tipografia e espaçamento base */
html, body, .stApp { font-size: 16px; }
@media (max-width: 768px) {
  html, body, .stApp { font-size: 17px; }
  section[data-testid="stSidebar"] { display:none !important; }
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

FRIENDLY_TOURNAMENTS = {
    325: "Brasileirão Série A", "325": "Brasileirão Série A",
    390: "Brasileirão Série B", "390": "Brasileirão Série B",
    17:  "Premier League (Inglês)", "17": "Premier League (Inglês)",
    8:   "La Liga (Espanhol)", "8": "La Liga (Espanhol)",
    23:  "Italiano (Séria A)", "23": "Italiano (Séria A)",
    35:  "Bundesliga (Alemão)", "35": "Bundesliga (Alemão)",
}

# ============================
# Status e normalizadores
# ============================
FINISHED_TOKENS = {"finished"}

FRIENDLY_STATUS_MAP = {
    "finished": "Finalizado",
    "nostarted": "Agendado",
    "not_started": "Agendado",
    "notstarted": "Agendado",
}

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
    if s in ("h", "casa", "home"):
        return rh > ra
    if s in ("d", "empate", "draw"):
        return rh == ra
    if s in ("a", "visitante", "away"):
        return rh < ra
    if s.startswith("over_"):
        th = _parse_threshold(s.split("over_", 1)[1])
        return None if th is None else (float(rh) + float(ra)) > th
    if s.startswith("under_"):
        th = _parse_threshold(s.split("under_", 1)[1])
        return None if th is None else (float(rh) + float(ra)) < th
    if s == "btts_yes":
        return (float(rh) > 0) and (float(ra) > 0)
    if s == "btts_no":
        return (float(rh) == 0) or (float(ra) == 0)
    return None

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
        v = v * 100
        return f"{v:.2f}%"
    except Exception:
        return "N/A"

def parse_score_pred(x: Any) -> Tuple[Optional[int], Optional[int]]:
    """Extrai (home, away) de score_predicted em formatos diversos."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return (None, None)
    if isinstance(x, dict):
        for hk, ak in (("home", "away"), ("h", "a")):
            if hk in x and ak in x:
                try:
                    return int(x[hk]), int(x[ak])
                except Exception:
                    return (None, None)
    if isinstance(x, (list, tuple)) and len(x) == 2:
        try:
            return int(x[0]), int(x[1])
        except Exception:
            return (None, None)
    s = str(x)
    m = re.search(r"(\d+)\D+(\d+)", s)
    if m:
        try:
            return int(m.group(1)), int(m.group(2))
        except Exception:
            return (None, None)
    return (None, None)

def eval_result_pred_row(row) -> Optional[bool]:
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

# Helpers de exibição de prob+odd e checagem de colunas
def _po(row, prob_key: str, odd_key: str) -> str:
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

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["odds_H", "odds_D", "odds_A"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["result_home", "result_away"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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
    for col in ["bet_suggestion", "goal_bet_suggestion", "result_predicted"]:
        if col in out.columns:
            out[col] = out[col].map(FRIENDLY_MARKETS).fillna(out[col])

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

    if "status" in out.columns:
        out["status"] = out["status"].apply(status_label)
    if "tournament_id" in out.columns:
        out["tournament_id"] = out["tournament_id"].apply(tournament_label)

    out = out.rename(columns=FRIENDLY_COLS)
    return out

# ============================
# UI de Filtros (sidebar no desktop / gaveta no mobile)
# ============================
def filtros_ui(df: pd.DataFrame, prefs: dict) -> dict:
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

    if "date" in df.columns and df["date"].notna().any():
        min_date = df["date"].dropna().min().date()
        max_date = df["date"].dropna().max().date()
    else:
        min_date = max_date = None

    target = st.sidebar if not MODO_MOBILE else st
    container = target.expander("🔎 Filtros", expanded=not MODO_MOBILE)

    with container:
        c1, c2 = st.columns(2) if MODO_MOBILE else st.columns(4)
        with c1:
            status_sel = st.multiselect(
                FRIENDLY_COLS["status"], status_opts,
                default=prefs.get("status_sel") or status_opts,
                format_func=status_label, key="status_sel"
            )
        with c2:
            models_sel = st.multiselect(
                FRIENDLY_COLS["model"], model_opts,
                default=prefs.get("models_sel") or model_opts,
                key="models_sel"
            )

        c3, c4 = st.columns(2)
        with c3:
            tournaments_sel = st.multiselect(
                FRIENDLY_COLS["tournament_id"], tourn_opts,
                default=prefs.get("tournaments_sel") or tourn_opts,
                format_func=tournament_label, key="tournaments_sel"
            )
        with c4:
            teams_sel = st.multiselect(
                "Equipe (Casa ou Visitante)", team_opts,
                default=prefs.get("teams_sel") or ([] if MODO_MOBILE else team_opts),
                key="teams_sel"
            )

        c5, c6 = st.columns(2)
        with c5:
            bet_sel  = st.multiselect(
                FRIENDLY_COLS["bet_suggestion"], bet_opts,
                default=prefs.get("bet_sel") or [],
                format_func=market_label, key="bet_sel"
            )
        with c6:
            goal_sel = st.multiselect(
                FRIENDLY_COLS["goal_bet_suggestion"], goal_opts,
                default=prefs.get("goal_sel") or [],
                format_func=market_label, key="goal_sel"
            )

        with st.expander("Ajustes finos (período e odds)", expanded=False):
            if min_date:
                default_periodo = prefs.get("selected_date_range") or (min_date, max_date)
                selected_date_range = st.date_input(
                    "Período",
                    value=default_periodo,
                    min_value=min_date, max_value=max_date, key="selected_date_range"
                )
            else:
                selected_date_range = ()

            def _range(series: pd.Series, default=(0.0, 1.0)):
                s = series.dropna()
                return (float(s.min()), float(s.max())) if not s.empty else default

            selH = prefs.get("selH") or (0.0, 1.0)
            selD = prefs.get("selD") or (0.0, 1.0)
            selA = prefs.get("selA") or (0.0, 1.0)
            if "odds_H" in df.columns:
                minH, maxH = _range(df["odds_H"])
                selH = st.slider(FRIENDLY_COLS["odds_H"], minH, maxH, (max(minH, selH[0]), min(maxH, selH[1])), key="selH")
            if "odds_D" in df.columns:
                minD, maxD = _range(df["odds_D"])
                selD = st.slider(FRIENDLY_COLS["odds_D"], minD, maxD, (max(minD, selD[0]), min(maxD, selD[1])), key="selD")
            if "odds_A" in df.columns:
                minA, maxA = _range(df["odds_A"])
                selA = st.slider(FRIENDLY_COLS["odds_A"], minA, maxA, (max(minA, selA[0]), min(maxA, selA[1])), key="selA")

    return dict(
        status_sel=status_sel, tournaments_sel=tournaments_sel, models_sel=models_sel, teams_sel=teams_sel,
        bet_sel=bet_sel, goal_sel=goal_sel, selected_date_range=selected_date_range, selH=selH, selD=selD, selA=selA
    )

# ============================
# Cards (lista mobile-first) — com sugestões destacadas
# ============================
def display_list_view(df: pd.DataFrame):
    for _, row in df.iterrows():
        dt_txt = row["date"].strftime("%d/%m %H:%M") if ("date" in df.columns and pd.notna(row["date"])) else "N/A"
        title = f"{row.get('home','?')} vs {row.get('away','?')}"
        status_txt = status_label(row.get("status","N/A"))

        hit_res = eval_result_pred_row(row)
        hit_score = eval_score_pred_row(row)
        badge_res   = "✅" if hit_res is True else ("❌" if hit_res is False else "⏳")
        badge_score = "✅" if hit_score is True else ("❌" if hit_score is False else "⏳")

        aposta_txt = market_label(row.get('bet_suggestion', '—'))
        gols_txt = market_label(row.get('goal_bet_suggestion', '—'))

        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{title}**")
                st.caption(f"{dt_txt} • {tournament_label(row.get('tournament_id'))} • {row.get('model','—')}")
                st.markdown(
                    f"**Prev.:** {market_label(row.get('result_predicted'))} {badge_res}  "
                    f"•  **Placar:** {row.get('score_predicted','—')} {badge_score}"
                )
                # Sugestões em destaque (mesmo padrão visual)
                st.markdown(
                    f"**💡 {FRIENDLY_COLS['bet_suggestion']}:** {aposta_txt}\n\n"
                    f"**⚽ {FRIENDLY_COLS['goal_bet_suggestion']}:** {gols_txt}"
                )

            with c2:
                st.markdown(f'<span class="badge badge-wait">{status_txt}</span>', unsafe_allow_html=True)
                if _norm_status_key(row.get("status","")) in FINISHED_TOKENS:
                    rh, ra = row.get("result_home"), row.get("result_away")
                    final_txt = f"{int(rh)}-{int(ra)}" if pd.notna(rh) and pd.notna(ra) else "—"
                    st.markdown(f"**Final:** {final_txt}")

            with st.expander("Detalhes, Probabilidades & Odds"):
                st.markdown(
                    f"- **Sugestão:** {aposta_txt}\n"
                    f"- **Sugestão de Gols:** {gols_txt}\n"
                    f"- **Odds 1x2:** {fmt_odd(row.get('odds_H'))} / {fmt_odd(row.get('odds_D'))} / {fmt_odd(row.get('odds_A'))}\n"
                    f"- **Prob. (H/D/A):** {fmt_prob(row.get('prob_H'))} / {fmt_prob(row.get('prob_D'))} / {fmt_prob(row.get('prob_A'))}"
                )

                st.markdown("---")
                st.markdown("**Over/Under (Prob. — Odd)**")

                under_lines = []
                if _exists(df, "prob_under_0_5"): under_lines.append(f"- **Under 0.5:** {_po(row, 'prob_under_0_5', 'odds_match_goals_0.5_under')}")
                if _exists(df, "prob_under_1_5"): under_lines.append(f"- **Under 1.5:** {_po(row, 'prob_under_1_5', 'odds_match_goals_1.5_under')}")
                if _exists(df, "prob_under_2_5"): under_lines.append(f"- **Under 2.5:** {_po(row, 'prob_under_2_5', 'odds_match_goals_2.5_under')}")
                if _exists(df, "prob_under_3_5"): under_lines.append(f"- **Under 3.5:** {_po(row, 'prob_under_3_5', 'odds_match_goals_3.5_under')}")

                over_lines = []
                if _exists(df, "prob_over_0_5"): over_lines.append(f"- **Over 0.5:** {_po(row, 'prob_over_0_5', 'odds_match_goals_0.5_over')}")
                if _exists(df, "prob_over_1_5"): over_lines.append(f"- **Over 1.5:** {_po(row, 'prob_over_1_5', 'odds_match_goals_1.5_over')}")
                if _exists(df, "prob_over_2_5"): over_lines.append(f"- **Over 2.5:** {_po(row, 'prob_over_2_5', 'odds_match_goals_2.5_over')}")
                if _exists(df, "prob_over_3_5"): over_lines.append(f"- **Over 3.5:** {_po(row, 'prob_over_3_5', 'odds_match_goals_3.5_over')}")

                if under_lines:
                    st.markdown("\n".join(under_lines))
                if over_lines:
                    st.markdown("\n".join(over_lines))

                if _exists(df, "prob_btts_yes") or _exists(df, "prob_btts_no"):
                    st.markdown("---")
                    st.markdown("**BTTS (Prob. — Odd)**")
                    if _exists(df, "prob_btts_yes"):
                        st.markdown(f"- **Ambos marcam — Sim:** {_po(row, 'prob_btts_yes', 'odds_btts_yes')}")
                    if _exists(df, "prob_btts_no"):
                        st.markdown(f"- **Ambos marcam — Não:** {_po(row, 'prob_btts_no', 'odds_btts_no')}")

            st.markdown('</div>', unsafe_allow_html=True)
            st.write("")

# ============================
# App principal
# ============================
try:
    df = load_data()
    if df.empty:
        st.error("O arquivo `PrevisaoJogos.xlsx` está vazio ou não pôde ser lido.")
    else:
        # -------- Filtros (mobile: gaveta no topo / desktop: sidebar) --------
        flt = filtros_ui(df, prefs)

        # Visualização: usa prefs quando houver
        use_list_view_default = True if MODO_MOBILE else False
        use_list_view = prefs.get("use_list_view", use_list_view_default)
        use_list_view = True if MODO_MOBILE else st.sidebar.checkbox(
            "Usar visualização em lista (mobile)", value=use_list_view, key="use_list_view"
        )

        # Botão para salvar filtros na URL (persistência entre visitas)
        aplicar = st.button("💾 Aplicar & salvar filtros no link", use_container_width=True)
        if aplicar:
            write_filters_to_url({**flt, "use_list_view": st.session_state.get("use_list_view", use_list_view)}, MODO_MOBILE)
            st.success("Filtros salvos na URL! Você pode favoritar/compartilhar este link.")

        # -------- Filtros combinados --------
        final_mask = pd.Series(True, index=df.index)
        if flt["status_sel"] and "status" in df.columns:
            final_mask &= df["status"].isin(flt["status_sel"])
        if flt["tournaments_sel"] and "tournament_id" in df.columns:
            final_mask &= df["tournament_id"].isin(flt["tournaments_sel"])
        if flt["models_sel"] and "model" in df.columns:
            final_mask &= df["model"].isin(flt["models_sel"])
        if flt["teams_sel"] and {"home", "away"}.issubset(df.columns):
            home_str = df["home"].astype(str); away_str = df["away"].astype(str)
            final_mask &= (home_str.isin(flt["teams_sel"]) | away_str.isin(flt["teams_sel"]))
        if flt["bet_sel"] and "bet_suggestion" in df.columns:
            final_mask &= df["bet_suggestion"].isin(flt["bet_sel"])
        if flt["goal_sel"] and "goal_bet_suggestion" in df.columns:
            final_mask &= df["goal_bet_suggestion"].isin(flt["goal_sel"])
        if flt["selected_date_range"] and len(flt["selected_date_range"]) == 2 and "date" in df.columns:
            start_date, end_date = flt["selected_date_range"]
            final_mask &= (df["date"].dt.date.between(start_date, end_date)) | (df["date"].isna())
        if "odds_H" in df.columns:
            final_mask &= ((df["odds_H"] >= flt["selH"][0]) & (df["odds_H"] <= flt["selH"][1])) | (df["odds_H"].isna())
        if "odds_D" in df.columns:
            final_mask &= ((df["odds_D"] >= flt["selD"][0]) & (df["odds_D"] <= flt["selD"][1])) | (df["odds_D"].isna())
        if "odds_A" in df.columns:
            final_mask &= ((df["odds_A"] >= flt["selA"][0]) & (df["odds_A"] <= flt["selA"][1])) | (df["odds_A"].isna())

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

            real_code = pd.Series(index=df_filtered.index, dtype="object")
            real_code.loc[mask_valid & (rh > ra)] = "H"
            real_code.loc[mask_valid & (rh == ra)] = "D"
            real_code.loc[mask_valid & (rh < ra)] = "A"

            pred_code = normalize_pred_code(df_filtered.get("result_predicted", pd.Series(index=df_filtered.index, dtype="object")))
            pred_correct = mask_valid & (pred_code == real_code)
            pred_wrong   = mask_valid & pred_code.notna() & real_code.notna() & (pred_code != real_code)

            bet_codes = df_filtered.get("bet_suggestion", pd.Series(index=df_filtered.index, dtype="object"))
            bet_eval = pd.Series(index=df_filtered.index, dtype="object")
            for idx in df_filtered.index:
                bet_eval.loc[idx] = evaluate_market(bet_codes.loc[idx], rh.loc[idx], ra.loc[idx]) if mask_valid.loc[idx] else None
            bet_correct = bet_eval == True
            bet_wrong   = bet_eval == False

            goal_codes = df_filtered.get("goal_bet_suggestion", pd.Series(index=df_filtered.index, dtype="object"))
            goal_eval = pd.Series(index=df_filtered.index, dtype="object")
            for idx in df_filtered.index:
                goal_eval.loc[idx] = evaluate_market(goal_codes.loc[idx], rh.loc[idx], ra.loc[idx]) if mask_valid.loc[idx] else None
            goal_correct = goal_eval == True
            goal_wrong   = goal_eval == False

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
                    "result_home", "result_away",
                    "bet_suggestion", "goal_bet_suggestion",
                    "odds_H", "odds_D", "odds_A"
                ]
                base_cols = [c for c in base_cols if c in df_filtered.columns]
                display_df_full = apply_friendly_for_display(df_filtered[base_cols])
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
                st.dataframe(display_df, use_container_width=True, hide_index=True)

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
                k1 = st.container(); k2 = st.container(); k3 = st.container()
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
