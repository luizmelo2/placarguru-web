import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
from typing import Any, Tuple, Optional

# ============================
# Configuração da página
# ============================
st.set_page_config(layout="wide", page_title="Análise de Previsões de Futebol")
st.title("Análise de Previsões de Futebol")

# --- Tema escuro global (app + sidebar) ---
st.markdown("""
<style>
/* fundo da app */
#.stApp { background-color: #000000; color: #e5e7eb; }

/* header transparente */
#[data-testid="stHeader"] { background: rgba(0,0,0,0); }

/* sidebar escura */
#[data-testid="stSidebar"] { background-color: #0b0b0b; }

/* links e inputs mais legíveis no escuro */
#a { color: #60a5fa; }
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
# Status (apenas finished / nostarted)
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
    Tenta extrair (home, away) do campo score_predicted.
    Aceita formatos:
      - '2-1', '2x1', '2 : 1', '2 – 1' etc
      - lista/tupla [2,1]
      - dict {'home':2, 'away':1} ou {'h':2,'a':1}
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

    # string: usar regex para "n algo n"
    s = str(x)
    m = re.search(r"(\d+)\D+(\d+)", s)
    if m:
        try:
            return int(m.group(1)), int(m.group(2))
        except Exception:
            return (None, None)

    return (None, None)

def eval_result_pred_row(row) -> Optional[bool]:
    """Acerto/erro do Resultado Previsto (H/D/A) para usar na lista (expanders)."""
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
    """Acerto/erro do Placar Previsto para usar na lista (expanders)."""
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

def display_list_view(df: pd.DataFrame):
    """Lista (expanders) com rótulos amigáveis, placar final e badge ✅/❌ nas previsões."""
    for _, row in df.iterrows():
        dt_txt = row["date"].strftime("%Y-%m-%d %H:%M") if ("date" in df.columns and pd.notna(row["date"])) else "N/A"
        match_title = f"{dt_txt} - {row.get('home', '?')} vs {row.get('away', '?')} ({row.get('model', '?')})"
        with st.expander(match_title):
            st.markdown(f"**{FRIENDLY_COLS['date']}:** `{dt_txt}`")
            st.markdown(f"**{FRIENDLY_COLS['status']}:** `{status_label(row.get('status', 'N/A'))}`")
            if "tournament_id" in df.columns:
                st.markdown(f"**{FRIENDLY_COLS['tournament_id']}:** `{tournament_label(row.get('tournament_id'))}`")

            # Badges de acerto/erro
            badge_res = ""
            hit_res = eval_result_pred_row(row)
            if hit_res is True:
                badge_res = " ✅"
            elif hit_res is False:
                badge_res = " ❌"

            badge_score = ""
            hit_score = eval_score_pred_row(row)
            if hit_score is True:
                badge_score = " ✅"
            elif hit_score is False:
                badge_score = " ❌"


            status_val = _norm_status_key(row.get("status", ""))
            if status_val in FINISHED_TOKENS:
                rh = row.get("result_home"); ra = row.get("result_away")
                if pd.notna(rh) and pd.notna(ra):
                    try:
                        final_txt = f"{int(rh)}-{int(ra)}"
                    except Exception:
                        final_txt = f"{rh}-{ra}"
                else:
                    final_txt = "N/A"
                st.markdown(f"**{FRIENDLY_COLS['final_score']}:** `{final_txt}`")

            st.markdown(
                f"**Previsões:**\n"
                f"- {FRIENDLY_COLS['result_predicted']}: `{market_label(row.get('result_predicted'))}{badge_res}`\n"
                f"- {FRIENDLY_COLS['score_predicted']}: `{row.get('score_predicted', 'N/A')}{badge_score}`\n"
                f"- {FRIENDLY_COLS['bet_suggestion']}: `{market_label(row.get('bet_suggestion'))}`\n"
                f"- {FRIENDLY_COLS['goal_bet_suggestion']}: `{market_label(row.get('goal_bet_suggestion'))}`\n"

            )

            st.markdown(
                f"**Odds:**\n"
                f"- {FRIENDLY_COLS['odds_H']}: `{fmt_odd(row.get('odds_H'))}`\n"
                f"- {FRIENDLY_COLS['odds_D']}: `{fmt_odd(row.get('odds_D'))}`\n"
                f"- {FRIENDLY_COLS['odds_A']}: `{fmt_odd(row.get('odds_A'))}`"
            )

            st.markdown(
                f"**Probabilidades:**\n"
                f"- {FRIENDLY_COLS['prob_H']}: `{fmt_prob(row.get('prob_H'))} - Odd: {fmt_odd(row.get('odds_H'))}`\n"
                f"- {FRIENDLY_COLS['prob_D']}: `{fmt_prob(row.get('prob_D'))} - Odd: {fmt_odd(row.get('odds_D'))}`\n"
                f"- {FRIENDLY_COLS['prob_A']}: `{fmt_prob(row.get('prob_A'))} - Odd: {fmt_odd(row.get('odds_A'))}`\n"
                f"- {FRIENDLY_COLS['under_0_5']}: `{fmt_prob(row.get('prob_under_0_5'))} - Odd: {fmt_odd(row.get('odds_match_goals_0.5_under'))}`\n"
                f"- {FRIENDLY_COLS['under_1_5']}: `{fmt_prob(row.get('prob_under_1_5'))} - Odd: {fmt_odd(row.get('odds_match_goals_1.5_under'))}`\n"
                f"- {FRIENDLY_COLS['under_2_5']}: `{fmt_prob(row.get('prob_under_2_5'))} - Odd: {fmt_odd(row.get('odds_match_goals_2.5_under'))}`\n"
                f"- {FRIENDLY_COLS['under_3_5']}: `{fmt_prob(row.get('prob_under_3_5'))} - Odd: {fmt_odd(row.get('odds_match_goals_3.5_under'))}`\n"
                f"- {FRIENDLY_COLS['over_0_5']}: `{fmt_prob(row.get('prob_over_0_5'))} - Odd: {fmt_odd(row.get('odds_match_goals_0.5_over'))}`\n"
                f"- {FRIENDLY_COLS['over_1_5']}: `{fmt_prob(row.get('prob_over_1_5'))} - Odd: {fmt_odd(row.get('odds_match_goals_1.5_over'))}`\n"
                f"- {FRIENDLY_COLS['over_2_5']}: `{fmt_prob(row.get('prob_over_2_5'))} - Odd: {fmt_odd(row.get('odds_match_goals_2.5_over'))}`\n"
                f"- {FRIENDLY_COLS['over_3_5']}: `{fmt_prob(row.get('prob_over_3_5'))} - Odd: {fmt_odd(row.get('odds_match_goals_3.5_over'))}`\n"
                f"- {FRIENDLY_COLS['btts_yes']}: `{fmt_prob(row.get('prob_btts_yes'))} - Odd: {fmt_odd(row.get('odds_btts_yes'))}`\n"
                f"- {FRIENDLY_COLS['btts_no']}: `{fmt_prob(row.get('prob_btts_no'))} - Odd: {fmt_odd(row.get('odds_btts_no'))}`\n"
            )

# ============================
# App principal
# ============================
try:
    df = load_data()
    if df.empty:
        st.error("O arquivo `PrevisaoJogos.xlsx` está vazio ou não pôde ser lido.")
    else:
        st.sidebar.header("Filtros")
        use_list_view = st.sidebar.checkbox("Usar visualização em lista (mobile)", value=False)

        # STATUS (exibe amigável, filtra por código original)
        status_opts = sorted(df["status"].dropna().unique()) if "status" in df.columns else []
        status_sel = st.sidebar.multiselect(
            FRIENDLY_COLS["status"],
            options=status_opts,
            default=status_opts,
            format_func=status_label
        )

        # TORNEIOS
        tourn_opts = sorted(df["tournament_id"].dropna().unique().tolist()) if "tournament_id" in df.columns else []
        tournaments_sel = st.sidebar.multiselect(
            FRIENDLY_COLS["tournament_id"], options=tourn_opts, default=tourn_opts, format_func=tournament_label
        )

        # MODELOS
        model_opts = sorted(df["model"].dropna().unique()) if "model" in df.columns else []
        models_sel = st.sidebar.multiselect(FRIENDLY_COLS["model"], options=model_opts, default=model_opts)

        # EQUIPES
        if {"home", "away"}.issubset(df.columns):
            team_opts = pd.concat([df["home"], df["away"]], ignore_index=True).dropna()
            team_opts = sorted(team_opts.astype(str).unique())
        else:
            team_opts = []
        teams_sel = st.sidebar.multiselect("Equipe (Casa ou Visitante)", options=team_opts, default=team_opts)

        # SUGESTÕES
        bet_opts = sorted(df["bet_suggestion"].dropna().unique()) if "bet_suggestion" in df.columns else []
        bet_sel = st.sidebar.multiselect(FRIENDLY_COLS["bet_suggestion"], options=bet_opts, default=bet_opts, format_func=market_label)

        goal_opts = sorted(df["goal_bet_suggestion"].dropna().unique()) if "goal_bet_suggestion" in df.columns else []
        goal_sel = st.sidebar.multiselect(FRIENDLY_COLS["goal_bet_suggestion"], options=goal_opts, default=goal_opts, format_func=market_label)

        # Datas
        if "date" in df.columns and df["date"].notna().any():
            min_date = df["date"].dropna().min().date()
            max_date = df["date"].dropna().max().date()
            selected_date_range = st.sidebar.date_input("Período", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        else:
            selected_date_range = ()

        # Odds
        def _range(series: pd.Series, default=(0.0, 1.0)):
            s = series.dropna()
            return (float(s.min()), float(s.max())) if not s.empty else default

        if "odds_H" in df.columns:
            minH, maxH = _range(df["odds_H"])
            selH = st.sidebar.slider(FRIENDLY_COLS["odds_H"], minH, maxH, (minH, maxH))
        else:
            selH = (0.0, 1.0)
        if "odds_D" in df.columns:
            minD, maxD = _range(df["odds_D"])
            selD = st.sidebar.slider(FRIENDLY_COLS["odds_D"], minD, maxD, (minD, maxD))
        else:
            selD = (0.0, 1.0)
        if "odds_A" in df.columns:
            minA, maxA = _range(df["odds_A"])
            selA = st.sidebar.slider(FRIENDLY_COLS["odds_A"], minA, maxA, (minA, maxA))
        else:
            selA = (0.0, 1.0)

        # Filtros combinados
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
            # ---------- Lógica de acerto/erro (para tabela e gráficos) ----------
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

            # ---------- TABELA ----------
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

                # Estilo: verde/vermelho + odds com 2 casas
                color_ok = "#d4f8ce"
                color_bad = "#ffd7d7"

                def style_row(row: pd.Series):
                    styles = [""] * len(row)
                    idx = row.name
                    def paint(col_name, ok_series, bad_series):
                        nonlocal styles
                        if col_name in row.index:
                            col_idx = row.index.get_loc(col_name)
                            if idx in ok_series.index and bool(ok_series.loc[idx]):
                                styles[col_idx] = f"background-color: {color_ok}; font-weight: 600"
                            elif idx in bad_series.index and bool(bad_series.loc[idx]):
                                styles[col_idx] = f"background-color: {color_bad}; font-weight: 600"
                    paint("Resultado Previsto",  pred_correct,  pred_wrong)
                    paint("Placar Previsto",     score_correct, score_wrong)  # ⬅️ novo destaque
                    paint("Sugestão de Aposta",  bet_correct,   bet_wrong)
                    paint("Sugestão de Gols",    goal_correct,  goal_wrong)
                    return styles

                fmt_map = {c: "{:.2f}" for c in ["Odd Casa", "Odd Empate", "Odd Visitante"] if c in display_df.columns}
                styled = (
                    display_df
                    .style
                    .format(fmt_map, na_rep="—")
                    .apply(style_row, axis=1)
                )
                st.dataframe(styled, use_container_width=True)

            # ---------- KPIs e gráfico (mantidos para Resultado / Aposta / Gols) ----------
            def compute_acc(ok_mask: pd.Series, bad_mask: pd.Series):
                total = int((ok_mask | bad_mask).sum())
                correct = int(ok_mask.sum())
                acc = (correct / total * 100.0) if total > 0 else np.nan
                return acc, correct, total

            acc_pred, c_pred, t_pred = compute_acc(pred_correct, pred_wrong)
            acc_bet,  c_bet,  t_bet  = compute_acc(bet_correct,  bet_wrong)
            acc_goal, c_goal, t_goal = compute_acc(goal_correct, goal_wrong)

            st.subheader("Percentual de acerto (apenas jogos finalizados com placar válido)")
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
            ).properties(height=260)

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
