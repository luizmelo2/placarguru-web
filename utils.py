
"""Módulo de utilitários com funções e constantes compartilhadas."""
import pandas as pd
import re
from typing import Any, Tuple, Optional
import requests
import streamlit as st
from urllib.parse import quote_plus

def generate_sofascore_link(home: str, away: str) -> str:
    """Gera um link de busca do Sofascore."""
    if not home or not away:
        return ""
    q = f'site:sofascore.com "{home}" vs "{away}"'
    return f"https://www.google.com/search?q={quote_plus(q)}"

FRIENDLY_COLS = {
    "status": "Status", "tournament_id": "Torneio", "model": "Modelo",
    "date": "Data/Hora", "home": "Casa", "away": "Visitante",
    "result_predicted": "Resultado Previsto", "score_predicted": "Placar Previsto",
    "bet_suggestion": "Sugestão", "goal_bet_suggestion": "Gols",
    "guru_highlight": "Guru", "final_score": "Placar Final",
}
FRIENDLY_MARKETS = {
    "H": "Casa", "D": "Empate", "A": "Visitante", "Hx": "Casa/Empate", "xA": "Fora/Empate",
    "over_2.5": "Mais de 2.5", "under_2.5": "Menos de 2.5",
    "btts_yes": "Sim", "btts_no": "Não",
}
FRIENDLY_TOURNAMENTS = {
    325: "Brasileirão A", 390: "Brasileirão B", 17: "Premier League", 8: "La Liga",
    23: "Serie A", 35: "Bundesliga", 7: "Champions League", 34: "Ligue 1",
}
MARKET_TO_ODDS_COLS = {
    "H": ("prob_H", "odds_H"), "D": ("prob_D", "odds_D"), "A": ("prob_A", "odds_A"),
    "over_2.5": ("prob_over_2.5", "odds_match_goals_2.5_over"),
    "under_2.5": ("prob_under_2.5", "odds_match_goals_2.5_under"),
    "btts_yes": ("prob_btts_yes", "odds_btts_yes"), "btts_no": ("prob_btts_no", "odds_btts_no"),
}
FINISHED_TOKENS = {"finished"}
FRIENDLY_STATUS_MAP = {"finished": "Finalizado", "not_started": "Agendado"}
PRED_NORMALIZER = {"H": "H", "D": "D", "A": "A"}
RELEASE_URL = "https://github.com/luizmelo2/arquivos/releases/download/latest/PrevisaoJogos.xlsx"

def market_label(v: Any, default: str = "N/A") -> str:
    return FRIENDLY_MARKETS.get(v, default)

def tournament_label(x: Any) -> str:
    return FRIENDLY_TOURNAMENTS.get(x, f"Torneio {x}")

def norm_status_key(s: Any) -> str:
    return str(s).lower().replace(" ", "_")

def status_label(s: Any) -> str:
    return FRIENDLY_STATUS_MAP.get(norm_status_key(s), str(s))

def fmt_odd(x: Any) -> str:
    return f"{float(x):.2f}" if pd.notna(x) else "N/A"

def fmt_prob(x: Any) -> str:
    return f"{float(x)*100:.1f}%" if pd.notna(x) else "N/A"

def get_prob_and_odd(row: pd.Series, market: Any) -> str:
    if pd.notna(market) and market in MARKET_TO_ODDS_COLS:
        p_col, o_col = MARKET_TO_ODDS_COLS[market]
        return f"({fmt_prob(row.get(p_col))}, {fmt_odd(row.get(o_col))})"
    return ""

def evaluate_market(code: Any, rh: Any, ra: Any) -> Optional[bool]:
    if pd.isna(code) or pd.isna(rh) or pd.isna(ra): return None
    s = str(code).lower()
    if s == "h": return rh > ra
    if s == "d": return rh == ra
    if s == "a": return rh < ra
    if "over" in s: return (rh + ra) > 2.5
    if "under" in s: return (rh + ra) < 2.5
    if s == "btts_yes": return rh > 0 and ra > 0
    if s == "btts_no": return rh == 0 or ra == 0
    return None

def parse_score(x: Any) -> Tuple[Optional[int], Optional[int]]:
    if pd.isna(x): return (None, None)
    if isinstance(x, dict): return (x.get("home"), x.get("away"))
    m = re.search(r'(\d+)\D+(\d+)', str(x))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def fmt_score(x: Any, default: str = "N/A") -> str:
    ph, pa = parse_score(x)
    return f"{ph}-{pa}" if ph is not None else default

def _is_finished(row: pd.Series) -> bool:
    return norm_status_key(row.get("status", "")) in FINISHED_TOKENS and pd.notna(row.get("result_home"))

def eval_result_pred(row: pd.Series) -> Optional[bool]:
    if not _is_finished(row): return None
    real = "H" if row.result_home > row.result_away else ("D" if row.result_home == row.result_away else "A")
    pred = PRED_NORMALIZER.get(str(row.get("result_predicted")).upper())
    return pred == real if pred else None

@st.cache_data
def fetch_release_file(url: str):
    r = requests.get(url, timeout=60, verify=False)
    r.raise_for_status()
    return r.content, r.headers.get("ETag", ""), r.headers.get("Last-Modified", "")

@st.cache_data
def load_data(_file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_excel(_file_bytes)
    if "date" in df: df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["bet_suggestion", "goal_bet_suggestion"]:
        if col in df: df[col] = df[col].apply(lambda x: x.get("market") if isinstance(x, dict) else x)
    return df
