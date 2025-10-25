import pandas as pd
import numpy as np
import re
from typing import Any, Tuple, Optional
import requests
import streamlit as st

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
    "over_0_5": "Mais de 0.5 gols",
    "over_1_5": "Mais de 1.5 gols",
    "over_2_5": "Mais de 2.5 gols",
    "over_3_5": "Mais de 3.5 gols",
    "under_0_5": "Menos de 0.5 gols",
    "under_1_5": "Menos de 1.5 gols",
    "under_2_5": "Menos de 2.5 gols",
    "under_3_5": "Menos de 3.5 gols",
    "btts_yes": "Ambos Marcam Sim",
    "btts_no": "Ambos Marcam Não",
    "final_score": "Resultado Final",
}

FRIENDLY_MARKETS = {
    "H": "Casa", "D": "Empate", "A": "Visitante",
    "over_0_5": "Mais de 0.5 gols", "over_1_5": "Mais de 1.5 gols",
    "over_2_5": "Mais de 2.5 gols", "over_3_5": "Mais de 3.5 gols",
    "under_0_5": "Menos de 0.5 gols", "under_1_5": "Menos de 1.5 gols",
    "under_2_5": "Menos de 2.5 gols", "under_3_5": "Menos de 3.5 gols",
    "btts_yes": "Ambos Marcam Sim", "btts_no": "Ambos Marcam Não",
}

FRIENDLY_TOURNAMENTS = {
    325: "Brasileirão Série A", "325": "Brasileirão Série A",
    390: "Brasileirão Série B", "390": "Brasileirão Série B",
    17: "Premier League (Inglês)", "17": "Premier League (Inglês)",
    8: "La Liga (Espanhol)", "8": "La Liga (Espanhol)",
    23: "Italiano (Séria A)", "23": "Italiano (Séria A)",
    35: "Bundesliga (Alemão)", "35": "Bundesliga (Alemão)",
}

# ============================
# Status (apenas finished)
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
def is_na_like(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str) and x.strip().lower() in {"", "nan", "none", "null"}:
        return True
    return False

def safe_text(v: Any, default: str = "Sem previsão calculada") -> str:
    return default if is_na_like(v) else str(v)

def market_label(v: Any, default: str = "Sem previsão calculada") -> str:
    """Mapa amigável com fallback caso venha NaN/None/vazio."""
    if is_na_like(v):
        return default
    return FRIENDLY_MARKETS.get(v, str(v))

def _canon_tourn_key(x: Any):
    if x is None or (isinstance(x, float) and np.isnan(x)): return None
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, float): return int(x) if float(x).is_integer() else x
    try:
        s = str(x).strip(); return int(s)
    except Exception:
        return str(x).strip()

def tournament_label(x: Any) -> str:
    k = _canon_tourn_key(x)
    if k in FRIENDLY_TOURNAMENTS: return FRIENDLY_TOURNAMENTS[k]
    ks = str(k) if k is not None else None
    if ks in FRIENDLY_TOURNAMENTS: return FRIENDLY_TOURNAMENTS[ks]
    return f"Torneio {x}"

def norm_status_key(s: Any) -> str:
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")

def status_label(s: Any) -> str:
    return FRIENDLY_STATUS_MAP.get(norm_status_key(s), str(s))

def normalize_pred_code(series: pd.Series) -> pd.Series:
    if series is None: return pd.Series(dtype="object")
    s = series.astype(str).str.strip().str.upper()
    return s.map(lambda x: PRED_NORMALIZER.get(x, np.nan))

def _parse_threshold(token: str) -> Optional[float]:
    if token is None: return None
    t = str(token).replace("_", ".").strip()
    try: return float(t)
    except Exception: return None

# ---- formatação e wrappers ----
def fmt_odd(x):
    try:
        v = float(x)
        if pd.isna(v): return "N/A"
        return f"{v:.2f}"
    except Exception:
        return "N/A"

def fmt_prob(x):
    try:
        v = float(x)
        if pd.isna(v): return "N/A"
        return f"{v*100:.2f}%"
    except Exception:
        return "N/A"

def green_html(txt: Any) -> str:
    return f'<span class="accent-green">{txt}</span>'

def evaluate_market(code: Any, rh: Any, ra: Any) -> Optional[bool]:
    if pd.isna(code) or pd.isna(rh) or pd.isna(ra): return None
    s = str(code).strip().lower()
    if s in ("h", "casa", "home"): return rh > ra
    if s in ("d", "empate", "draw"): return rh == ra
    if s in ("a", "visitante", "away"): return rh < ra
    if s.startswith("over_"):
        th = _parse_threshold(s.split("over_", 1)[1]); return None if th is None else (float(rh) + float(ra)) > th
    if s.startswith("under_"):
        th = _parse_threshold(s.split("under_", 1)[1]); return None if th is None else (float(rh) + float(ra)) < th
    if s == "btts_yes": return (float(rh) > 0) and (float(ra) > 0)
    if s == "btts_no":  return (float(rh) == 0) or (float(ra) == 0)
    return None

def parse_score_pred(x: Any) -> Tuple[Optional[int], Optional[int]]:
    if x is None or (isinstance(x, float) and np.isnan(x)): return (None, None)
    if isinstance(x, dict):
        for hk, ak in (("home","away"), ("h","a")):
            if hk in x and ak in x:
                try: return int(x[hk]), int(x[ak])
                except Exception: return (None, None)
    if isinstance(x, (list,tuple)) and len(x)==2:
        try: return int(x[0]), int(x[1])
        except Exception: return (None, None)
    s = str(x); m = re.search(r"(\d+)\D+(\d+)", s)
    if m:
        try: return int(m.group(1)), int(m.group(2))
        except Exception: return (None, None)
    return (None, None)

def fmt_score_pred_text(x: Any, default: str = "Sem previsão calculada") -> str:
    ph, pa = parse_score_pred(x)
    if ph is None or pa is None:
        return default
    return f"{ph}-{pa}"

def eval_result_pred_row(row) -> Optional[bool]:
    if norm_status_key(row.get("status","")) not in FINISHED_TOKENS: return None
    rh, ra = row.get("result_home"), row.get("result_away")
    if pd.isna(rh) or pd.isna(ra): return None
    real = "H" if rh > ra else ("D" if rh == ra else "A")
    pred = PRED_NORMALIZER.get(str(row.get("result_predicted")).strip().upper(), np.nan)
    if pd.isna(pred): return None
    return pred == real

def eval_score_pred_row(row) -> Optional[bool]:
    if norm_status_key(row.get("status","")) not in FINISHED_TOKENS: return None
    rh, ra = row.get("result_home"), row.get("result_away")
    if pd.isna(rh) or pd.isna(ra): return None
    ph, pa = parse_score_pred(row.get("score_predicted"))
    if ph is None or pa is None: return None
    try: return (int(rh) == int(ph)) and (int(ra) == int(pa))
    except Exception: return None

def _row_is_finished(row) -> bool:
    if norm_status_key(row.get("status","")) not in FINISHED_TOKENS: return False
    rh, ra = row.get("result_home"), row.get("result_away")
    return pd.notna(rh) and pd.notna(ra)

def eval_bet_row(row) -> Optional[bool]:
    if not _row_is_finished(row): return None
    return evaluate_market(row.get("bet_suggestion"), row.get("result_home"), row.get("result_away"))

def eval_goal_row(row) -> Optional[bool]:
    if not _row_is_finished(row): return None
    return evaluate_market(row.get("goal_bet_suggestion"), row.get("result_home"), row.get("result_away"))

def _po(row, prob_key: str, odd_key: str) -> str:
    return f"{green_html(fmt_prob(row.get(prob_key)))} - Odd: {green_html(fmt_odd(row.get(odd_key)))}"

def _exists(df: pd.DataFrame, *cols) -> bool:
    return all(c in df.columns for c in cols)

def _get_prob_from_market_code(code: Any) -> Optional[str]:
    """Mapeia um código de mercado para sua coluna de probabilidade."""
    if is_na_like(code):
        return None
    s = str(code).strip().lower()
    if s in ("h", "casa", "home"): return "prob_H"
    if s in ("d", "empate", "draw"): return "prob_D"
    if s in ("a", "visitante", "away"): return "prob_A"
    if s.startswith("over_") or s.startswith("under_") or s.startswith("btts_"):
        return f"prob_{s}"
    return None

# ============================
# Download da release (GitHub)
# ============================
RELEASE_URL = "https://github.com/luizmelo2/arquivos/releases/download/latest/PrevisaoJogos.xlsx"

@st.cache_data(show_spinner=False)
def fetch_release_file(url: str):
    """
    Baixa o arquivo da Release pública do GitHub.
    Retorna: (bytes, etag, last_modified)
    """
    r = requests.get(url, timeout=60, verify=False)
    r.raise_for_status()
    etag = r.headers.get("ETag", "")
    last_mod = r.headers.get("Last-Modified", "")
    return r.content, etag, last_mod

# ============================
# Carregamento e normalização
# ============================
@st.cache_data(show_spinner=False)
def load_data(_file_bytes: bytes) -> pd.DataFrame:
    """
    Carrega o Excel. O parâmetro _file_bytes é usado para invalidar o cache
    quando o conteúdo do arquivo mudar.
    """
    df = pd.read_excel(_file_bytes)

    # Tipos
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["odds_H", "odds_D", "odds_A"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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
