"""Módulo de utilitários com funções e constantes compartilhadas."""
import pandas as pd
import numpy as np
import re
from typing import Any, Tuple, Optional
from urllib.parse import quote_plus

def generate_sofascore_link(home_team: str, away_team: str) -> str:
    """Gera um link de busca do Google para a partida no Sofascore."""
    if not home_team or not away_team:
        return ""

    query = f'site:sofascore.com "{home_team}" vs "{away_team}"'
    return f"https://www.google.com/search?q={quote_plus(query)}"

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
    "guru_highlight": "Sugestão Guru",
    "guru_highlight_scope": "Sugestão Guru (detalhe)",
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
    "btts_suggestion": "Previsão BTTS",
}

FRIENDLY_MARKETS = {
    "H": "Casa", "D": "Empate", "A": "Visitante", "Hx": "Casa ou Empate", "xA": "Visitante ou Empate",
    "over_0_5": "Mais de 0.5 gols", "over_1_5": "Mais de 1.5 gols",
    "over_2_5": "Mais de 2.5 gols", "over_3_5": "Mais de 3.5 gols",
    "under_0_5": "Menos de 0.5 gols", "under_1_5": "Menos de 1.5 gols",
    "under_2_5": "Menos de 2.5 gols", "under_3_5": "Menos de 3.5 gols",
    "btts_yes": "BTTS Sim", "btts_no": "BTTS Não",
}

FRIENDLY_TOURNAMENTS = {
    325: "Brasileirão Série A", "325": "Brasileirão Série A",
    390: "Brasileirão Série B", "390": "Brasileirão Série B",
    17: "Premier League (Inglês)", "17": "Premier League (Inglês)",
    8: "La Liga (Espanhol)", "8": "La Liga (Espanhol)",
    23: "Italiano (Séria A)", "23": "Italiano (Séria A)",
    35: "Bundesliga (Alemão)", "35": "Bundesliga (Alemão)",
    7: "Liga dos Campeões da UEFA", "7": "Liga dos Campeões da UEFA",
    34: "Francês (Séria A)", "34": "Francês (Séria A)",
}

MARKET_TO_ODDS_COLS = {
    "H": ("prob_H", "odds_H"), "D": ("prob_D", "odds_D"), "A": ("prob_A", "odds_A"),
    "Hx": ("prob_Hx", "odds_Hx"), "xA": ("prob_xA", "odds_xA"),
    "over_0_5": ("prob_over_0_5", "odds_match_goals_0.5_over"),
    "over_1_5": ("prob_over_1_5", "odds_match_goals_1.5_over"),
    "over_2_5": ("prob_over_2_5", "odds_match_goals_2.5_over"),
    "over_3_5": ("prob_over_3_5", "odds_match_goals_3.5_over"),
    "under_0_5": ("prob_under_0_5", "odds_match_goals_0.5_under"),
    "under_1_5": ("prob_under_1_5", "odds_match_goals_1.5_under"),
    "under_2_5": ("prob_under_2_5", "odds_match_goals_2.5_under"),
    "under_3_5": ("prob_under_3_5", "odds_match_goals_3.5_under"),
    "btts_yes": ("prob_btts_yes", "odds_btts_yes"),
    "btts_no": ("prob_btts_no", "odds_btts_no"),
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
    "HX": "Hx", "XA": "xA",
}

# ============================
# Constantes de Lógica
# ============================
BTTS_PROB_THRESHOLD = 0.65
GOAL_MARKET_THRESHOLDS = ["0.5", "1.5", "2.5", "3.5"]

# ============================
# Helpers
# ============================
def is_na_like(x: Any) -> bool:
    """Verifica se um valor é 'NA-like' (None, NaN, ou string vazia/nula)."""
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str) and x.strip().lower() in {"", "nan", "none", "null"}:
        return True
    return False

def market_label(v: Any, default: str = "Sem previsão calculada") -> str:
    """Mapa amigável com fallback caso venha NaN/None/vazio."""
    if is_na_like(v):
        return default
    return FRIENDLY_MARKETS.get(v, str(v))

def _canon_tourn_key(x: Any) -> Optional[Any]:
    """Converte a chave de um torneio para um tipo canônico (int ou str)."""
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
    """Retorna o nome amigável de um torneio a partir de sua chave."""
    k = _canon_tourn_key(x)
    if k in FRIENDLY_TOURNAMENTS:
        return FRIENDLY_TOURNAMENTS[k]
    ks = str(k) if k is not None else None
    if ks in FRIENDLY_TOURNAMENTS:
        return FRIENDLY_TOURNAMENTS[ks]
    return f"Torneio {x}"


def norm_status_key(s: Any) -> str:
    """Normaliza uma string de status para um formato de chave padrão."""
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")

def status_label(s: Any) -> str:
    """Retorna o rótulo amigável para um status de jogo."""
    return FRIENDLY_STATUS_MAP.get(norm_status_key(s), str(s))

def normalize_pred_code(series: pd.Series) -> pd.Series:
    """Normaliza uma série de códigos de previsão para um formato padrão (H, D, A, etc.)."""
    if series is None:
        return pd.Series(dtype="object")
    s = series.astype(str).str.strip().str.upper()
    return s.map(lambda x: PRED_NORMALIZER.get(x, np.nan))


def _parse_threshold(token: str) -> Optional[float]:
    """Extrai um valor de limiar numérico de uma string (ex: 'over_2_5' -> 2.5)."""
    if token is None:
        return None
    t = str(token).replace("_", ".").strip()
    try:
        return float(t)
    except Exception:
        return None

# ---- formatação e wrappers ----
def fmt_odd(x: Any) -> str:
    """Formata um valor numérico como uma string de odd com duas casas decimais."""
    try:
        v = float(x)
        if pd.isna(v):
            return "N/A"
        return f"{v:.2f}"
    except Exception:
        return "N/A"


def fmt_prob(x: Any) -> str:
    """Formata um valor numérico (0-1) como uma string de porcentagem."""
    try:
        v = float(x)
        if pd.isna(v):
            return "N/A"
        return f"{v*100:.2f}%"
    except Exception:
        return "N/A"

def green_html(txt: Any) -> str:
    """Envolve um texto em uma tag span com a classe CSS 'accent-green'."""
    return f'<span class="accent-green">{txt}</span>'

def get_prob_and_odd_for_market(row: pd.Series, market_code: Any) -> str:
    """Busca a probabilidade e a odd para um mercado específico e retorna uma string formatada."""
    if pd.isna(market_code):
        return ""

    market_code_str = str(market_code).strip()
    if market_code_str in MARKET_TO_ODDS_COLS:
        prob_col, odd_col = MARKET_TO_ODDS_COLS[market_code_str]
        prob = row.get(prob_col)
        odd = row.get(odd_col)

        prob_str = fmt_prob(prob) if pd.notna(prob) else "N/A"
        odd_str = fmt_odd(odd) if pd.notna(odd) else "N/A"

        return f'<span class="text-odds">(Prob: {prob_str}, Odd: {odd_str})</span>'
    return ""

def evaluate_market(code: Any, rh: Any, ra: Any) -> Optional[bool]:
    """Avalia o resultado de um mercado de aposta (código) contra um placar final (rh, ra)."""
    if pd.isna(code) or pd.isna(rh) or pd.isna(ra):
        return None
    s = str(code).strip().lower()
    if s in ("h", "casa", "home"):
        return rh > ra
    if s in ("d", "empate", "draw"):
        return rh == ra
    if s in ("a", "visitante", "away"):
        return rh < ra
    if s == "hx":
        return rh >= ra
    if s == "xa":
        return rh <= ra
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

def parse_score_pred(x: Any) -> Tuple[Optional[int], Optional[int]]:
    """Extrai um placar (casa, visitante) de vários formatos de entrada (dict, lista, tupla, str)."""
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

def fmt_score_pred_text(x: Any, default: str = "Sem previsão calculada") -> str:
    """Formata um placar previsto em uma string 'C-V' ou retorna um texto padrão."""
    ph, pa = parse_score_pred(x)
    if ph is None or pa is None:
        return default
    return f"{ph}-{pa}"

def eval_result_pred_row(row: pd.Series) -> Optional[bool]:
    """Avalia se a previsão do resultado (1x2) está correta para uma linha de dados."""
    if norm_status_key(row.get("status", "")) not in FINISHED_TOKENS:
        return None
    rh, ra = row.get("result_home"), row.get("result_away")
    if pd.isna(rh) or pd.isna(ra):
        return None
    real = "H" if rh > ra else ("D" if rh == ra else "A")
    pred = PRED_NORMALIZER.get(
        str(row.get("result_predicted")).strip().upper(), np.nan
    )
    if pd.isna(pred):
        return None

    if pred == "Hx":
        return real in ("H", "D")
    if pred == "xA":
        return real in ("D", "A")
    return pred == real

def eval_score_pred_row(row: pd.Series) -> Optional[bool]:
    """Avalia se a previsão do placar exato está correta."""
    if norm_status_key(row.get("status", "")) not in FINISHED_TOKENS:
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


def _row_is_finished(row: pd.Series) -> bool:
    """Verifica se uma linha representa um jogo finalizado com placar válido."""
    if norm_status_key(row.get("status", "")) not in FINISHED_TOKENS:
        return False
    rh, ra = row.get("result_home"), row.get("result_away")
    return pd.notna(rh) and pd.notna(ra)


def eval_bet_row(row: pd.Series) -> Optional[bool]:
    """Avalia se a sugestão de aposta principal ('bet_suggestion') está correta."""
    if not _row_is_finished(row):
        return None
    return evaluate_market(
        row.get("bet_suggestion"), row.get("result_home"), row.get("result_away")
    )


def eval_goal_row(row: pd.Series) -> Optional[bool]:
    """Avalia se a sugestão de aposta de gols ('goal_bet_suggestion') está correta."""
    if not _row_is_finished(row):
        return None
    return evaluate_market(
        row.get("goal_bet_suggestion"), row.get("result_home"), row.get("result_away")
    )


def eval_btts_suggestion_row(row: pd.Series) -> Optional[bool]:
    """Avalia se a sugestão de aposta de BTTS ('btts_suggestion') está correta."""
    if not _row_is_finished(row):
        return None
    return evaluate_market(
        row.get("btts_suggestion"), row.get("result_home"), row.get("result_away")
    )


def eval_sugestao_combo_row(row: pd.Series) -> Optional[bool]:
    """
    Avalia se tanto a previsão de resultado quanto a sugestão de aposta foram corretas.
    Retorna True se ambas foram acertos, False se pelo menos uma foi erro, e None se alguma não pôde ser avaliada.
    """
    if not _row_is_finished(row):
        return None

    res_acerto = eval_result_pred_row(row)
    bet_acerto = eval_bet_row(row)

    # Se qualquer um não puder ser avaliado, o combo não pode ser avaliado
    if res_acerto is None or bet_acerto is None:
        return None

    # Se ambos foram acertos
    if res_acerto is True and bet_acerto is True:
        return True

    # Se um ou ambos foram erros (e nenhum foi None)
    return False


def _po(row: pd.Series, prob_key: str, odd_key: str) -> str:
    """Helper para formatar uma string de Probabilidade e Odd."""
    return f"{green_html(fmt_prob(row.get(prob_key)))} - Odd: {green_html(fmt_odd(row.get(odd_key)))}"

def _exists(df: pd.DataFrame, *cols: str) -> bool:
    """Verifica se uma ou mais colunas existem em um DataFrame."""
    return all(c in df.columns for c in cols)
