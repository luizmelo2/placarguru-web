"""Módulo para carregamento e normalização de dados."""
import pandas as pd
import numpy as np
import requests
import streamlit as st

# ============================
# Download da release (GitHub)
# ============================
RELEASE_URL = "https://github.com/luizmelo2/arquivos/releases/download/latest/PrevisaoJogos.xlsx"
#RELEASE_URL = "PrevisaoJogos.xlsx"

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


def _fetch_release_file(local_path: str):
    """
    Lê um arquivo Excel local e retorna o conteúdo em bytes.
    Como não há requisição HTTP, não existem ETag ou Last-Modified reais.
    Retorna: (bytes, etag, last_modified)
    """
    with open(local_path, "rb") as f:
        content = f.read()

    # Você pode devolver campos vazios ou simulados
    etag = ""
    last_mod = ""
    return content, etag, last_mod

# ============================
# Carregamento e normalização
# ============================
def _calculate_single_dc(df: pd.DataFrame, prob_col: str, odd_col: str, prob1_col: str, prob2_col: str) -> pd.DataFrame:
    """Função auxiliar para calcular uma aposta de dupla chance."""
    if prob_col not in df.columns or df[prob_col].isnull().all():
        if prob1_col in df.columns and prob2_col in df.columns:
            df[prob_col] = df[prob1_col] + df[prob2_col]

    if odd_col not in df.columns or df[odd_col].isnull().all():
        if prob_col in df.columns:
            df[odd_col] = df[prob_col].apply(lambda p: 1 / p if p > 0 else np.nan)
    return df

def calculate_double_chance(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula probabilidades e odds para apostas de dupla chance se não existirem."""
    df = _calculate_single_dc(df, "prob_Hx", "odds_Hx", "prob_H", "prob_D")
    df = _calculate_single_dc(df, "prob_xA", "odds_xA", "prob_D", "prob_A")
    return df

@st.cache_data(show_spinner=False)
def load_data(_file_bytes: bytes) -> pd.DataFrame:
    """
    Carrega o Excel. O parâmetro _file_bytes é usado para invalidar o cache
    quando o conteúdo do arquivo mudar.
    """
    df = pd.read_excel(_file_bytes)
    df = calculate_double_chance(df)

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
