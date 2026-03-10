"""Gerenciamento centralizado de estado e constantes da aplicação."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import os
import json
from pathlib import Path
from datetime import date
from typing import List, Optional, Tuple

import pandas as pd

import streamlit as st
import streamlit.components.v1 as components


# Breakpoint único compartilhado entre Python e CSS (mobile < 1024px)
MOBILE_BREAKPOINT = 1024
DEFAULT_TABLE_DENSITY = "compact"


@dataclass
class FilterState:
    """Representa o estado completo dos filtros da página."""

    tournaments_sel: List = field(default_factory=list)
    models_sel: List = field(default_factory=list)
    teams_sel: List = field(default_factory=list)
    bet_sel: List = field(default_factory=list)
    goal_sel: List = field(default_factory=list)
    selected_date_range: Tuple[date, date] | tuple = field(default_factory=tuple)
    sel_h: tuple[float, float] | None = None
    sel_d: tuple[float, float] | None = None
    sel_a: tuple[float, float] | None = None
    search_query: str = ""
    guru_only: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def active_count(self) -> int:
        return count_active_filters(self)




def count_active_filters(
    state: "FilterState",
    *,
    tournament_total: int = 0,
    model_total: int = 0,
    full_date_range: tuple[date, date] | tuple = (),
) -> int:
    """Conta filtros ativos considerando seleção total como estado neutro."""

    count = 0
    if state.tournaments_sel and (tournament_total <= 0 or len(state.tournaments_sel) != tournament_total):
        count += 1
    if state.models_sel and (model_total <= 0 or len(state.models_sel) != model_total):
        count += 1
    if state.teams_sel:
        count += 1
    if state.search_query:
        count += 1
    if state.bet_sel or state.goal_sel:
        count += 1
    if state.selected_date_range:
        if not full_date_range or tuple(state.selected_date_range) != tuple(full_date_range):
            count += 1
    if state.guru_only:
        count += 1
    return count

def get_filter_state(defaults: Optional[dict] = None) -> FilterState:
    """Obtém o estado de filtros armazenado em sessão, aplicando defaults fornecidos."""

    defaults = defaults or {}
    cache = st.session_state.get("pg_filters_cache", {})
    allowed_keys = set(FilterState.__annotations__.keys())
    merged = {k: v for k, v in {**defaults, **cache}.items() if k in allowed_keys}
    state = FilterState(**merged)
    return state


def set_filter_state(state: FilterState) -> None:
    """Persiste o estado de filtros em sessão."""

    st.session_state["pg_filters_cache"] = state.to_dict()
    save_persisted_filters({
        "tournaments_sel": state.tournaments_sel,
        "models_sel": state.models_sel,
        "search_query": state.search_query,
    })


def reset_filters(defaults: Optional[dict] = None) -> FilterState:
    """Limpa o cache de filtros e retorna um novo estado com defaults."""

    st.session_state["pg_filters_cache"] = defaults or {}
    fresh_state = get_filter_state(defaults)
    set_filter_state(fresh_state)
    return fresh_state


def _odds_default(df: pd.DataFrame, col: str, fallback: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
    """Retorna o range padrão para a coluna de odds quando o filtro está oculto."""

    if col in df.columns:
        series = df[col].dropna()
        if not series.empty:
            return (float(series.min()), float(series.max()))
    return fallback


def get_filter_options(df: pd.DataFrame) -> dict:
    """Coleta as opções disponíveis para cada filtro baseado no DataFrame."""

    model_opts = sorted(df["model"].dropna().unique()) if "model" in df.columns else []
    tourn_opts = sorted(df["tournament_id"].dropna().unique()) if "tournament_id" in df.columns else []
    team_opts = (
        sorted(pd.concat([df["home"], df["away"]]).dropna().astype(str).unique())
        if {"home", "away"}.issubset(df.columns)
        else []
    )
    bet_opts = sorted(df["bet_suggestion"].dropna().unique()) if "bet_suggestion" in df.columns else []
    goal_opts = (
        sorted(df["goal_bet_suggestion"].dropna().unique())
        if "goal_bet_suggestion" in df.columns
        else []
    )
    min_date = df["date"].min().date() if "date" in df and df["date"].notna().any() else None
    max_date = df["date"].max().date() if "date" in df and df["date"].notna().any() else None

    return {
        "model_opts": model_opts,
        "tourn_opts": tourn_opts,
        "team_opts": team_opts,
        "bet_opts": bet_opts,
        "goal_opts": goal_opts,
        "min_date": min_date,
        "max_date": max_date,
    }


def build_filter_defaults(df: pd.DataFrame, modo_mobile: bool) -> tuple[dict, dict]:
    """Centraliza a montagem dos defaults e opções disponíveis para os filtros."""

    options = get_filter_options(df)
    model_opts = options["model_opts"]
    tourn_opts = options["tourn_opts"]
    team_opts = options["team_opts"]
    bet_opts = options["bet_opts"]
    goal_opts = options["goal_opts"]

    default_models = []
    if model_opts:
        url_models = [v.strip().lower() for v in st.session_state.get("model_init_raw", [])]
        wanted = [m for m in model_opts if str(m).strip().lower() in url_models]
        if not wanted:
            wanted = [m for m in model_opts if str(m).strip().lower() == "combo"]
        default_models = wanted or model_opts

    min_date = options["min_date"]
    max_date = options["max_date"]
    persisted = load_persisted_filters()

    defaults = {
        "tournaments_sel": list(tourn_opts),
        "models_sel": default_models,
        "teams_sel": [] if modo_mobile else team_opts,
        "bet_sel": [],
        "goal_sel": [],
        "selected_date_range": (min_date, max_date) if min_date and max_date else (),
        "sel_h": _odds_default(df, "odds_H"),
        "sel_d": _odds_default(df, "odds_D"),
        "sel_a": _odds_default(df, "odds_A"),
        "search_query": persisted.get("search_query", st.session_state.get("pg_q_team_shared", "")),
        "guru_only": False,
    }

    defaults.update({k: v for k, v in persisted.items() if k in defaults})

    return defaults, options


def detect_viewport_width(default: int = 1280, debounce_ms: int = 260) -> int:
    """Sincroniza a largura do viewport com debounce e listener de rotação."""

    width = components.html(
        f"""
        <script>
          const cleanupViewportListener = () => {{
            if (window.pgViewportHandler) {{
              window.removeEventListener('resize', window.pgViewportHandler);
              window.removeEventListener('orientationchange', window.pgViewportHandler);
            }}
          }};
          cleanupViewportListener();
          const sendWidth = () => {{
            const width = window.innerWidth || document.documentElement.clientWidth;
            if (window.Streamlit && window.Streamlit.setComponentValue) {{
              window.Streamlit.setComponentValue(width);
            }}
          }};
          sendWidth();
          const handler = () => {{
            clearTimeout(window.pgViewportTimer);
            window.pgViewportTimer = setTimeout(sendWidth, {int(debounce_ms)});
          }};
          window.pgViewportHandler = handler;
          window.addEventListener('resize', handler, {{ passive: true }});
          window.addEventListener('orientationchange', handler, {{ passive: true }});
        </script>
        """,
        height=0,
    )

    try:
        viewport_width = int(width) if width else int(st.session_state.get("pg_viewport_width", default))
    except Exception:
        viewport_width = int(st.session_state.get("pg_viewport_width", default))

    st.session_state["pg_viewport_width"] = viewport_width or default
    return st.session_state["pg_viewport_width"]


PERSISTED_FILTERS_KEY = "pg_persisted_filters"
PERSISTED_FILTERS_SCHEMA_VERSION = 1
PERSISTED_FILTERS_FILE = Path(os.getenv("PG_PERSISTED_FILTERS_FILE", Path.home() / ".placarguru_filters.json"))


def _load_file_persisted_filters() -> dict:
    """Carrega filtros persistidos em arquivo local com schema versionado."""

    if not PERSISTED_FILTERS_FILE.exists():
        return {}
    try:
        raw = json.loads(PERSISTED_FILTERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(raw, dict):
        return {}

    # compatibilidade retroativa: arquivo legado sem envelope
    if "payload" not in raw:
        return raw

    version = raw.get("version")
    payload = raw.get("payload", {})
    if version != PERSISTED_FILTERS_SCHEMA_VERSION or not isinstance(payload, dict):
        return {}
    return payload


def _save_file_persisted_filters(state: dict) -> None:
    """Persiste filtros em arquivo local para sobreviver entre sessões."""

    try:
        envelope = {
            "version": PERSISTED_FILTERS_SCHEMA_VERSION,
            "payload": state,
        }
        PERSISTED_FILTERS_FILE.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # Persistência em arquivo é best-effort
        return


def load_persisted_filters() -> dict:
    """Recupera seleções persistidas de torneio/mercado e busca rápida."""

    raw = st.session_state.get(PERSISTED_FILTERS_KEY, {})
    if isinstance(raw, dict) and raw:
        return raw.copy()

    file_raw = _load_file_persisted_filters()
    if file_raw:
        st.session_state[PERSISTED_FILTERS_KEY] = file_raw.copy()
        return file_raw
    return raw.copy() if isinstance(raw, dict) else {}


def save_persisted_filters(state: dict) -> None:
    """Salva seleções principais em sessão preservando valores intencionais (inclusive vazios)."""

    if not isinstance(state, dict):
        return
    payload = state.copy()
    st.session_state[PERSISTED_FILTERS_KEY] = payload
    _save_file_persisted_filters(payload)


TABLE_COLUMN_PRESETS = {
    "desktop": [
        "date", "home", "away", "tournament_id", "model",
        "guru_highlight", "guru_highlight_scope", "status", "bet_suggestion", "goal_bet_suggestion",
        "btts_suggestion", "result_predicted", "score_predicted", "final_score",
    ],
    "mobile": [
        "date", "home", "away", "tournament_id", "model", "guru_highlight", "guru_highlight_scope",
        "status", "bet_suggestion", "result_predicted", "final_score",
    ],
    "compact": [
        "date", "home", "away", "guru_highlight", "guru_highlight_scope",
        "status", "bet_suggestion", "result_predicted", "final_score",
    ],
}
