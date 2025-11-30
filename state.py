"""Gerenciamento centralizado de estado e constantes da aplicação."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from typing import List, Optional, Tuple

import pandas as pd

import streamlit as st
import streamlit.components.v1 as components


# Breakpoint único compartilhado entre Python e CSS (mobile < 1024px)
MOBILE_BREAKPOINT = 1024
DEFAULT_TABLE_DENSITY = "comfortable"


@dataclass
class FilterState:
    """Representa o estado completo dos filtros da página."""

    tournaments_sel: List = None
    models_sel: List = None
    teams_sel: List = None
    bet_sel: List = None
    goal_sel: List = None
    selected_date_range: Tuple[date, date] | tuple = ()
    sel_h: tuple[float, float] | None = None
    sel_d: tuple[float, float] | None = None
    sel_a: tuple[float, float] | None = None
    search_query: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def active_count(self) -> int:
        count = 0
        if self.tournaments_sel:
            count += 1
        if self.models_sel:
            count += 1
        if self.teams_sel:
            count += 1
        if self.search_query:
            count += 1
        if self.bet_sel or self.goal_sel:
            count += 1
        if self.selected_date_range:
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
    st.session_state["pg_q_team_shared"] = state.search_query
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


def build_filter_defaults(df: pd.DataFrame, modo_mobile: bool) -> tuple[dict, dict]:
    """Centraliza a montagem dos defaults e opções disponíveis para os filtros."""

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

    default_models = []
    if model_opts:
        url_models = [v.strip().lower() for v in st.session_state.get("model_init_raw", [])]
        wanted = [m for m in model_opts if str(m).strip().lower() in url_models]
        if not wanted:
            wanted = [m for m in model_opts if str(m).strip().lower() == "combo"]
        default_models = wanted or model_opts

    min_date = df["date"].min().date() if "date" in df and df["date"].notna().any() else None
    max_date = df["date"].max().date() if "date" in df and df["date"].notna().any() else None
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
    }

    defaults.update({k: v for k, v in persisted.items() if v})

    options = {
        "model_opts": model_opts,
        "tourn_opts": tourn_opts,
        "team_opts": team_opts,
        "bet_opts": bet_opts,
        "goal_opts": goal_opts,
        "min_date": min_date,
        "max_date": max_date,
    }
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
        key="pg_viewport_sync",
    )

    try:
        viewport_width = int(width) if width else int(st.session_state.get("pg_viewport_width", default))
    except Exception:
        viewport_width = int(st.session_state.get("pg_viewport_width", default))

    st.session_state["pg_viewport_width"] = viewport_width or default
    return st.session_state["pg_viewport_width"]


@st.cache_data(show_spinner=False)
def _persisted_filters_store(state: Optional[dict] = None) -> dict:
    """Armazena seleções principais de forma leve entre sessões."""

    return state or {}


def load_persisted_filters() -> dict:
    """Recupera seleções persistidas de torneio/mercado e busca rápida."""

    return _persisted_filters_store(None)


def save_persisted_filters(state: dict) -> None:
    """Salva seleções principais em cache leve com opção de reset externo."""

    _persisted_filters_store.clear()
    _persisted_filters_store(state)


TABLE_COLUMN_PRESETS = {
    "desktop": [
        "date", "home", "away", "tournament_id", "model",
        "guru_highlight", "status", "bet_suggestion", "goal_bet_suggestion",
        "btts_suggestion", "result_predicted", "score_predicted", "final_score",
    ],
    "mobile": [
        "date", "home", "away", "tournament_id", "model", "guru_highlight",
        "status", "bet_suggestion", "result_predicted", "final_score",
    ],
    "compact": [
        "date", "home", "away", "guru_highlight",
        "status", "bet_suggestion", "result_predicted", "final_score",
    ],
}

