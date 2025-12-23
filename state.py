
"""Gerenciamento centralizado de estado e constantes da aplicação."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import date
from typing import List, Optional, Tuple
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

MOBILE_BREAKPOINT = 1024
DEFAULT_TABLE_DENSITY = "compact"

@dataclass
class FilterState:
    """Representa o estado completo dos filtros da página."""
    tournaments_sel: Optional[List[str]] = None
    models_sel: Optional[List[str]] = None
    teams_sel: Optional[List[str]] = None
    bet_sel: Optional[List[str]] = None
    goal_sel: Optional[List[str]] = None
    selected_date_range: Optional[Tuple[date, date]] = None
    sel_h: Optional[Tuple[float, float]] = None
    sel_d: Optional[Tuple[float, float]] = None
    sel_a: Optional[Tuple[float, float]] = None
    search_query: str = ""
    guru_only: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def active_count(self) -> int:
        return sum(1 for v in [self.tournaments_sel, self.models_sel, self.teams_sel, self.search_query, self.bet_sel, self.goal_sel, self.selected_date_range] if v)

def get_filter_state(defaults: Optional[dict] = None) -> FilterState:
    """Obtém o estado de filtros da sessão."""
    cache = st.session_state.get("pg_filters_cache", {})
    return FilterState(**{**defaults, **cache}) if defaults else FilterState(**cache)

def set_filter_state(state: FilterState) -> None:
    """Persiste o estado de filtros na sessão."""
    st.session_state["pg_filters_cache"] = state.to_dict()

def reset_filters(defaults: Optional[dict] = None) -> FilterState:
    """Limpa o cache de filtros."""
    st.session_state["pg_filters_cache"] = defaults or {}
    return get_filter_state(defaults)

def build_filter_defaults(df: pd.DataFrame, modo_mobile: bool) -> tuple[dict, dict]:
    """Cria os defaults e opções para os filtros."""
    opts = {
        "model_opts": sorted(df["model"].dropna().unique()),
        "tourn_opts": sorted(df["tournament_id"].dropna().unique()),
        "team_opts": sorted(pd.concat([df["home"], df["away"]]).dropna().unique()),
        "bet_opts": sorted(df["bet_suggestion"].dropna().unique()),
        "goal_opts": sorted(df["goal_bet_suggestion"].dropna().unique()),
        "min_date": df["date"].min().date(),
        "max_date": df["date"].max().date(),
    }
    defaults = {
        "tournaments_sel": opts["tourn_opts"],
        "models_sel": [m for m in opts["model_opts"] if m.lower() == "combo"] or opts["model_opts"],
        "teams_sel": [] if modo_mobile else opts["team_opts"],
        "selected_date_range": (opts["min_date"], opts["max_date"]),
        "search_query": "", "guru_only": False, "bet_sel": [], "goal_sel": [],
    }
    return defaults, opts

def detect_viewport_width(default: int = 1280) -> int:
    """Detecta a largura do viewport."""
    width = components.html("<script>window.addEventListener('resize', () => window.Streamlit.setComponentValue(window.innerWidth)); window.Streamlit.setComponentValue(window.innerWidth);</script>", height=0)
    return int(width) if width else default

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
