"""Gerenciamento centralizado de estado e constantes da aplicação."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from typing import List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components


MOBILE_BREAKPOINT = 980


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


def reset_filters(defaults: Optional[dict] = None) -> FilterState:
    """Limpa o cache de filtros e retorna um novo estado com defaults."""

    st.session_state["pg_filters_cache"] = defaults or {}
    fresh_state = get_filter_state(defaults)
    set_filter_state(fresh_state)
    return fresh_state


def detect_viewport_width(default: int = 1280, debounce_ms: int = 260) -> int:
    """Sincroniza a largura do viewport com debounce e listener de rotação."""

    width = components.html(
        f"""
        <script>
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

