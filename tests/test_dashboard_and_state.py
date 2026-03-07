from datetime import date
from pathlib import Path

import pandas as pd

from dashboard_service import FilterParams, apply_dashboard_filters
from state import save_persisted_filters, load_persisted_filters
import state as state_mod
import streamlit as st


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tournament_id": [1, 2, 1],
            "model": ["A", "B", "A"],
            "home": ["Team X", "Team Y", "Team X"],
            "away": ["Team Z", "Team W", "Team K"],
            "bet_suggestion": ["H", "A", "D"],
            "goal_bet_suggestion": ["over_2_5", "under_2_5", "over_1_5"],
            "result_predicted": ["H", "A", "D"],
            "btts_suggestion": ["btts_yes", "btts_no", "btts_yes"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "odds_H": [1.8, 2.1, 1.7],
            "odds_D": [3.2, 3.0, 3.1],
            "odds_A": [4.0, 3.6, 4.2],
            "prob_H": [0.82, 0.15, 0.31],
            "prob_A": [0.12, 0.84, 0.22],
            "prob_D": [0.06, 0.01, 0.80],
            "prob_over_2_5": [0.81, 0.20, 0.40],
            "prob_under_2_5": [0.19, 0.82, 0.30],
            "prob_over_1_5": [0.75, 0.40, 0.81],
            "prob_btts_yes": [0.85, 0.20, 0.83],
            "prob_btts_no": [0.15, 0.82, 0.17],
        }
    )


def test_apply_dashboard_filters_date_and_guru_only():
    df = _base_df()
    params = FilterParams(
        tournaments_sel=[1, 2],
        models_sel=["A", "B"],
        teams_sel=[],
        bet_sel=[],
        goal_sel=[],
        selected_date_range=(date(2024, 1, 1), date(2024, 1, 2)),
        sel_h=(1.0, 5.0),
        sel_d=(1.0, 5.0),
        sel_a=(1.0, 5.0),
        q_team="",
        guru_only=True,
    )

    filtered, _, _, _ = apply_dashboard_filters(df, params)

    # 3a linha está fora do range de data; as duas primeiras entram e têm guru ativo
    assert len(filtered) == 2
    assert filtered["date"].dt.date.max() <= date(2024, 1, 2)
    assert filtered["guru_highlight"].all()


def test_apply_dashboard_filters_team_search():
    df = _base_df()
    params = FilterParams(
        tournaments_sel=[1, 2],
        models_sel=["A", "B"],
        teams_sel=[],
        bet_sel=[],
        goal_sel=[],
        selected_date_range=(),
        sel_h=(1.0, 5.0),
        sel_d=(1.0, 5.0),
        sel_a=(1.0, 5.0),
        q_team="team y",
        guru_only=False,
    )

    filtered, _, _, _ = apply_dashboard_filters(df, params)

    assert len(filtered) == 1
    assert filtered.iloc[0]["home"] == "Team Y"


def test_state_persistence_cross_session_file(tmp_path, monkeypatch):
    persisted_file = tmp_path / "persisted_filters.json"
    monkeypatch.setattr(state_mod, "PERSISTED_FILTERS_FILE", persisted_file)

    st.session_state.clear()

    payload = {
        "tournaments_sel": [1],
        "models_sel": ["A"],
        "search_query": "x",
    }
    save_persisted_filters(payload)

    # Simula nova sessão
    st.session_state.pop(state_mod.PERSISTED_FILTERS_KEY, None)
    loaded = load_persisted_filters()

    assert loaded == payload
    assert persisted_file.exists()
