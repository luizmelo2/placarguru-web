"""Serviços de domínio para filtros e preparação de dados do dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils import MARKET_TO_ODDS_COLS


@dataclass
class FilterParams:
    tournaments_sel: list
    models_sel: list
    teams_sel: list
    bet_sel: list
    goal_sel: list
    selected_date_range: tuple | list
    sel_h: tuple[float, float]
    sel_d: tuple[float, float]
    sel_a: tuple[float, float]
    q_team: str = ""
    guru_only: bool = False


def _market_threshold_mask(df: pd.DataFrame, suggestion_col: str, threshold: float = 0.80) -> pd.Series:
    """Calcula em lote se a sugestão de mercado atingiu o limiar de probabilidade."""

    if suggestion_col not in df.columns:
        return pd.Series(False, index=df.index)

    suggestion_values = df[suggestion_col].astype(str).str.strip()
    mask = pd.Series(False, index=df.index)

    for market_code, (prob_col, _) in MARKET_TO_ODDS_COLS.items():
        if prob_col not in df.columns:
            continue
        code_mask = suggestion_values.eq(market_code)
        if not code_mask.any():
            continue
        probs = pd.to_numeric(df.loc[code_mask, prob_col], errors="coerce")
        mask.loc[code_mask] = probs.ge(threshold).fillna(False)

    return mask


def compute_guru_columns(df: pd.DataFrame, threshold: float = 0.80) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Retorna escopo textual, flags por mercado e flag geral de destaque Guru."""

    if df.empty:
        empty = pd.Series(dtype="object")
        empty_flags = pd.DataFrame(index=df.index)
        empty_bool = pd.Series(dtype="bool")
        return empty, empty_flags, empty_bool

    result_mask = _market_threshold_mask(df, "result_predicted", threshold)
    bet_mask = _market_threshold_mask(df, "bet_suggestion", threshold)
    goal_mask = _market_threshold_mask(df, "goal_bet_suggestion", threshold)
    btts_mask = _market_threshold_mask(df, "btts_suggestion", threshold)

    flags = pd.DataFrame(
        {
            "Resultado": result_mask,
            "Sugestão": bet_mask,
            "Gols": goal_mask,
            "Ambos Marcam": btts_mask,
        },
        index=df.index,
    )

    scope = np.full(len(df), "", dtype=object)
    for label in ["Resultado", "Sugestão", "Gols", "Ambos Marcam"]:
        m = flags[label].to_numpy(dtype=bool)
        scope = np.where(m & (scope == ""), label, scope)
        scope = np.where(m & (scope != "") & (scope != label), scope + " · " + label, scope)

    scope_series = pd.Series(scope, index=df.index)
    any_flag = flags.any(axis=1)
    return scope_series, flags, any_flag


def build_filter_mask(df: pd.DataFrame, params: FilterParams) -> pd.Series:
    """Constroi máscara final de filtros."""

    mask = pd.Series(True, index=df.index)

    if params.tournaments_sel and "tournament_id" in df.columns:
        mask &= df["tournament_id"].isin(params.tournaments_sel)

    if params.models_sel and "model" in df.columns:
        mask &= df["model"].isin(params.models_sel)

    if params.teams_sel and {"home", "away"}.issubset(df.columns):
        home_ser = df["home"].astype(str)
        away_ser = df["away"].astype(str)
        mask &= (home_ser.isin(params.teams_sel) | away_ser.isin(params.teams_sel))

    if params.q_team and {"home", "away"}.issubset(df.columns):
        q = str(params.q_team).strip()
        if q:
            home_contains = df["home"].astype(str).str.contains(q, case=False, na=False, regex=False)
            away_contains = df["away"].astype(str).str.contains(q, case=False, na=False, regex=False)
            mask &= (home_contains | away_contains)

    if params.bet_sel and "bet_suggestion" in df.columns:
        mask &= df["bet_suggestion"].astype(str).isin([str(x) for x in params.bet_sel])

    if params.goal_sel and "goal_bet_suggestion" in df.columns:
        mask &= df["goal_bet_suggestion"].astype(str).isin([str(x) for x in params.goal_sel])

    if (
        params.selected_date_range
        and isinstance(params.selected_date_range, (list, tuple))
        and len(params.selected_date_range) == 2
        and "date" in df.columns
    ):
        start_date, end_date = params.selected_date_range
        mask &= df["date"].dt.date.between(start_date, end_date)

    if "odds_H" in df.columns:
        mask &= ((df["odds_H"] >= params.sel_h[0]) & (df["odds_H"] <= params.sel_h[1])) | (df["odds_H"].isna())
    if "odds_D" in df.columns:
        mask &= ((df["odds_D"] >= params.sel_d[0]) & (df["odds_D"] <= params.sel_d[1])) | (df["odds_D"].isna())
    if "odds_A" in df.columns:
        mask &= ((df["odds_A"] >= params.sel_a[0]) & (df["odds_A"] <= params.sel_a[1])) | (df["odds_A"].isna())

    return mask


def apply_dashboard_filters(df: pd.DataFrame, params: FilterParams) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Aplica filtros e anotações de destaque Guru."""

    mask = build_filter_mask(df, params)
    guru_scope_all, guru_flags_all, guru_flag_all = compute_guru_columns(df)

    if params.guru_only:
        mask &= guru_flag_all

    df_filtered = df[mask].assign(
        guru_highlight_scope=guru_scope_all[mask],
        guru_highlight=guru_flag_all[mask],
    )

    return df_filtered, guru_flags_all, guru_flag_all, mask
