"""Módulo para a lógica de filtragem de dados."""
import pandas as pd

def apply_filters(df: pd.DataFrame, flt: dict, guru_flag_all: pd.Series) -> pd.Series:
    """
    Aplica uma série de filtros a um DataFrame com base em um dicionário de configurações
    e uma série booleana para o filtro 'Sugestão Guru'.

    Retorna uma máscara booleana para filtrar o DataFrame original.
    """
    tournaments_sel = flt.get("tournaments_sel", [])
    models_sel = flt.get("models_sel", [])
    teams_sel = flt.get("teams_sel", [])
    bet_sel = flt.get("bet_sel", [])
    goal_sel = flt.get("goal_sel", [])
    guru_only = flt.get("guru_only", False)
    selected_date_range = flt.get("selected_date_range", ())
    sel_h = flt.get("sel_h", (0.0, 100.0))
    sel_d = flt.get("sel_d", (0.0, 100.0))
    sel_a = flt.get("sel_a", (0.0, 100.0))
    q_team = flt.get("search_query", "")

    # Máscara combinada (sem status)
    final_mask = pd.Series(True, index=df.index)

    # ▶️ Aplicar filtro global de campeonatos
    if tournaments_sel and "tournament_id" in df.columns:
        final_mask &= df["tournament_id"].isin(tournaments_sel)

    if models_sel and "model" in df.columns:
        final_mask &= df["model"].isin(models_sel)

    if teams_sel and {"home", "away"}.issubset(df.columns):
        home_ser = df["home"].astype(str)
        away_ser = df["away"].astype(str)
        final_mask &= (home_ser.isin(teams_sel) | away_ser.isin(teams_sel))

    if q_team and {"home", "away"}.issubset(df.columns):
        q = str(q_team).strip()
        if q:
            home_contains = df["home"].astype(str).str.contains(q, case=False, na=False)
            away_contains = df["away"].astype(str).str.contains(q, case=False, na=False)
            final_mask &= (home_contains | away_contains)

    if bet_sel and "bet_suggestion" in df.columns:
        final_mask &= df["bet_suggestion"].astype(str).isin([str(x) for x in bet_sel])

    if goal_sel and "goal_bet_suggestion" in df.columns:
        final_mask &= df["goal_bet_suggestion"].astype(str).isin([str(x) for x in goal_sel])

    if selected_date_range and isinstance(selected_date_range, (list, tuple)) and len(selected_date_range) == 2 and "date" in df.columns:
        start_date, end_date = selected_date_range
        final_mask &= (df["date"].dt.date.between(start_date, end_date)) | (df["date"].isna())

    if "odds_H" in df.columns:
        final_mask &= ((df["odds_H"] >= sel_h[0]) & (df["odds_H"] <= sel_h[1])) | (df["odds_H"].isna())
    if "odds_D" in df.columns:
        final_mask &= ((df["odds_D"] >= sel_d[0]) & (df["odds_D"] <= sel_d[1])) | (df["odds_D"].isna())
    if "odds_A" in df.columns:
        final_mask &= ((df["odds_A"] >= sel_a[0]) & (df["odds_A"] <= sel_a[1])) | (df["odds_A"].isna())

    if guru_only:
        final_mask &= guru_flag_all

    return final_mask
