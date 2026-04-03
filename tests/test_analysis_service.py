import pandas as pd

from analysis_service import compute_hit_columns
from analysis import (
    get_best_model_by_market,
    build_model_ranking_by_market,
    build_weekly_accuracy_by_model,
)
from insights_service import metric_stats_for


def test_compute_hit_columns_handles_finished_and_pending():
    df = pd.DataFrame(
        {
            "status": ["finished", "nostarted", "finished"],
            "result_home": [2, None, 1],
            "result_away": [1, None, 1],
            "result_predicted": ["H", "D", "D"],
            "bet_suggestion": ["Hx", "A", "D"],
            "goal_bet_suggestion": ["over_2_5", "under_2_5", "under_3_5"],
            "btts_suggestion": ["btts_yes", "btts_no", "btts_yes"],
            "score_predicted": ["2-1", "0-0", {"home": 1, "away": 1}],
        }
    )

    out = compute_hit_columns(df)

    assert out.loc[0, "hit_result"] == 1.0
    assert pd.isna(out.loc[1, "hit_result"])
    assert out.loc[2, "hit_score"] == 1.0
    assert out.loc[0, "hit_bet"] == 1.0  # Hx em 2x1


def test_get_best_model_by_market_uses_numeric_hits_without_loss():
    df = pd.DataFrame(
        {
            "status": ["finished", "finished", "finished", "finished"],
            "tournament_id": [1, 1, 1, 1],
            "model": ["A", "A", "B", "B"],
            "result_home": [1, 0, 1, 0],
            "result_away": [0, 1, 1, 0],
            "result_predicted": ["H", "A", "D", "D"],
            "bet_suggestion": ["H", "A", "D", "D"],
            "goal_bet_suggestion": ["over_1_5", "under_1_5", "over_1_5", "under_1_5"],
            "btts_suggestion": ["btts_yes", "btts_no", "btts_yes", "btts_no"],
            "score_predicted": ["1-0", "0-1", "1-1", "0-0"],
        }
    )

    out = get_best_model_by_market(df)

    assert not out.empty
    assert set(out["Mercado de Aposta"]) == {
        "Resultado Final",
        "Sugestão de Aposta",
        "Sugestão de Gols",
        "Ambos Marcam",
    }


def test_metric_stats_for_handles_empty_metric_rows():
    metrics_df = pd.DataFrame(
        {
            "Métrica": ["Resultado"],
            "Acerto (%)": [75.0],
            "Acertos": [3],
            "Total Avaliado": [4],
        }
    )

    stats = metric_stats_for(metrics_df, ["Resultado", "Sugestão de Aposta"])

    assert stats["Resultado"] == (75.0, 3, 4)
    assert stats["Sugestão de Aposta"] == (0.0, 0, 0)


def test_compute_hit_columns_score_parsing_fast_path_and_fallback():
    df = pd.DataFrame(
        {
            "status": ["finished", "finished"],
            "result_home": [3, 2],
            "result_away": [1, 0],
            "result_predicted": ["H", "H"],
            "bet_suggestion": ["H", "H"],
            "goal_bet_suggestion": ["over_2_5", "over_1_5"],
            "btts_suggestion": ["btts_yes", "btts_no"],
            "score_predicted": ["3-1", {"home": 2, "away": 0}],
        }
    )

    out = compute_hit_columns(df)

    assert out.loc[0, "hit_score"] == 1.0  # regex string path
    assert out.loc[1, "hit_score"] == 1.0  # fallback parse_score_pred path


def test_build_model_ranking_by_market_returns_sorted_tables():
    df = pd.DataFrame(
        {
            "status": ["finished"] * 4,
            "model": ["A", "A", "B", "B"],
            "result_home": [1, 2, 0, 1],
            "result_away": [0, 1, 1, 1],
            "result_predicted": ["H", "H", "A", "D"],
            "bet_suggestion": ["H", "H", "A", "D"],
            "goal_bet_suggestion": ["over_1_5", "over_2_5", "under_1_5", "under_2_5"],
            "btts_suggestion": ["btts_no", "btts_yes", "btts_no", "btts_yes"],
            "score_predicted": ["1-0", "2-1", "0-1", "1-1"],
        }
    )

    rankings = build_model_ranking_by_market(df)

    assert "Resultado" in rankings
    result_df = rankings["Resultado"]
    assert list(result_df.columns) == [
        "Ranking",
        "Modelo",
        "Acerto (%)",
        "Acertos",
        "Total de Jogos Avaliados",
    ]
    assert result_df.iloc[0]["Acerto (%)"] >= result_df.iloc[1]["Acerto (%)"]
    assert result_df.iloc[0]["Ranking"] == 1


def test_build_model_ranking_by_market_handles_empty_input():
    out = build_model_ranking_by_market(pd.DataFrame())
    assert out == {}


def test_build_weekly_accuracy_by_model_groups_by_week_and_model():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05", "2026-01-06"]
            ),
            "model": ["A", "A", "A", "A", "A", "A"],
            "status": ["finished"] * 6,
            "result_home": [1, 1, 1, 1, 1, 1],
            "result_away": [0, 0, 0, 0, 0, 0],
            "result_predicted": ["H", "H", "H", "H", "H", "H"],
            "bet_suggestion": ["H", "H", "H", "H", "H", "H"],
            "goal_bet_suggestion": ["under_1_5", "over_1_5", "under_1_5", "over_1_5", "under_1_5", "under_1_5"],
            "btts_suggestion": ["btts_no"] * 6,
            "score_predicted": ["1-0"] * 6,
        }
    )

    weekly = build_weekly_accuracy_by_model(df, market_label="Sugestão de Gols", block_size=5)

    assert not weekly.empty
    assert set(["Bloco", "Data de Corte", "Modelo", "Acerto (%)", "Acertos", "Total"]).issubset(weekly.columns)
    assert list(weekly["Bloco"]) == [1, 2]
    assert list(weekly["Acertos"]) == [3, 1]
    assert list(weekly["Total"]) == [5, 1]


def test_build_weekly_accuracy_by_model_returns_empty_for_invalid_market():
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "model": ["A"]})
    out = build_weekly_accuracy_by_model(df, market_label="Mercado Inexistente")
    assert out.empty


def test_build_weekly_accuracy_by_model_returns_empty_for_invalid_block_size():
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "model": ["A"]})
    out = build_weekly_accuracy_by_model(df, market_label="Sugestão de Gols", block_size=0)
    assert out.empty
