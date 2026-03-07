import pandas as pd

from analysis_service import compute_hit_columns
from analysis import get_best_model_by_market
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
