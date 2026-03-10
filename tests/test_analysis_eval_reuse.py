import pandas as pd

import analysis


def test_ensure_eval_df_reuses_precomputed_hits(monkeypatch):
    df = pd.DataFrame({
        "model": ["A"],
        "tournament_id": [1],
        "date": pd.to_datetime(["2024-01-01"]),
        "hit_result": [1.0],
        "hit_bet": [1.0],
        "hit_goal": [1.0],
        "hit_btts": [1.0],
        "hit_combo": [1.0],
        "hit_score": [1.0],
    })

    called = {"n": 0}

    def _boom(_):
        called["n"] += 1
        raise AssertionError("compute_hit_columns should not be called")

    monkeypatch.setattr(analysis, "compute_hit_columns", _boom)

    out = analysis._ensure_eval_df(df)
    assert out is df
    assert called["n"] == 0


def test_ensure_eval_df_computes_when_hits_missing(monkeypatch):
    df = pd.DataFrame({"model": ["A"]})

    called = {"n": 0}

    def _fake_compute(frame):
        called["n"] += 1
        out = frame.copy()
        out["hit_result"] = 1.0
        out["hit_bet"] = 1.0
        out["hit_goal"] = 1.0
        out["hit_btts"] = 1.0
        out["hit_combo"] = 1.0
        out["hit_score"] = 1.0
        return out

    monkeypatch.setattr(analysis, "compute_hit_columns", _fake_compute)

    out = analysis._ensure_eval_df(df)
    assert called["n"] == 1
    assert set(["hit_result", "hit_bet", "hit_goal", "hit_btts", "hit_combo", "hit_score"]).issubset(out.columns)
