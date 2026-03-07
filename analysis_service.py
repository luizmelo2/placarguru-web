"""Serviços de análise vetorizada para métricas e acurácia."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import PRED_NORMALIZER, norm_status_key, parse_score_pred


def _market_eval_series(codes: pd.Series, rh: pd.Series, ra: pd.Series) -> pd.Series:
    """Avalia mercados em lote retornando 1.0 (acerto), 0.0 (erro) ou NaN (não avaliado)."""

    out = pd.Series(np.nan, index=codes.index, dtype="float")
    valid = rh.notna() & ra.notna()
    if not valid.any():
        return out

    code = codes.astype(str).str.strip().str.lower()
    total_goals = rh.astype(float) + ra.astype(float)

    def put(mask: pd.Series, truth: pd.Series):
        idx = valid & mask
        if idx.any():
            out.loc[idx] = truth.loc[idx].astype(float)

    put(code.isin(["h", "casa", "home"]), rh > ra)
    put(code.isin(["d", "empate", "draw"]), rh == ra)
    put(code.isin(["a", "visitante", "away"]), rh < ra)
    put(code.eq("hx"), rh >= ra)
    put(code.eq("xa"), rh <= ra)

    over_mask = code.str.startswith("over_")
    if over_mask.any():
        th = pd.to_numeric(code.where(over_mask).str.replace("over_", "", regex=False).str.replace("_", ".", regex=False), errors="coerce")
        idx = valid & over_mask & th.notna()
        out.loc[idx] = (total_goals.loc[idx] > th.loc[idx]).astype(float)

    under_mask = code.str.startswith("under_")
    if under_mask.any():
        th = pd.to_numeric(code.where(under_mask).str.replace("under_", "", regex=False).str.replace("_", ".", regex=False), errors="coerce")
        idx = valid & under_mask & th.notna()
        out.loc[idx] = (total_goals.loc[idx] < th.loc[idx]).astype(float)

    put(code.eq("btts_yes"), (rh > 0) & (ra > 0))
    put(code.eq("btts_no"), (rh == 0) | (ra == 0))

    return out


def compute_hit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Computa colunas de acerto em lote para resultado, mercados, combo e placar."""

    if df.empty:
        return df.copy()

    out = df.copy()
    status_norm = out.get("status", pd.Series("", index=out.index)).astype(str).map(norm_status_key)
    finished = status_norm.eq("finished")

    rh = pd.to_numeric(out.get("result_home", pd.Series(np.nan, index=out.index)), errors="coerce")
    ra = pd.to_numeric(out.get("result_away", pd.Series(np.nan, index=out.index)), errors="coerce")
    valid_score = finished & rh.notna() & ra.notna()

    real = pd.Series(np.nan, index=out.index, dtype="object")
    real.loc[valid_score & (rh > ra)] = "H"
    real.loc[valid_score & (rh == ra)] = "D"
    real.loc[valid_score & (rh < ra)] = "A"

    pred_raw = out.get("result_predicted", pd.Series(np.nan, index=out.index)).astype(str).str.strip().str.upper()
    pred = pred_raw.map(PRED_NORMALIZER)
    hit_result = pd.Series(np.nan, index=out.index, dtype="float")

    idx_std = valid_score & pred.isin(["H", "D", "A"])
    hit_result.loc[idx_std] = (pred.loc[idx_std] == real.loc[idx_std]).astype(float)

    idx_hx = valid_score & pred.eq("Hx")
    hit_result.loc[idx_hx] = real.loc[idx_hx].isin(["H", "D"]).astype(float)

    idx_xa = valid_score & pred.eq("xA")
    hit_result.loc[idx_xa] = real.loc[idx_xa].isin(["D", "A"]).astype(float)

    hit_bet = _market_eval_series(out.get("bet_suggestion", pd.Series(np.nan, index=out.index)), rh, ra)
    hit_goal = _market_eval_series(out.get("goal_bet_suggestion", pd.Series(np.nan, index=out.index)), rh, ra)
    hit_btts = _market_eval_series(out.get("btts_suggestion", pd.Series(np.nan, index=out.index)), rh, ra)

    hit_combo = pd.Series(np.nan, index=out.index, dtype="float")
    combo_idx = hit_result.notna() & hit_bet.notna()
    hit_combo.loc[combo_idx] = ((hit_result.loc[combo_idx] == 1.0) & (hit_bet.loc[combo_idx] == 1.0)).astype(float)

    # placar previsto: usa parser por elemento para cobrir dict/list/str variados
    parsed = out.get("score_predicted", pd.Series(np.nan, index=out.index)).apply(parse_score_pred)
    ph = pd.to_numeric(parsed.apply(lambda t: t[0]), errors="coerce")
    pa = pd.to_numeric(parsed.apply(lambda t: t[1]), errors="coerce")
    hit_score = pd.Series(np.nan, index=out.index, dtype="float")
    score_idx = valid_score & ph.notna() & pa.notna()
    hit_score.loc[score_idx] = ((rh.loc[score_idx].astype(int) == ph.loc[score_idx].astype(int)) & (ra.loc[score_idx].astype(int) == pa.loc[score_idx].astype(int))).astype(float)

    out["hit_result"] = hit_result
    out["hit_bet"] = hit_bet
    out["hit_goal"] = hit_goal
    out["hit_btts"] = hit_btts
    out["hit_combo"] = hit_combo
    out["hit_score"] = hit_score

    return out
