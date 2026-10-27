import numpy as np
import pandas as pd

def _safe_series(df, universe):
    if df is None:
        return pd.Series(np.nan, index=universe, dtype=float)
    try:
        s = df.reindex(columns=universe).iloc[-1]
        return pd.to_numeric(s, errors='coerce').astype(float)
    except Exception:
        return pd.Series(np.nan, index=universe, dtype=float)

def generate_weights(history, universe):
    universe = pd.Index([str(x) for x in universe])
    w = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return w
    ao_df = history.get('adjusted_open', None)
    if ao_df is None or len(ao_df.index) < 6:
        return w
    ao = ao_df.reindex(columns=universe).astype(float)
    pos = ao > 0
    ret = ao / ao.shift(1) - 1.0
    ret = ret.where(pos & pos.shift(1), np.nan)
    ret5 = ao / ao.shift(5) - 1.0
    ret5 = ret5.where(pos & pos.shift(5), np.nan)
    vol20 = ret.rolling(20, min_periods=10).std(ddof=0)
    vol60 = ret.rolling(60, min_periods=20).std(ddof=0)
    eps = 1e-8
    comp = -np.log((vol20 + eps) / (vol60 + eps))
    volume = history.get('volume', None)
    close = history.get('close', None)
    if volume is not None:
        volm = volume.reindex(columns=universe).astype(float)
        if close is not None:
            price = close.reindex(columns=universe).astype(float)
            price = price.where(price > 0, ao)
        else:
            price = ao
        dollar = (price * volm).replace([np.inf, -np.inf], np.nan)
        dv20 = dollar.rolling(20, min_periods=10).mean()
    else:
        dv20 = None
    r = ret5.iloc[-1].reindex(universe)
    c = comp.iloc[-1].reindex(universe)
    px = ao.iloc[-1].reindex(universe)
    valid = pd.Series(np.isfinite(r) & np.isfinite(c) & np.isfinite(px) & (px > 0), index=universe)
    if dv20 is not None:
        d = dv20.iloc[-1].reindex(universe)
        valid = valid & pd.Series(np.isfinite(d) & (d > 0), index=universe)
    else:
        d = pd.Series(np.nan, index=universe)
    elig = history.get('eligible', None)
    if elig is not None:
        try:
            e = elig.reindex(columns=universe).iloc[-1]
            e = e.fillna(False).astype(bool)
            valid = valid & e.reindex(universe).fillna(False)
        except Exception:
            pass
    valid = valid.fillna(False).astype(bool)
    if dv20 is not None and valid.sum() >= 20:
        dr = d.rank(pct=True)
        valid = valid & (dr >= 0.10)
    if valid.sum() < 2:
        return w
    rv = r[valid]
    cv = c[valid]
    rank_ret = rv.rank(method='average', pct=True)
    rank_comp = cv.rank(method='average', pct=True)
    score = 0.65 * rank_ret + 0.35 * rank_comp
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 2:
        return w
    score = score.sort_values(ascending=False)
    n = len(score)
    k = min(25, n // 2)
    if k < 1:
        return w
    weight = min(0.02, 1.0 / (2.0 * k))
    longs = score.index[:k]
    shorts = score.index[-k:]
    w.loc[longs] = weight
    w.loc[shorts] = -weight
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return w.astype(float)