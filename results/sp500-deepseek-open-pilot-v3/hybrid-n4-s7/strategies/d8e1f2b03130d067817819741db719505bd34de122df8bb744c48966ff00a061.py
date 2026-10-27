import numpy as np
import pandas as pd

def generate_weights(history, universe):
    idx = pd.Index(universe, dtype=object)
    weights = pd.Series(0.0, index=idx, dtype=float)
    if len(idx) == 0:
        return weights
    price_key = 'adjusted_close' if 'adjusted_close' in history else 'close'
    if price_key not in history:
        return weights
    ac = history[price_key].reindex(columns=idx).tail(320)
    if ac.shape[0] < 60:
        return weights
    close = ac.to_numpy(dtype=float)
    n, m = close.shape
    vol = None
    if 'volume' in history:
        v = history['volume'].reindex(columns=idx).tail(320)
        if v.shape[0] == n:
            vol = v.to_numpy(dtype=float)
    elig = np.ones(m, dtype=bool)
    if 'eligible' in history:
        e = history['eligible'].reindex(columns=idx).tail(1)
        if e.shape[0] == 1:
            arr = e.iloc[0].to_numpy()
            elig = np.where(pd.isna(arr), True, arr).astype(bool)
    long_win = min(200, n)
    if long_win < 60:
        return weights
    with np.errstate(invalid='ignore'):
        ma = np.nanmean(close[-long_win:], axis=0)
    last = close[-1]
    rev_win = 10
    if n <= rev_win:
        return weights
    with np.errstate(divide='ignore', invalid='ignore'):
        r5 = last / close[-1-rev_win] - 1.0
    vwin = min(60, n)
    if vol is not None:
        with np.errstate(invalid='ignore'):
            dollar = np.nanmean(vol[-vwin:] * close[-vwin:], axis=0)
    else:
        dollar = np.full(m, np.nan)
    valid = np.isfinite(last) & np.isfinite(ma) & np.isfinite(r5) & elig
    if valid.sum() < 20:
        return weights
    vol_valid = valid & np.isfinite(dollar) & (dollar > 0)
    if vol_valid.sum() >= 20:
        thr = np.nanpercentile(dollar[vol_valid], 50)
        vol_ok = vol_valid & (dollar >= thr)
    else:
        vol_ok = valid
    above = last > ma
    below = last < ma
    breadth = float(np.mean(above[valid]))
    if breadth >= 0.65:
        long_ok = above
        short_ok = below & (last < ma * 0.97)
        regime_scale = 0.85 if breadth < 0.75 else 0.7
    elif breadth <= 0.35:
        long_ok = above & (last > ma * 1.03)
        short_ok = below
        regime_scale = 0.85 if breadth > 0.25 else 0.7
    else:
        long_ok = above
        short_ok = below
        regime_scale = 1.0
    cand_long = valid & vol_ok & long_ok
    cand_short = valid & vol_ok & short_ok
    if cand_long.sum() < 25 or cand_short.sum() < 25:
        cand_long2 = valid & long_ok
        cand_short2 = valid & short_ok
        if cand_long2.sum() >= 25 and cand_short2.sum() >= 25:
            cand_long, cand_short = cand_long2, cand_short2
    if cand_long.sum() < 10 or cand_short.sum() < 10:
        cand_long = valid & above
        cand_short = valid & below
        regime_scale = min(regime_scale, 0.7)
    if cand_long.sum() == 0 or cand_short.sum() == 0:
        return weights
    long_idx = np.where(cand_long)[0]
    short_idx = np.where(cand_short)[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        trend = last / ma - 1.0
    order_long = np.lexsort((-trend[long_idx], r5[long_idx]))
    long_sorted = long_idx[order_long]
    order_short = np.lexsort((trend[short_idx], -r5[short_idx]))
    short_sorted = short_idx[order_short]
    K = min(100, len(long_sorted), len(short_sorted))
    if K <= 0:
        return weights
    per_side = min(0.5, 0.02 * K) * regime_scale
    if per_side <= 0:
        return weights
    per_w = min(per_side / K, 0.02)
    per_side = per_w * K
    long_tickers = idx[long_sorted[:K]]
    short_tickers = idx[short_sorted[:K]]
    weights.loc[long_tickers] = weights.loc[long_tickers] + per_w
    weights.loc[short_tickers] = weights.loc[short_tickers] - per_w
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    weights = weights.clip(lower=-0.02, upper=0.02)
    gross = float(weights.abs().sum())
    if gross > 1.0:
        weights = weights * (1.0 / gross)
    return weights