import numpy as np
import pandas as pd


def _cs_rank(s):
    return s.rank(method='average', pct=True)


def generate_weights(history, universe):
    u = pd.Index(universe)
    zeros = pd.Series(0.0, index=u, dtype=float)
    if len(u) == 0:
        return zeros

    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None:
        return zeros

    ac = ac.reindex(columns=u)
    if ac.shape[0] < 80:
        return zeros

    tail = ac.iloc[-253:].copy()
    n = len(tail)
    if n < 80:
        return zeros

    last_px = tail.iloc[-1]

    elig_df = history.get('eligible')
    if elig_df is not None:
        try:
            e = elig_df.reindex(columns=u).iloc[-1]
            e = e.fillna(False).astype(bool)
        except Exception:
            e = pd.Series(True, index=u)
    else:
        e = pd.Series(True, index=u)
    e = e & last_px.notna()

    ret = tail.pct_change()
    vol = ret.iloc[-60:].std(ddof=0) if n >= 20 else ret.std(ddof=0)

    skip = min(5, n - 1)
    p_end = n - 1 - skip
    p_fast = max(0, n - 1 - 63)
    p_slow = max(0, n - 1 - 252)

    fast = tail.iloc[p_end] / tail.iloc[p_fast] - 1.0
    slow = tail.iloc[p_end] / tail.iloc[p_slow] - 1.0
    fast = fast.replace([np.inf, -np.inf], np.nan)
    slow = slow.replace([np.inf, -np.inf], np.nan)
    vol = vol.replace([np.inf, -np.inf], np.nan)

    base = 0.5 * _cs_rank(fast) + 0.5 * _cs_rank(slow)
    vol_clip = vol.clip(lower=0.005, upper=0.10)
    score = base / vol_clip
    score = score.where(e)
    score = score.replace([np.inf, -np.inf], np.nan)
    valid = score.dropna()

    if len(valid) < 2:
        return zeros

    nside = min(25, len(valid) // 2)
    if nside <= 0:
        return zeros

    ordered = valid.sort_values(ascending=False, kind='mergesort')
    longs = ordered.index[:nside]
    shorts = ordered.index[-nside:]

    sub = tail[valid.index]
    win = sub.iloc[-min(200, len(sub)):]
    ma = win.mean(axis=0)
    above = sub.iloc[-1] > ma
    market_above = bool(above.mean() > 0.5) if len(above) else True
    gross_scale = 1.0 if market_above else 0.5

    w = min(0.02, 0.5 / nside) * gross_scale
    weights = pd.Series(0.0, index=u, dtype=float)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.fillna(0.0).clip(-0.02, 0.02)
    return weights