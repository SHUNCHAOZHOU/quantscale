import numpy as np
import pandas as pd

def generate_weights(history, universe):
    u = pd.Index(universe)
    out = pd.Series(0.0, index=u)
    try:
        close = history['adjusted_close'].reindex(columns=u)
    except Exception:
        return out
    if close.shape[0] < 22:
        return out
    last = close.iloc[-1]
    valid = last.notna()
    if 'eligible' in history:
        try:
            e = history['eligible'].reindex(columns=u).iloc[-1]
            valid = valid & e.fillna(False).astype(bool)
        except Exception:
            pass
    ret = close.pct_change(fill_method=None)
    vol = ret.iloc[-63:].std()
    vol = vol.replace([np.inf, -np.inf], np.nan).clip(lower=0.005)
    p0 = close.iloc[-1]
    p1 = close.iloc[-22]
    ret21 = p0 / p1 - 1.0
    score = -ret21 / vol
    score = score.where(valid & np.isfinite(score))
    s = score.dropna()
    if len(s) < 2:
        return out
    n = min(50, len(s) // 2)
    if n < 1:
        return out
    ss = s.sort_values()
    shorts = ss.index[:n]
    longs = ss.index[-n:]
    w = min(0.5 / n, 0.02)
    out.loc[longs] = w
    out.loc[shorts] = -w
    return out