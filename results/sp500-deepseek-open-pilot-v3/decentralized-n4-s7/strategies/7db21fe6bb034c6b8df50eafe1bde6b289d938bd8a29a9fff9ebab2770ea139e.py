import numpy as np
import pandas as pd


def _zrank(s):
    s = s.replace([np.inf, -np.inf], np.nan)
    r = s.rank(pct=True)
    return (2.0 * r - 1.0).fillna(0.0)


def generate_weights(history, universe):
    u = list(universe)
    if len(u) == 0:
        return pd.Series(dtype=float)
    zeros = pd.Series(0.0, index=u, dtype=float)
    if history is None:
        return zeros

    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None:
        return zeros
    try:
        if ac.empty:
            return zeros
    except Exception:
        return zeros

    cols = [t for t in u if t in ac.columns]
    if len(cols) < 10:
        return zeros

    px = ac.loc[:, cols].astype(float)
    if len(px) < 40:
        return zeros

    idx = px.index
    t = idx[-1]

    # Rebalance weekly using Monday close; hold through the week.
    if t.weekday() == 0:
        sig = t
    else:
        sig = None
        for j in range(len(idx) - 1, max(-1, len(idx) - 15), -1):
            if idx[j].weekday() == 0:
                sig = idx[j]
                break
        if sig is None:
            sig = idx[-1]

    sub = px.loc[:sig]
    n = len(sub)
    if n < 30:
        return zeros

    elig = pd.Series(True, index=cols)
    e = history.get('eligible')
    if e is not None:
        try:
            ee = e.reindex(columns=cols).loc[:sig]
            if len(ee) > 0:
                elig = ee.iloc[-1].fillna(False).astype(bool)
        except Exception:
            pass

    def at(k):
        pos = n - 1 - k
        if pos < 0:
            return pd.Series(np.nan, index=cols)
        return sub.iloc[pos]

    p0 = at(0)
    p5 = at(5)
    p10 = at(10)
    p21 = at(21)
    p63 = at(63)

    rets = sub.pct_change()
    vol = rets.iloc[-61:].std()

    dv = pd.Series(np.nan, index=cols)
    v = history.get('volume')
    c = history.get('close')
    if v is not None and c is not None:
        try:
            vv = v.reindex(index=sub.index, columns=cols).iloc[-20:]
            cc = c.reindex(index=sub.index, columns=cols).iloc[-20:]
            dv = (vv * cc).median(axis=0, skipna=True)
        except Exception:
            pass

    with np.errstate(divide='ignore', invalid='ignore'):
        r5 = (p0 / p5 - 1.0).replace([np.inf, -np.inf], np.nan)
        r10 = (p0 / p10 - 1.0).replace([np.inf, -np.inf], np.nan)
        r21 = (p0 / p21 - 1.0).replace([np.inf, -np.inf], np.nan)
        r63 = (p0 / p63 - 1.0).replace([np.inf, -np.inf], np.nan)

    s_rev = _zrank(-r5) + 0.7 * _zrank(-r21) + 0.3 * _zrank(-r10)
    s_vol = _zrank(vol)
    s_trend = _zrank(r63)
    score = s_rev - 0.4 * s_vol + 0.1 * s_trend

    valid = elig & p0.notna() & (p0 > 1.0) & p5.notna() & p21.notna() & vol.notna() & (vol > 0)

    if dv.notna().any():
        posdv = dv[dv > 0]
        if len(posdv) > 5:
            thr = posdv.quantile(0.20)
            valid = valid & dv.notna() & (dv >= thr)
        else:
            valid = valid & dv.notna()
    valid = valid.fillna(False)

    if int(valid.sum()) < 12:
        valid = elig & p0.notna() & (p0 > 1.0) & p5.notna()
        valid = valid.fillna(False)
        if int(valid.sum()) < 8:
            return zeros
        score = _zrank(-r5) - 0.2 * _zrank(vol)

    score = score.where(valid)
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 8:
        return zeros

    n_side = min(25, len(score) // 2)
    if n_side < 4:
        return zeros

    w_side = min(0.02, 0.5 / float(n_side))
    ranked = score.sort_values(kind='mergesort')
    shorts = ranked.index[:n_side]
    longs = ranked.index[-n_side:]

    w = zeros.copy()
    w.loc[longs] = w_side
    w.loc[shorts] = -w_side
    return w
