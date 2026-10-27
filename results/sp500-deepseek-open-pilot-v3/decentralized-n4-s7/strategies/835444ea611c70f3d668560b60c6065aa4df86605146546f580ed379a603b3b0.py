import numpy as np
import pandas as pd

_CACHE = {}


def _zrank(s, valid=None):
    s = s.astype(float).replace([np.inf, -np.inf], np.nan)
    if valid is not None:
        s = s.where(valid)
    r = s.rank(pct=True)
    return (2.0 * r - 1.0).fillna(0.0)


def generate_weights(history, universe):
    u = list(universe)
    if len(u) == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=u, dtype=float)
    if history is None:
        return zero
    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None:
        return zero
    try:
        if ac.empty:
            return zero
    except Exception:
        return zero
    cols = [t for t in u if t in ac.columns]
    if len(cols) < 10:
        return zero
    px = ac.loc[:, cols].astype(float)
    if not px.index.is_monotonic_increasing:
        px = px.sort_index()
    if len(px) > 400:
        px = px.iloc[-400:]
    px = px.dropna(how='all')
    if len(px) < 40:
        return zero
    idx = px.index
    t = idx[-1]
    try:
        nb = t + pd.offsets.BDay(1)
        at_me = (nb.year != t.year) or (nb.month != t.month)
    except Exception:
        at_me = False
    if at_me:
        sig = t
    else:
        try:
            months = np.asarray(idx.year, dtype=np.int64) * 12 + np.asarray(idx.month, dtype=np.int64)
            cur = int(t.year) * 12 + int(t.month)
            prior = idx[months < cur]
        except Exception:
            prior = idx[:-1]
        if len(prior) == 0:
            return zero
        sig = prior[-1]
    key = (sig, tuple(u))
    if key in _CACHE:
        return _CACHE[key].reindex(u).fillna(0.0).astype(float)
    sub = px.loc[:sig]
    n = len(sub)
    if n < 274:
        return zero
    p0 = sub.iloc[-1]
    skip = 21
    long = 252
    end_i = n - 1 - skip
    start_i = end_i - long
    if start_i < 0:
        return zero
    p_end = sub.iloc[end_i]
    p_start = sub.iloc[start_i]
    mom12 = (p_end / p_start - 1.0).replace([np.inf, -np.inf], np.nan)
    start6 = end_i - 126
    if start6 >= 0:
        mom6 = (p_end / sub.iloc[start6] - 1.0).replace([np.inf, -np.inf], np.nan)
    else:
        mom6 = pd.Series(np.nan, index=cols)
    rets = sub.pct_change()
    if len(rets) >= 126:
        vol = rets.iloc[-126:].std()
    else:
        vol = rets.std()
    dvol = pd.Series(np.nan, index=cols)
    v = history.get('volume')
    c = history.get('close')
    if v is not None and c is not None:
        try:
            vv = v.reindex(columns=cols)
            cc = c.reindex(columns=cols)
            vv = vv.reindex(sub.index).tail(20)
            cc = cc.reindex(sub.index).tail(20)
            dvol = (vv * cc).median(axis=0, skipna=True)
        except Exception:
            pass
    elig = pd.Series(True, index=cols)
    e = history.get('eligible')
    if e is not None:
        try:
            ee = e.reindex(columns=cols)
            ee = ee.reindex(sub.index)
            if len(ee) > 0:
                elig = ee.iloc[-1].fillna(False).astype(bool)
        except Exception:
            pass
    valid = elig.reindex(cols).fillna(False).astype(bool)
    valid = valid & p0.notna() & (p0 > 1.0) & mom12.notna() & mom6.notna() & vol.notna() & (vol > 0.0)
    if dvol.notna().any():
        pos = dvol[dvol > 0.0]
        if len(pos) >= 10:
            thr = pos.quantile(0.20)
            valid = valid & dvol.notna() & (dvol >= thr)
        else:
            valid = valid & dvol.notna()
    valid = valid.fillna(False)
    if int(valid.sum()) < 20:
        valid = elig.reindex(cols).fillna(False).astype(bool) & p0.notna() & (p0 > 1.0) & mom12.notna()
        valid = valid.fillna(False)
        if int(valid.sum()) < 10:
            return zero
        score = _zrank(mom12, valid)
    else:
        z12 = _zrank(mom12, valid)
        z6 = _zrank(mom6, valid)
        zv = _zrank(vol, valid)
        score = 0.7 * z12 + 0.3 * z6 - 0.2 * zv
    score = score.replace([np.inf, -np.inf], np.nan)
    score[~valid] = np.nan
    score = score.dropna()
    if len(score) < 10:
        return zero
    n_side = min(50, len(score) // 2)
    if n_side < 5:
        return zero
    w_side = min(0.02, 0.5 / float(n_side))
    ranked = score.sort_values(kind='mergesort')
    shorts = ranked.index[:n_side]
    longs = ranked.index[-n_side:]
    w = zero.copy()
    w.loc[longs] = w_side
    w.loc[shorts] = -w_side
    out = w.reindex(u).fillna(0.0).astype(float)
    _CACHE[key] = out
    return out