import numpy as np
import pandas as pd

_CACHE = {}

def _zero(universe):
    return pd.Series(0.0, index=universe, dtype=float)

def _robust_z(s):
    s = pd.Series(s, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return pd.Series(dtype=float)
    med = s.median()
    mad = (s - med).abs().median()
    if (not np.isfinite(mad)) or mad < 1e-12:
        sd = s.std(ddof=0)
        if (not np.isfinite(sd)) or sd < 1e-12:
            return pd.Series(0.0, index=s.index)
        z = (s - med) / sd
    else:
        z = (s - med) / (1.4826 * mad)
    return z.clip(-3.0, 3.0)

def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)

    price_key = 'adjusted_close' if 'adjusted_close' in history else 'close'
    px_all = history[price_key]
    if px_all is None or px_all.empty:
        return _zero(universe)

    idx = px_all.index
    if len(idx) == 0:
        return _zero(universe)

    months = idx.month.values
    years = idx.year.values
    is_first = np.empty(len(idx), dtype=bool)
    is_first[0] = True
    if len(idx) > 1:
        is_first[1:] = (months[1:] != months[:-1]) | (years[1:] != years[:-1])
    pos = np.where(is_first)[0][-1]
    reb_date = idx[pos]

    cache_key = (reb_date, tuple(universe))
    if cache_key in _CACHE:
        return _CACHE[cache_key].copy()

    px = px_all.loc[:reb_date].reindex(columns=universe).astype(float)
    n = len(px)
    if n < 70:
        w = _zero(universe)
        _CACHE[cache_key] = w
        return w.copy()

    last = px.iloc[-1]
    rets = px.pct_change()

    vol_window = min(126, n - 1)
    if vol_window < 20:
        w = _zero(universe)
        _CACHE[cache_key] = w
        return w.copy()

    vol = rets.iloc[-vol_window:].std(ddof=0)
    valid = last.notna() & (last > 0) & vol.notna() & (vol > 1e-8)

    if n >= 22:
        r21 = last / px.iloc[-22] - 1.0
    else:
        r21 = pd.Series(np.nan, index=universe)
    if n >= 6:
        r5 = last / px.iloc[-6] - 1.0
    else:
        r5 = pd.Series(np.nan, index=universe)
    rev = -0.7 * r21 - 0.3 * r5

    if n >= 253:
        mom = px.iloc[-22] / px.iloc[-253] - 1.0
    elif n >= 130:
        mom = px.iloc[-22] / px.iloc[-130] - 1.0
    else:
        mom = pd.Series(np.nan, index=universe)

    hi_window = min(252, n)
    hi = px.iloc[-hi_window:].max()
    prox = last / hi - 1.0
    lowvol = -vol

    mkt = rets.mean(axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(mkt) > 0:
        m = mkt.iloc[-200:] if len(mkt) >= 200 else mkt
        trend = float(np.prod(1.0 + m.values) - 1.0)
    else:
        trend = 0.0
    up = trend > 0.0

    valid_idx = valid[valid].index
    if len(valid_idx) < 4:
        w = _zero(universe)
        _CACHE[cache_key] = w
        return w.copy()

    def zsafe(s):
        z = _robust_z(s.reindex(valid_idx))
        return z.reindex(valid_idx).fillna(0.0)

    z_rev = zsafe(rev)
    z_mom = zsafe(mom)
    z_prox = zsafe(prox)
    z_lowvol = zsafe(lowvol)

    if up:
        score = 0.40 * z_prox + 0.30 * z_mom + 0.15 * z_rev + 0.15 * z_lowvol
    else:
        score = 0.50 * z_rev + 0.30 * z_lowvol + 0.10 * z_prox + 0.10 * z_mom

    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 4:
        w = _zero(universe)
        _CACHE[cache_key] = w
        return w.copy()

    k = min(25, len(score) // 2)
    if k < 1:
        w = _zero(universe)
        _CACHE[cache_key] = w
        return w.copy()

    score = score.sort_values(ascending=False)
    longs = score.index[:k]
    shorts = score.index[-k:]

    gross_scale = 1.0 if up else 0.7
    wt = 0.02 * gross_scale
    w = _zero(universe)
    w.loc[longs] = wt
    w.loc[shorts] = -wt

    _CACHE[cache_key] = w
    return w.copy()
