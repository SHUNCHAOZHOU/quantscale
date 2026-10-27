import numpy as np
import pandas as pd

_CACHE = {}

def generate_weights(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zeros

    close = history.get('adjusted_close')
    if close is None or len(close) == 0:
        close = history.get('close')
    if close is None or len(close) == 0:
        return zeros

    close = close.reindex(columns=universe)
    idx = close.index
    if len(idx) == 0:
        return zeros

    d = idx[-1]
    month_start = pd.Timestamp(year=d.year, month=d.month, day=1)
    prior = idx[idx < month_start]
    if len(prior) == 0:
        r = idx[0]
    else:
        r = prior[-1]

    key = (r, tuple(universe))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached.reindex(universe).fillna(0.0).astype(float)

    close_sig = close.loc[:r]
    if len(close_sig) < 30:
        return zeros

    last_close = close_sig.iloc[-1]

    elig = history.get('eligible')
    if elig is not None:
        elig_sig = elig.reindex(columns=universe).loc[:r]
        if len(elig_sig) > 0:
            e = elig_sig.iloc[-1]
            elig_last = e.fillna(True).astype(bool)
        else:
            elig_last = pd.Series(True, index=universe)
    else:
        elig_last = pd.Series(True, index=universe)

    vol = history.get('volume')
    if vol is not None:
        vol_sig = vol.reindex(columns=universe).loc[:r]
        k = min(60, len(close_sig))
        dollar = (vol_sig.tail(k) * close_sig.tail(k)).median()
    else:
        dollar = pd.Series(1.0, index=universe)

    valid = last_close.notna() & np.isfinite(last_close) & (last_close > 0.5)
    valid = valid & elig_last
    valid = valid & dollar.notna() & (dollar > 0)

    if valid.sum() > 20:
        thresh = dollar[valid].quantile(0.2)
        valid = valid & (dollar >= thresh)

    n = len(close_sig)

    if n >= 253:
        mom = close_sig.iloc[-22] / close_sig.iloc[-253] - 1.0
    elif n >= 127:
        mom = close_sig.iloc[-22] / close_sig.iloc[-127] - 1.0
    elif n >= 64:
        mom = close_sig.iloc[-22] / close_sig.iloc[-64] - 1.0
    else:
        mom = pd.Series(0.0, index=universe)

    if n >= 22:
        ret21 = close_sig.iloc[-1] / close_sig.iloc[-22] - 1.0
    elif n >= 11:
        ret21 = close_sig.iloc[-1] / close_sig.iloc[-11] - 1.0
    else:
        ret21 = pd.Series(0.0, index=universe)

    rets = close_sig.pct_change()
    vol60 = rets.tail(60).std()

    cand = valid & last_close.notna()
    if cand.sum() < 2:
        return zeros

    sub = pd.DataFrame({
        'mom': mom[cand],
        'rev': (-ret21)[cand],
        'vol': vol60[cand],
        'liq': dollar[cand]
    })

    for col in ['mom', 'rev', 'vol', 'liq']:
        med = sub[col].median()
        if not np.isfinite(med):
            med = 0.0
        sub[col] = sub[col].fillna(med)

    r_mom = sub['mom'].rank(pct=True, method='average')
    r_rev = sub['rev'].rank(pct=True, method='average')
    r_vol = (-sub['vol']).rank(pct=True, method='average')
    r_liq = sub['liq'].rank(pct=True, method='average')

    sub['score'] = (
        0.50 * (r_mom - 0.5)
        + 0.35 * (r_rev - 0.5)
        + 0.10 * (r_vol - 0.5)
        + 0.05 * (r_liq - 0.5)
    )

    n_cand = len(sub)
    N = min(100, n_cand // 2)
    if N < 1:
        return zeros

    ranked = sub.sort_values('score', ascending=False)
    long_names = ranked.index[:N]
    short_names = ranked.index[-N:]

    side_w = min(0.5 / float(N), 0.02)
    w = pd.Series(0.0, index=universe, dtype=float)
    w.loc[long_names] = side_w
    w.loc[short_names] = -side_w

    _CACHE[key] = w.copy()
    return w
