import numpy as np
import pandas as pd

MAX_W = 0.02
N_SIDE = 30


def _rank_pct(s, mask):
    return s.where(mask).rank(pct=True)


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None:
        return zero
    ac = ac.reindex(columns=universe).astype(float)
    n = len(ac)
    if n < 150:
        return zero

    vol = history.get('volume')
    if vol is not None:
        vol = vol.reindex(columns=universe).astype(float)

    ret = ac.pct_change()
    vol60 = ret.rolling(60, min_periods=30).std()

    last = n - 1
    skip = 21
    i_end = last - skip
    if i_end < 60:
        return zero

    i_start12 = max(0, i_end - 231)
    mom12 = ac.iloc[i_end].div(ac.iloc[i_start12]) - 1.0
    i_start6 = max(0, i_end - 126)
    mom6 = ac.iloc[i_end].div(ac.iloc[i_start6]) - 1.0
    if n < 280:
        mom12 = mom6.copy()

    r21 = ac.iloc[last].div(ac.iloc[max(0, last - 21)]) - 1.0

    dv20 = None
    if vol is not None:
        dv20 = (ac * vol).rolling(20, min_periods=5).mean().iloc[last]

    price = ac.iloc[last]
    v = vol60.iloc[last]
    mask = (price.notna() & (price > 5.0) &
            v.notna() & np.isfinite(v) & (v > 0) &
            mom12.notna() & mom6.notna() &
            np.isfinite(mom12) & np.isfinite(mom6))

    if dv20 is not None:
        mask = mask & dv20.notna() & (dv20 > 0)
        if int(mask.sum()) > 50:
            thr = dv20.where(mask).quantile(0.20)
            if np.isfinite(thr):
                mask = mask & (dv20 >= thr)

    elig = history.get('eligible')
    if elig is not None:
        e = elig.reindex(columns=universe)
        if len(e) > 0:
            mask = mask & e.iloc[-1].fillna(False).astype(bool)

    vv = v.where(mask)
    if int(vv.notna().sum()) >= 20:
        cap = vv.quantile(0.80)
        if np.isfinite(cap):
            mask = mask & (v <= cap)

    if int(mask.sum()) < 4:
        return zero

    r12 = _rank_pct(mom12, mask)
    r6 = _rank_pct(mom6, mask)
    rv = _rank_pct(v, mask)

    score = -0.60 * r12 - 0.30 * r6 - 0.10 * rv
    if dv20 is not None:
        rliq = _rank_pct(dv20, mask)
        score = score + 0.05 * rliq
    score = score.where(mask)

    s = score.dropna()
    if len(s) < 4:
        return zero

    order = s.sort_values(ascending=False)
    n_side = min(N_SIDE, len(order) // 2)
    if n_side < 1:
        return zero

    longs = list(order.index[:n_side])
    shorts = list(order.index[-n_side:])

    parabolic = set(r21.index[(r21 > 1.0).fillna(False)])
    shorts = [t for t in shorts if t not in parabolic]
    if len(shorts) < n_side:
        candidates = [t for t in order.index[::-1]
                      if t not in parabolic and t not in shorts]
        shorts.extend(candidates[:n_side - len(shorts)])

    shorts = [t for t in shorts if t not in longs]
    n_side = min(len(longs), len(shorts))
    if n_side < 1:
        return zero
    longs = longs[:n_side]
    shorts = shorts[:n_side]

    w = min(MAX_W, 0.5 / n_side)
    weights = pd.Series(0.0, index=universe, dtype=float)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.clip(-MAX_W, MAX_W)
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights
