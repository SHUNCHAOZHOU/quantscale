import numpy as np
import pandas as pd

MAX_W = 0.02
N_SIDE = 25
LAGS = (0, 3, 6, 9)
LAG_W = (0.40, 0.30, 0.20, 0.10)


def _rank_pct(s, mask):
    return s.where(mask).rank(pct=True)


def _score_at(ac, resid, vol60_full, end):
    """Composite reversal score using data strictly up to row `end` (inclusive)."""
    if end is None or end < 30:
        return None
    # recent idiosyncratic return, skipping the most recent session
    res5 = resid.iloc[max(0, end - 5):end].sum()
    res21 = resid.iloc[max(0, end - 21):end].sum()
    i0 = max(0, end - 21)
    i12 = max(0, end - 252)
    p0 = ac.iloc[i0]
    p12 = ac.iloc[i12]
    mom12 = p0.div(p12) - 1.0
    v60 = vol60_full.iloc[end]
    mask = (res5.notna() & res21.notna() & mom12.notna() &
            v60.notna() & np.isfinite(v60) & (v60 > 0))
    if int(mask.sum()) < 20:
        return None
    score = (-0.45 * _rank_pct(res5, mask)
             - 0.35 * _rank_pct(res21, mask)
             - 0.15 * _rank_pct(mom12, mask))
    return score.where(mask)


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
    T = len(ac)
    if T < 45:
        return zero

    vol = history.get('volume')
    if vol is not None:
        vol = vol.reindex(columns=universe).astype(float)

    ret = ac.pct_change()
    mkt = ret.mean(axis=1)
    resid = ret.sub(mkt, axis=0)
    vol60_full = ret.rolling(60, min_periods=20).std()

    last = T - 1

    # ---- lagged blend of the composite reversal score ----
    blend = pd.Series(0.0, index=universe, dtype=float)
    wsum = 0.0
    for lag, w in zip(LAGS, LAG_W):
        s = _score_at(ac, resid, vol60_full, last - lag)
        if s is None:
            continue
        centered = (s - s.median()).fillna(0.0)
        blend = blend + w * centered
        wsum += w
    if wsum <= 0:
        return zero
    blend = blend / wsum

    # ---- tradability filters evaluated at the decision close ----
    price_last = ac.iloc[last]
    v60_now = vol60_full.iloc[last]
    mask = (price_last.notna() & (price_last > 5.0) &
            v60_now.notna() & np.isfinite(v60_now) & (v60_now > 0))

    dv_last = None
    if vol is not None:
        dv_last = (ac * vol).rolling(20, min_periods=5).mean().iloc[last]
        mask = mask & dv_last.notna() & (dv_last > 0)
        if int(mask.sum()) > 50:
            thr = dv_last.where(mask).quantile(0.20)
            if np.isfinite(thr):
                mask = mask & (dv_last >= thr)

    elig = history.get('eligible')
    if elig is not None:
        e = elig.reindex(columns=universe)
        if len(e) > 0:
            mask = mask & e.iloc[-1].fillna(False).astype(bool)

    # volatility cap: drop the most volatile quintile on both legs
    vv = v60_now.where(mask)
    if int(vv.notna().sum()) >= 20:
        cap = vv.quantile(0.80)
        if np.isfinite(cap):
            mask = mask & (v60_now <= cap)

    # no parabolic names (limits short-leg squeeze risk)
    i21 = max(0, last - 21)
    r21 = price_last.div(ac.iloc[i21]) - 1.0
    mask = mask & r21.notna() & (r21 < 1.0)

    s = blend.where(mask).dropna()
    if len(s) < 4:
        return zero

    order = s.sort_values(ascending=False)
    n_side = min(N_SIDE, len(order) // 2)
    if n_side < 1:
        return zero

    longs = order.index[:n_side]
    shorts = order.index[-n_side:]
    w_side = min(MAX_W, 0.5 / n_side)

    weights = pd.Series(0.0, index=universe, dtype=float)
    weights.loc[longs] = w_side
    weights.loc[shorts] = -w_side
    weights = weights.clip(-MAX_W, MAX_W)
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights
