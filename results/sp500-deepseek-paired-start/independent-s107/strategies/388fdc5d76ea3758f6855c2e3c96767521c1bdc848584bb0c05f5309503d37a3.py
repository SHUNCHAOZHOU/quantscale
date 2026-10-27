import numpy as np
import pandas as pd


def _momentum(px, lookback, skip):
    n = px.shape[0]
    if n < lookback + skip + 1:
        return pd.Series(np.nan, index=px.columns)
    p_end = px.iloc[n - 1 - skip]
    p_start = px.iloc[n - 1 - skip - lookback]
    return p_end / p_start - 1.0


def generate_weights(history, universe):
    idx = pd.Index(universe)
    zero = pd.Series(0.0, index=idx, dtype=float)
    if len(idx) == 0:
        return zero

    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None:
        return zero

    n = len(ac.index)
    if n < 60:
        return zero

    # Rebalance every 21 trading sessions; on non-rebalance days the same
    # historical block is used, so target weights are unchanged.
    reb = ((n - 1) // 21) * 21
    if reb < 50:
        return zero

    cols = [c for c in idx if c in ac.columns]
    if len(cols) == 0:
        return zero

    lookback = 300
    start = max(0, reb - lookback)
    px = ac.iloc[start:reb + 1][cols].astype(float).copy()
    px = px.replace([np.inf, -np.inf], np.nan)
    if px.shape[0] < 50:
        return zero

    mom6 = _momentum(px, 126, 21)
    mom12 = _momentum(px, 252, 21)
    if mom6.notna().sum() == 0:
        return zero

    score = mom6.copy()
    mask12 = mom12.notna()
    score[mask12] = 0.5 * mom6[mask12] + 0.5 * mom12[mask12]

    rets = px.pct_change()
    if px.shape[0] >= 126:
        vol = rets.iloc[-126:].std(axis=0, ddof=1)
    else:
        vol = rets.std(axis=0, ddof=1)
    vol = vol.replace([0.0, np.inf, -np.inf], np.nan)

    win = min(148, px.shape[0])
    coverage = px.iloc[-win:].notna().sum(axis=0)
    valid = (coverage >= int(win * 0.95)) & score.notna() & vol.notna()
    valid = valid & (vol > 0.002) & (vol < 0.12)

    elig = history.get('eligible')
    if elig is not None and reb < len(elig.index):
        try:
            erow = elig.iloc[reb].reindex(cols)
            if erow.notna().any():
                valid = valid & erow.fillna(False).astype(bool)
        except Exception:
            pass

    volu = history.get('volume')
    if volu is not None:
        vcols = [c for c in cols if c in volu.columns]
        if len(vcols) > 0:
            vsub = volu.iloc[start:reb + 1][vcols].astype(float)
            csub = px[vcols]
            dollar = vsub.mul(csub, axis=0)
            liq = dollar.iloc[-60:].median(axis=0).reindex(cols)
            liq = liq.replace([np.inf, -np.inf], np.nan)
            valid = valid & (liq > 0)
            if liq.notna().sum() >= 20:
                valid = valid & (liq >= liq.quantile(0.20))

    valid = valid & score.notna()
    cand = score[valid].sort_values(ascending=False)
    if len(cand) < 10:
        return zero

    k = min(25, max(10, len(cand) // 10))
    if k < 5:
        return zero

    longs = cand.index[:k].tolist()
    shorts = cand.index[-k:].tolist()
    wmag = min(0.02, 0.5 / k)

    w = zero.copy()
    w.loc[longs] = wmag
    w.loc[shorts] = -wmag
    return w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
