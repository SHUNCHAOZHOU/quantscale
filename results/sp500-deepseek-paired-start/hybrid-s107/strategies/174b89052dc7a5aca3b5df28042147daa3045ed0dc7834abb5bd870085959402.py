import numpy as np
import pandas as pd

def generate_weights(history, universe):
    idx = pd.Index(universe)
    if len(idx) == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=idx, dtype=float)

    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None:
        return zero

    n = len(ac.index)
    if n < 30:
        return zero

    reb = ((n - 1) // 5) * 5
    if reb < 25:
        return zero

    cols = [c for c in idx if c in ac.columns]
    if len(cols) == 0:
        return zero

    start = max(0, reb - 44)
    close = ac.iloc[start:reb + 1][cols].astype(float).copy()
    close = close.replace([np.inf, -np.inf], np.nan)
    if close.shape[0] < 25:
        return zero

    last = close.iloc[-1]
    valid = last.notna() & (close.notna().sum(axis=0) >= 25)

    ret = close / close.shift(1) - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)
    vol = ret.iloc[-20:].std(axis=0, ddof=1)
    vol = vol.where(vol > 0, np.nan)

    r5 = close.iloc[-1] / close.iloc[-6] - 1.0
    eps = 1e-8
    score = -r5 / (vol + eps)

    vol_df = history.get('volume')
    if vol_df is not None:
        vcols = [c for c in cols if c in vol_df.columns]
        if len(vcols) > 0:
            vsub = vol_df.iloc[start:reb + 1][vcols].astype(float)
            csub = close[vcols]
            dollar = vsub * csub
            liq = dollar.iloc[-20:].median(axis=0)
            liq = liq.replace([np.inf, -np.inf], np.nan)
            liq = liq.reindex(cols)
            valid = valid & (liq > 0)
            if liq.notna().sum() >= 20:
                thresh = liq.quantile(0.20)
                valid = valid & (liq >= thresh)

    valid = valid & score.notna()
    candidates = [c for c in cols if bool(valid.get(c, False))]
    if len(candidates) < 4:
        return zero

    sc = score.loc[candidates].sort_values(ascending=False)
    k = min(30, len(candidates) // 2)
    if k < 1:
        return zero

    longs = sc.index[:k].tolist()
    shorts = sc.index[-k:].tolist()
    wmag = min(0.5 / k, 0.02)

    w = zero.copy()
    w.loc[longs] = wmag
    w.loc[shorts] = -wmag
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return w