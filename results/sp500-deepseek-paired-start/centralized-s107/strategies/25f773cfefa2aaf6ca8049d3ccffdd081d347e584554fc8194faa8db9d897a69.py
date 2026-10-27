import numpy as np
import pandas as pd


def generate_weights(history, universe):
    idx = pd.Index(universe)
    if len(idx) == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=idx, dtype=float)

    ac = history.get('adjusted_close')
    if ac is None or ac.empty:
        ac = history.get('close')
    if ac is None or ac.empty:
        return zero

    n = len(ac.index)
    period = 15
    if n < 80:
        return zero

    reb = ((n - 1) // period) * period
    if reb < 60:
        return zero

    cols = [c for c in idx if c in ac.columns]
    if len(cols) == 0:
        return zero

    lookback = 90
    start = max(0, reb - lookback + 1)
    close = ac.iloc[start:reb + 1][cols].astype(float).copy()
    close = close.replace([np.inf, -np.inf], np.nan)
    if close.shape[0] < 70:
        return zero

    last = close.iloc[-1]
    valid = last.notna() & (close.notna().sum(axis=0) >= 60)

    ret = close / close.shift(1) - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)
    vol = ret.iloc[-60:].std(axis=0, ddof=1)
    vol = vol.where(vol > 1e-6, np.nan)

    horizons = [5, 10, 21]
    scores = []
    for h in horizons:
        if close.shape[0] <= h:
            continue
        r = close.iloc[-1] / close.iloc[-1 - h] - 1.0
        r = r.replace([np.inf, -np.inf], np.nan)
        rv = r.where(valid & r.notna())
        if rv.notna().sum() < 10:
            continue
        mu = rv.mean(skipna=True)
        resid = r - mu
        scores.append(-resid / (vol + 1e-8))

    if not scores:
        return zero

    score = pd.concat(scores, axis=1).mean(axis=1, skipna=True)
    score = score.replace([np.inf, -np.inf], np.nan)

    vol_df = history.get('volume')
    if vol_df is not None and not vol_df.empty:
        vcols = [c for c in cols if c in vol_df.columns]
        if len(vcols) > 0:
            vsub = vol_df.iloc[start:reb + 1][vcols].astype(float)
            csub = close[vcols]
            dollar = vsub * csub
            liq = dollar.iloc[-60:].median(axis=0)
            liq = liq.replace([np.inf, -np.inf], np.nan).reindex(cols)
            valid = valid & (liq > 0)
            if liq.notna().sum() >= 20:
                thresh = liq.quantile(0.30)
                valid = valid & (liq >= thresh)

    elig_df = history.get('eligible')
    if elig_df is not None and not elig_df.empty:
        try:
            row = elig_df.iloc[reb] if reb < len(elig_df) else elig_df.iloc[-1]
            row = row.reindex(cols).fillna(False).astype(bool)
            valid = valid & row
        except Exception:
            pass

    valid = valid & score.notna()
    candidates = [c for c in cols if bool(valid.get(c, False))]
    if len(candidates) < 20:
        return zero

    sc = score.loc[candidates].sort_values(ascending=False)
    k = min(50, len(candidates) // 2)
    if k < 10:
        return zero

    longs = sc.index[:k].tolist()
    shorts = sc.index[-k:].tolist()
    wmag = min(0.02, 0.9 / (2.0 * k))

    w = zero.copy()
    w.loc[longs] = wmag
    w.loc[shorts] = -wmag

    s = float(w.sum())
    if abs(s) > 1e-12:
        if s > 0:
            ln = [c for c in longs if c in w.index and w.loc[c] > 0.0]
            if ln:
                adj = s / len(ln)
                for c in ln:
                    w.loc[c] = max(0.0, w.loc[c] - adj)
        else:
            sh = [c for c in shorts if c in w.index and w.loc[c] < 0.0]
            if sh:
                adj = (-s) / len(sh)
                for c in sh:
                    w.loc[c] = min(0.0, w.loc[c] + adj)

    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
    gross = float(w.abs().sum())
    if gross > 1.0:
        w = w * (1.0 / gross)
    return w
