import numpy as np
import pandas as pd


def _allocate(names, vols, target, maxw):
    if not names or target <= 0:
        return {}
    v = vols.reindex(names).astype(float)
    med = v.median()
    if not np.isfinite(med) or med <= 0:
        med = 0.02
    v = v.where(np.isfinite(v) & (v > 0), med)
    raw = 1.0 / v
    w = pd.Series(0.0, index=names, dtype=float)
    remaining = list(names)
    rem_target = float(target)
    for _ in range(20):
        if not remaining or rem_target <= 1e-12:
            break
        r = raw.loc[remaining]
        s = float(r.sum())
        if not np.isfinite(s) or s <= 0:
            break
        alloc = r / s * rem_target
        over = alloc[alloc > maxw + 1e-15]
        if len(over) == 0:
            w.loc[remaining] = alloc
            rem_target = 0.0
            break
        for nm in list(over.index):
            w.loc[nm] = maxw
            rem_target -= maxw
            if nm in remaining:
                remaining.remove(nm)
        if rem_target <= 1e-12:
            break
    if remaining and rem_target > 1e-12:
        w.loc[remaining] += rem_target / len(remaining)
    return w


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
    if n < 300:
        return zero

    reb = ((n - 1) // 21) * 21
    if reb < 280:
        return zero

    cols = [c for c in idx if c in ac.columns]
    if len(cols) < 20:
        return zero

    start = max(0, reb - 320)
    px = ac.iloc[start:reb + 1][cols].astype(float).copy()
    px = px.replace([np.inf, -np.inf], np.nan)
    if px.shape[0] < 300:
        return zero

    rets = px / px.shift(1) - 1.0
    rets = rets.replace([np.inf, -np.inf], np.nan)

    mkt = rets.mean(axis=1, skipna=True)
    resid = rets.sub(mkt, axis=0)
    resid = resid.replace([np.inf, -np.inf], np.nan)

    npx = resid.shape[0]
    skip = 21
    end = npx - skip
    if end < 252:
        return zero

    long_start = end - 252
    short_start = end - 126

    long_sum = resid.iloc[long_start:end].sum(axis=0, skipna=True)
    short_sum = resid.iloc[short_start:end].sum(axis=0, skipna=True)

    rv = resid.iloc[short_start:end].std(axis=0, ddof=1)
    rv = rv.replace([0.0, np.inf, -np.inf], np.nan)

    cov = resid.iloc[long_start:end].notna().sum(axis=0)
    last = px.iloc[-1]

    valid = (cov >= 220) & long_sum.notna() & short_sum.notna()
    valid = valid & rv.notna() & (rv > 0) & last.notna()

    score_long = long_sum / (rv * np.sqrt(252.0))
    score_short = short_sum / (rv * np.sqrt(126.0))
    score = 0.5 * score_long + 0.5 * score_short

    valid = valid & (rv > 0.002) & (rv < 0.10)

    elig = history.get('eligible')
    if elig is not None:
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

    valid = valid & score.notna() & np.isfinite(score)
    cand = score[valid].sort_values(ascending=False)
    if len(cand) < 15:
        return zero

    k = min(40, max(15, len(cand) // 8))
    k = min(k, len(cand) // 2)
    if k < 10:
        return zero

    longs = cand.index[:k].tolist()
    shorts = cand.index[-k:].tolist()

    rv_all = rv.reindex(cols)
    target = min(0.5, 0.02 * k)
    wl = _allocate(longs, rv_all, target, 0.02)
    ws = _allocate(shorts, rv_all, target, 0.02)

    w = zero.copy()
    for nm, val in wl.items():
        w.loc[nm] = val
    for nm, val in ws.items():
        w.loc[nm] = -val

    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return w