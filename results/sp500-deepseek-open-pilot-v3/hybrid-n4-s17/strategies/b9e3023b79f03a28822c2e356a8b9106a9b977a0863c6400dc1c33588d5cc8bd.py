import numpy as np
import pandas as pd


def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)

    ac = history.get('adjusted_close')
    if ac is None or ac.empty:
        ac = history.get('close')
    if ac is None or ac.empty:
        return pd.Series(0.0, index=universe)

    vol = history.get('volume')
    elig = history.get('eligible')

    cols = pd.Index([c for c in universe if c in ac.columns])
    if len(cols) == 0:
        return pd.Series(0.0, index=universe)

    n = len(ac.index)
    if n < 4:
        return pd.Series(0.0, index=universe)

    max_lb = 10
    vol_lb = 20
    start = max(0, n - max(max_lb + 1, vol_lb + 1))
    ac_sub = ac.iloc[start:n][cols].copy()
    vol_sub = vol.iloc[start:n][cols].copy() if (vol is not None and not vol.empty) else None
    elig_sub = elig.iloc[start:n][cols].copy() if (elig is not None and not elig.empty) else None

    if len(ac_sub) < 4:
        return pd.Series(0.0, index=universe)

    last_ac = ac_sub.iloc[-1]

    rets = []
    for lb, wt in ((3, 0.5), (10, 0.5)):
        if len(ac_sub) > lb:
            r = ac_sub.iloc[-1] / ac_sub.iloc[-1 - lb] - 1.0
            rets.append(wt * r)

    if not rets:
        return pd.Series(0.0, index=universe)

    denom = sum(wt for lb, wt in ((3, 0.5), (10, 0.5)) if len(ac_sub) > lb)
    comp_ret = sum(rets) / denom
    comp_ret = comp_ret.replace([np.inf, -np.inf], np.nan)

    vol_spike = pd.Series(np.nan, index=cols)
    if vol_sub is not None and len(vol_sub) >= 2:
        prior = vol_sub.iloc[:-1].tail(vol_lb)
        med = prior.median(axis=0, skipna=True)
        lastv = vol_sub.iloc[-1]
        vol_spike = lastv / med.where(med > 0, np.nan)
        vol_spike = vol_spike.replace([np.inf, -np.inf], np.nan)

    mask = comp_ret.notna() & np.isfinite(comp_ret) & (last_ac > 0)
    if elig_sub is not None and len(elig_sub) > 0:
        last_elig = elig_sub.iloc[-1].fillna(False).astype(bool)
        mask = mask & last_elig
    mask = mask.fillna(False)

    valid = mask[mask].index.tolist()
    if len(valid) < 10:
        return pd.Series(0.0, index=universe)

    sig = (-comp_ret.loc[valid]).astype(float)
    vs = vol_spike.reindex(valid)
    vr = vs.rank(method='average', pct=True).fillna(0.5)
    sig_adj = sig * (1.0 + 0.5 * vr)

    ranks = sig_adj.rank(method='average', pct=True)
    centered = ranks - ranks.mean()

    n_side = min(25, max(5, len(valid) // 10))
    n_side = min(n_side, len(valid) // 2)
    if n_side < 1:
        return pd.Series(0.0, index=universe)

    w = min(0.02, 0.5 / n_side)
    long_names = centered.nlargest(n_side).index
    short_names = centered.nsmallest(n_side).index

    weights = pd.Series(0.0, index=universe)
    for t in long_names:
        weights.loc[t] = w
    for t in short_names:
        weights.loc[t] = -w

    weights = weights.clip(-0.02, 0.02)
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights
