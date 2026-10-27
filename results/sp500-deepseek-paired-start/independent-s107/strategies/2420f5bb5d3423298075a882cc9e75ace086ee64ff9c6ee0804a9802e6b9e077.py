import numpy as np
import pandas as pd


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe)
    if len(universe) == 0:
        return zero

    adj = history.get('adjusted_close')
    if adj is None:
        adj = history.get('close')
    if adj is None or not isinstance(adj, pd.DataFrame) or adj.shape[0] == 0:
        return zero

    cols = [c for c in universe if c in adj.columns]
    if len(cols) == 0:
        return zero

    if not adj.index.is_monotonic_increasing:
        adj = adj.sort_index()
    px = adj[cols].tail(320)
    n = len(px)
    if n < 70:
        return zero

    last = px.iloc[-1]
    valid = last.notna() & (last > 0)
    valid = valid & (px.notna().sum() >= 70)

    if 'eligible' in history:
        edf = history['eligible']
        if isinstance(edf, pd.DataFrame) and len(edf) > 0:
            if not edf.index.is_monotonic_increasing:
                edf = edf.sort_index()
            ecols = [c for c in cols if c in edf.columns]
            if ecols:
                e = edf[ecols].tail(1).iloc[-1].reindex(cols).fillna(False).astype(bool)
                valid = valid & e

    if valid.sum() < 20:
        return zero

    rets = px.pct_change()

    def cum_ret(days):
        if n <= days:
            return pd.Series(np.nan, index=cols)
        return px.iloc[-1] / px.iloc[n - 1 - days] - 1.0

    r21 = cum_ret(21)
    r63 = cum_ret(63)

    vol = rets.tail(63).std() * np.sqrt(252.0)
    vol = vol.replace([np.inf, -np.inf], np.nan)

    dollar_vol = None
    vdf = history.get('volume')
    if vdf is not None and isinstance(vdf, pd.DataFrame) and len(vdf) > 0:
        if not vdf.index.is_monotonic_increasing:
            vdf = vdf.sort_index()
        vcols = [c for c in cols if c in vdf.columns]
        if vcols:
            vs = vdf[vcols].tail(21)
            ps = px[vcols].tail(21)
            dollar_vol = (vs * ps).median().reindex(cols)

    idx_valid = valid[valid].index
    if len(idx_valid) < 20:
        return zero

    def rp(s):
        return s.rank(method='average', pct=True)

    r21_v = r21.reindex(idx_valid)
    r63_v = r63.reindex(idx_valid)
    vol_v = vol.reindex(idx_valid)

    ok = r21_v.notna()
    idx_valid = idx_valid[ok]
    if len(idx_valid) < 20:
        return zero
    r21_v = r21_v.loc[idx_valid]
    r63_v = r63_v.loc[idx_valid].fillna(r21_v)
    vol_v = vol_v.loc[idx_valid]

    if dollar_vol is not None:
        dvol_v = dollar_vol.reindex(idx_valid).fillna(0.0)
        keep_liq = dvol_v >= dvol_v.quantile(0.20)
    else:
        keep_liq = pd.Series(True, index=idx_valid)

    if vol_v.notna().any():
        vol_filled = vol_v.fillna(vol_v.median())
        keep_vol = vol_filled <= vol_filled.quantile(0.90)
    else:
        keep_vol = pd.Series(True, index=idx_valid)

    keep = keep_liq & keep_vol
    idx_final = idx_valid[keep]
    if len(idx_final) < 20:
        return zero

    r21_f = r21_v.loc[idx_final]
    r63_f = r63_v.loc[idx_final]

    score = -(0.5 * rp(r21_f) + 0.5 * rp(r63_f))
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 10:
        return zero

    n_side = min(50, len(score) // 2)
    if n_side < 1:
        return zero

    longs = score.nlargest(n_side).index
    shorts = score.nsmallest(n_side).index

    w = min(0.02, 0.5 / float(n_side))
    weights = pd.Series(0.0, index=universe)
    weights.loc[longs] = w
    weights.loc[shorts] = -w

    gross = weights.abs().sum()
    if gross > 1.0 + 1e-12:
        weights = weights * (1.0 / gross)
    weights = weights.clip(-0.02, 0.02)
    return weights.reindex(universe).fillna(0.0)
