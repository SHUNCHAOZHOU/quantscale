import numpy as np
import pandas as pd


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe)
    if len(universe) == 0:
        return zero

    adj = history.get('adjusted_close')
    if adj is None or not isinstance(adj, pd.DataFrame) or adj.shape[0] == 0:
        adj = history.get('close')
    if adj is None or not isinstance(adj, pd.DataFrame) or adj.shape[0] == 0:
        return zero

    if not adj.index.is_monotonic_increasing:
        adj = adj.sort_index()

    cols = [c for c in universe if c in adj.columns]
    if len(cols) == 0:
        return zero

    n_full = len(adj)
    if n_full < 70:
        return zero

    # Rebalance every 21 trading sessions. Between rebalances, use the
    # signal computed at the most recent rebalance close.
    reb = ((n_full - 1) // 21) * 21
    px = adj.iloc[:reb + 1][cols].tail(300)
    n = len(px)
    if n < 64:
        return zero

    last = px.iloc[-1]
    valid = last.notna() & (last > 0)
    valid = valid & (px.notna().sum() >= 60)

    edf = history.get('eligible')
    if isinstance(edf, pd.DataFrame) and len(edf) > 0:
        if not edf.index.is_monotonic_increasing:
            edf = edf.sort_index()
        ecols = [c for c in cols if c in edf.columns]
        if ecols:
            e_tail = edf[ecols].iloc[:reb + 1].tail(1)
            if len(e_tail) > 0:
                e = e_tail.iloc[-1].reindex(cols).fillna(False).astype(bool)
                valid = valid & e

    if int(valid.sum()) < 20:
        return zero

    rets = px.pct_change()
    vol = rets.tail(63).std() * np.sqrt(252.0)
    vol = vol.replace([np.inf, -np.inf], np.nan)

    def cum_ret(days):
        if n <= days:
            return pd.Series(np.nan, index=cols)
        return px.iloc[-1] / px.iloc[n - 1 - days] - 1.0

    r21 = cum_ret(21)
    r63 = cum_ret(63)

    dvol = None
    vdf = history.get('volume')
    cdf = history.get('close')
    if (isinstance(vdf, pd.DataFrame) and isinstance(cdf, pd.DataFrame)
            and len(vdf) > 0 and len(cdf) > 0):
        if not vdf.index.is_monotonic_increasing:
            vdf = vdf.sort_index()
        if not cdf.index.is_monotonic_increasing:
            cdf = cdf.sort_index()
        vcols = [c for c in cols if c in vdf.columns and c in cdf.columns]
        if vcols:
            vs = vdf[vcols].iloc[:reb + 1].tail(21)
            ps = cdf[vcols].iloc[:reb + 1].tail(21)
            dvol = (vs * ps).median().reindex(cols)
            dvol = dvol.replace([np.inf, -np.inf], np.nan)

    idx = valid[valid].index
    r21_v = r21.reindex(idx)
    ok = r21_v.notna()
    idx = idx[ok.values]
    if len(idx) < 20:
        return zero

    r21_v = r21_v.loc[idx]
    r63_v = r63.reindex(idx).fillna(r21_v)
    vol_v = vol.reindex(idx)
    if vol_v.notna().any():
        vol_v = vol_v.fillna(vol_v.median())
    else:
        vol_v = pd.Series(1.0, index=idx)

    if dvol is not None:
        dv = dvol.reindex(idx).fillna(0.0)
        if len(dv) >= 5 and dv.notna().any():
            keep_liq = dv >= dv.quantile(0.20)
        else:
            keep_liq = pd.Series(True, index=idx)
    else:
        keep_liq = pd.Series(True, index=idx)

    if vol_v.notna().any():
        keep_vol = vol_v <= vol_v.quantile(0.90)
    else:
        keep_vol = pd.Series(True, index=idx)

    keep = keep_liq & keep_vol
    idx_final = idx[keep.values]
    if len(idx_final) < 20:
        return zero

    r21_f = r21_v.loc[idx_final]
    r63_f = r63_v.loc[idx_final]
    vol_f = vol_v.loc[idx_final]

    def rp(s):
        return s.rank(method='average', pct=True)

    # Reversal: recent losers get high scores, recent winners get low scores.
    score = -(0.65 * rp(r21_f) + 0.35 * rp(r63_f))
    # Mild low-volatility tilt on both sides.
    score = score + 0.05 * (1.0 - rp(vol_f))
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 20:
        return zero

    n_side = min(40, len(score) // 2)
    if n_side < 1:
        return zero

    longs = score.nlargest(n_side).index
    shorts = score.nsmallest(n_side).index

    weights = pd.Series(0.0, index=universe)
    weights.loc[longs] = 1.0
    weights.loc[shorts] = -1.0
    weights = weights.reindex(universe).fillna(0.0)

    nL = int((weights > 0).sum())
    nS = int((weights < 0).sum())
    if nL == 0 or nS == 0:
        return zero

    # Equal-weight each side, with equal gross on both sides.
    target = min(0.5, 0.02 * min(nL, nS))
    if target <= 0.0:
        return zero
    weights.loc[weights > 0] = target / float(nL)
    weights.loc[weights < 0] = -target / float(nS)

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    weights = weights.clip(-0.02, 0.02)

    gross = weights.abs().sum()
    if gross > 1.0 + 1e-12:
        weights = weights * (1.0 / gross)

    return weights.reindex(universe).fillna(0.0)