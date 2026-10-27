import numpy as np
import pandas as pd


def _rank_pct(values, valid):
    out = np.full(values.shape, np.nan, dtype=float)
    idx = np.where(valid & np.isfinite(values))[0]
    if idx.size == 0:
        return out
    s = pd.Series(values[idx])
    out[idx] = s.rank(method='average', pct=True).to_numpy(dtype=float)
    return out


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    ac = history.get('adjusted_close')
    if not isinstance(ac, pd.DataFrame) or ac.empty:
        ac = history.get('close')
    if not isinstance(ac, pd.DataFrame) or ac.empty:
        return zero

    cols = [c for c in universe if c in ac.columns]
    if len(cols) < 20:
        return zero

    n_rows = len(ac)
    if n_rows < 60:
        return zero

    rebalance = 10
    anchor = ((n_rows - 1) // rebalance) * rebalance
    if anchor < 40:
        return zero

    start = max(0, anchor - 130)
    px_df = ac.iloc[start:anchor + 1][cols]
    prices = px_df.to_numpy(dtype=float, copy=True)
    if prices.shape[0] < 30:
        return zero

    last_elig = np.ones(len(cols), dtype=bool)
    elig_df = history.get('eligible')
    if isinstance(elig_df, pd.DataFrame) and not elig_df.empty:
        e = elig_df.reindex(index=ac.index, columns=cols)
        if anchor < len(e):
            last_elig = e.iloc[anchor].fillna(False).astype(bool).to_numpy(dtype=bool)

    valid = np.isfinite(prices[-1]) & (prices[-1] > 1.0) & last_elig

    vol_df = history.get('volume')
    if isinstance(vol_df, pd.DataFrame) and not vol_df.empty:
        vdf = vol_df.reindex(index=ac.index, columns=cols)
        v = vdf.iloc[start:anchor + 1].to_numpy(dtype=float, copy=True)
        close_df = history.get('close')
        if isinstance(close_df, pd.DataFrame) and not close_df.empty:
            cdf = close_df.reindex(index=ac.index, columns=cols)
            cpx = cdf.iloc[start:anchor + 1].to_numpy(dtype=float, copy=True)
        else:
            cpx = prices
        with np.errstate(all='ignore'):
            dv = v * cpx
            wdv = min(20, dv.shape[0])
            med_dv = np.nanmedian(dv[-wdv:], axis=0)
        valid &= np.isfinite(med_dv) & (med_dv >= 1.0e6)

    rets = np.full_like(prices, np.nan)
    rets[1:] = prices[1:] / prices[:-1] - 1.0
    wvol = min(20, rets.shape[0] - 1)
    with np.errstate(all='ignore'):
        if wvol >= 2:
            vola = np.nanstd(rets[-wvol:], axis=0)
        else:
            vola = np.full(len(cols), np.nan)
    valid &= np.isfinite(vola) & (vola > 0.0) & (vola < 0.15)

    def rev(lb):
        if prices.shape[0] <= lb:
            return np.full(len(cols), np.nan)
        r = -(prices[-1] / prices[-1 - lb] - 1.0)
        r[~np.isfinite(r)] = np.nan
        return r

    r21 = rev(21)
    r10 = rev(10)
    r5 = rev(5)

    rk21 = _rank_pct(r21, valid)
    rk10 = _rank_pct(r10, valid)
    rk5 = _rank_pct(r5, valid)
    rkv = _rank_pct(-vola, valid)

    score = np.full(len(cols), np.nan, dtype=float)
    wsum = np.zeros(len(cols), dtype=float)
    ssum = np.zeros(len(cols), dtype=float)
    for rk, wt in ((rk21, 0.50), (rk10, 0.30), (rk5, 0.10), (rkv, 0.10)):
        has = np.isfinite(rk)
        ssum[has] += wt * rk[has]
        wsum[has] += wt
    good = wsum > 0.0
    score[good] = ssum[good] / wsum[good]

    cand = np.where(valid & np.isfinite(score))[0]
    if cand.size < 20:
        return zero

    n_side = int(min(40, cand.size // 2))
    if n_side < 10:
        return zero

    order = cand[np.argsort(score[cand])]
    shorts = order[:n_side]
    longs = order[-n_side:]

    w = float(min(0.02, 0.5 / float(n_side)))
    weights = np.zeros(len(cols), dtype=float)
    weights[longs] = w
    weights[shorts] = -w

    out = zero.copy()
    for i, ticker in enumerate(cols):
        if weights[i] != 0.0:
            out.loc[ticker] = float(weights[i])

    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
    return out
