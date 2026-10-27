import numpy as np
import pandas as pd


def _rank_pct(vals, mask):
    out = np.full(vals.shape, np.nan, dtype=float)
    idx = np.where(mask & np.isfinite(vals))[0]
    if idx.size == 0:
        return out
    r = pd.Series(vals[idx]).rank(method='average').to_numpy(dtype=float)
    out[idx] = (r - 0.5) / float(idx.size)
    return out


def _inv_vol_weights(vol, idx, side_dollar, cap=0.02):
    iv = 1.0 / vol[idx]
    w = iv / iv.sum() * side_dollar
    for _ in range(8):
        over = w > cap
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = ~over
        if not free.any() or excess <= 0.0:
            break
        w[free] += excess * iv[free] / iv[free].sum()
    return np.minimum(w, cap)


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    adj = history.get('adjusted_close')
    if not isinstance(adj, pd.DataFrame) or adj.empty:
        adj = history.get('close')
    if not isinstance(adj, pd.DataFrame) or adj.empty:
        return zero

    cols = [c for c in universe if c in adj.columns]
    if len(cols) < 20:
        return zero

    adj = adj.loc[:, cols].sort_index()
    n_rows = len(adj)
    if n_rows < 90:
        return zero

    rebalance = 21
    anchor = ((n_rows - 1) // rebalance) * rebalance
    if anchor < 90:
        return zero

    px = adj.iloc[:anchor + 1].ffill()
    prices = px.to_numpy(dtype=float, copy=True)
    n_p, n_c = prices.shape
    if n_p < 30 or n_c < 20:
        return zero

    last_px = prices[-1]
    valid = np.isfinite(last_px) & (last_px > 5.0)

    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame) and not elig.empty:
        try:
            e = elig.reindex(index=adj.index, columns=cols)
            e_row = e.iloc[anchor].fillna(False).astype(bool).to_numpy(dtype=bool)
            valid = valid & e_row
        except Exception:
            pass

    vol_df = history.get('volume')
    close_df = history.get('close')
    if isinstance(vol_df, pd.DataFrame) and isinstance(close_df, pd.DataFrame):
        try:
            v = vol_df.reindex(index=adj.index, columns=cols).iloc[:anchor + 1].to_numpy(dtype=float, copy=True)
            c = close_df.reindex(index=adj.index, columns=cols).iloc[:anchor + 1].to_numpy(dtype=float, copy=True)
            w = min(60, n_p)
            with np.errstate(all='ignore'):
                med_dv = np.nanmedian(v[-w:] * c[-w:], axis=0)
            valid = valid & np.isfinite(med_dv) & (med_dv >= 2.0e6)
        except Exception:
            pass

    rets = np.full((n_p, n_c), np.nan, dtype=float)
    with np.errstate(all='ignore'):
        rets[1:] = prices[1:] / prices[:-1] - 1.0
    rets[~np.isfinite(rets)] = np.nan
    rets[np.abs(rets) > 0.5] = np.nan

    win = min(126, n_p - 1)
    if win < 40:
        return zero
    tail = rets[-win:]
    with np.errstate(all='ignore'):
        vol = np.nanstd(tail, axis=0)
        cnt = np.isfinite(tail).sum(axis=0)
    valid = valid & np.isfinite(vol) & (vol > 0.002) & (vol < 0.25) & (cnt >= int(0.6 * win))
    if int(valid.sum()) < 20:
        return zero

    with np.errstate(all='ignore'):
        mkt = np.nanmean(rets, axis=1)
    resid = rets - mkt[:, None]

    def _resid_sum(lb):
        if n_p <= lb:
            return np.full(n_c, np.nan, dtype=float)
        seg = resid[-lb:]
        with np.errstate(all='ignore'):
            s = np.nansum(seg, axis=0)
            k = np.isfinite(seg).sum(axis=0)
        s = np.asarray(s, dtype=float)
        s[k < int(0.6 * lb)] = np.nan
        return s

    res21 = _resid_sum(21)
    res5 = _resid_sum(5)

    base = valid & np.isfinite(res21) & np.isfinite(res5) & np.isfinite(vol)
    if int(base.sum()) < 20:
        return zero

    with np.errstate(all='ignore'):
        sig21 = -res21 / vol
        sig5 = -res5 / vol
        sigv = -vol
    r21 = _rank_pct(sig21, base)
    r5 = _rank_pct(sig5, base)
    rv = _rank_pct(sigv, base)

    score = np.full(n_c, np.nan, dtype=float)
    ssum = np.zeros(n_c, dtype=float)
    wsum = np.zeros(n_c, dtype=float)
    for rk, wgt in ((r21, 0.55), (r5, 0.30), (rv, 0.15)):
        has = np.isfinite(rk)
        ssum[has] += wgt * rk[has]
        wsum[has] += wgt
    good = wsum > 0.0
    score[good] = ssum[good] / wsum[good]

    cand = np.where(base & np.isfinite(score))[0]
    if cand.size < 20:
        return zero

    n_side = int(min(30, cand.size // 2))
    if n_side < 8:
        return zero

    order = cand[np.argsort(score[cand])]
    shorts = order[:n_side]
    longs = order[-n_side:]

    side_dollar = 0.35
    wl = _inv_vol_weights(vol, longs, side_dollar)
    ws = _inv_vol_weights(vol, shorts, side_dollar)
    tgt = min(wl.sum(), ws.sum())
    if tgt <= 0.0 or not np.isfinite(tgt):
        return zero
    wl = wl * (tgt / wl.sum())
    ws = ws * (tgt / ws.sum())

    weights = np.zeros(n_c, dtype=float)
    weights[longs] = wl
    weights[shorts] = -ws

    out = zero.copy()
    out.loc[cols] = weights
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
    out = out.clip(-0.02, 0.02)

    net = float(out.sum())
    if abs(net) > 1e-10:
        longs_adj = out.clip(lower=0.0)
        sl = float(longs_adj.sum())
        if sl > 0.0:
            out = out - (net / sl) * longs_adj
    gross = float(out.abs().sum())
    if gross > 1.0:
        out = out * (1.0 / gross)
    return out.astype(float)