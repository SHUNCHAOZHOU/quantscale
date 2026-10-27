import numpy as np
import pandas as pd


def _rank(x):
    out = np.full(len(x), np.nan)
    mask = np.isfinite(x)
    m = int(mask.sum())
    if m == 0:
        return out
    vals = x[mask]
    order = np.argsort(vals)
    ranks = np.empty(m, dtype=float)
    ranks[order] = np.arange(m, dtype=float)
    if m > 1:
        ranks = ranks / (m - 1)
    else:
        ranks = np.zeros(m)
    out[mask] = ranks
    return out


def _cap(w, cap):
    w = w.copy()
    for _ in range(20):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = (~over) & (w > 0)
        if under.sum() == 0:
            break
        base = w[under].sum()
        if base <= 0:
            break
        w[under] = w[under] + excess * (w[under] / base)
    return w


def generate_weights(history, universe):
    tickers = list(universe)
    n = len(tickers)
    idx = pd.Index(tickers)
    weights = pd.Series(0.0, index=idx)
    if n == 0:
        return weights

    adj = history['adjusted_close'].reindex(columns=tickers)
    elig_df = history.get('eligible')
    if elig_df is not None and elig_df.shape[0] > 0:
        last = elig_df.reindex(columns=tickers).iloc[-1].fillna(0)
        last_elig = last.values.astype(bool)
    else:
        last_elig = np.ones(n, dtype=bool)

    p = adj.values.astype(float)
    T = p.shape[0]
    if T < 40:
        return weights

    ret = p[1:] / p[:-1] - 1.0
    ret = np.where(np.isfinite(ret), ret, np.nan)

    vol = np.nanstd(ret[-60:], axis=0)
    vol = np.where((vol > 1e-6) & np.isfinite(vol), vol, np.nan)

    skip = 21

    def mom(lb):
        if T - 1 - skip - lb < 0:
            return np.full(n, np.nan)
        return p[T - 1 - skip] / p[T - 1 - skip - lb] - 1.0

    m1 = mom(120)
    m2 = mom(250)
    r1 = _rank(m1)
    r2 = _rank(m2)

    with np.errstate(invalid='ignore'):
        sig = np.nanmean(np.vstack([r1, r2]), axis=0)

    valid = last_elig & np.isfinite(sig) & np.isfinite(vol)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 6:
        return weights

    s = sig[valid_idx]
    order = np.argsort(-s)
    k = min(30, max(2, len(valid_idx) // 3))

    long_idx = valid_idx[order[:k]]
    short_idx = valid_idx[order[-k:]]

    def book_weights(sel, target_gross):
        iv = 1.0 / vol[sel]
        tot = iv.sum()
        if tot <= 0:
            return np.full(len(sel), target_gross / len(sel))
        return iv / tot * target_gross

    lw = _cap(book_weights(long_idx, 0.5), 0.02)
    sw = _cap(book_weights(short_idx, 0.5), 0.02)

    w_arr = np.zeros(n)
    w_arr[long_idx] = lw
    w_arr[short_idx] = -sw

    gross = np.abs(w_arr).sum()
    if gross > 1.0:
        w_arr *= 1.0 / gross

    weights.iloc[:] = w_arr
    return weights