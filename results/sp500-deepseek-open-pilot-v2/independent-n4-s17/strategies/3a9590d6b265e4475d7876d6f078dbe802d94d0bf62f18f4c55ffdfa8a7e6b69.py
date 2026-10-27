import numpy as np
import pandas as pd


def generate_weights(history, universe):
    uni = pd.Index(universe)
    ans = pd.Series(0.0, index=uni, dtype=float)
    n_uni = len(uni)
    if n_uni == 0:
        return ans

    px_key = 'adjusted_close' if 'adjusted_close' in history else 'close'
    if px_key not in history:
        return ans
    px_df = history[px_key]
    if px_df is None or len(px_df) == 0:
        return ans

    cols = [t for t in uni if t in px_df.columns]
    if len(cols) == 0:
        return ans

    px = px_df.loc[:, cols].sort_index()
    P = px.to_numpy(dtype=float)
    m, n = P.shape
    if m < 30:
        return ans

    last = P[-1, :]
    valid = np.isfinite(last) & (last > 0.0)

    if 'eligible' in history:
        try:
            edf = history['eligible'].reindex(index=px.index, columns=cols)
            if len(edf) > 0:
                last_e = edf.iloc[-1].fillna(False).astype(bool).to_numpy()
                valid &= last_e
        except Exception:
            pass

    if m >= 252:
        p0 = P[-252, :]
        p1 = P[-21, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            mom12 = p1 / p0 - 1.0
        mom12 = np.where(np.isfinite(mom12), mom12, np.nan)
    else:
        mom12 = np.full(n, np.nan, dtype=float)

    if m >= 126:
        p0 = P[-126, :]
        p1 = P[-21, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            mom6 = p1 / p0 - 1.0
        mom6 = np.where(np.isfinite(mom6), mom6, np.nan)
    else:
        mom6 = np.full(n, np.nan, dtype=float)

    if m >= 6:
        with np.errstate(divide='ignore', invalid='ignore'):
            ret5 = P[-1, :] / P[-6, :] - 1.0
        ret5 = np.where(np.isfinite(ret5), ret5, np.nan)
    else:
        ret5 = np.full(n, np.nan, dtype=float)

    if m >= 25:
        start = max(0, m - 61)
        win = P[start:, :]
        win = np.where(win > 0.0, win, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            lr = np.diff(np.log(win), axis=0)
        cnt = np.sum(np.isfinite(lr), axis=0)
        vol = np.nanstd(lr, axis=0)
        vol = np.where((cnt >= 15) & (vol > 0.0), vol, np.nan)
    else:
        vol = np.full(n, np.nan, dtype=float)

    liq = np.full(n, np.nan, dtype=float)
    if 'volume' in history:
        try:
            vdf = history['volume']
            vsub = vdf.reindex(index=px.index, columns=cols)
            V = vsub.to_numpy(dtype=float)
            if m >= 20:
                start = max(0, m - 61)
                winP = P[start:, :]
                winV = V[start:, :]
                dollar = winP * winV
                liq = np.nanmedian(dollar, axis=0)
        except Exception:
            liq = np.full(n, np.nan, dtype=float)

    valid &= np.isfinite(mom12) & np.isfinite(mom6) & np.isfinite(ret5) & np.isfinite(vol)
    if np.isfinite(liq).sum() > 20:
        valid &= np.isfinite(liq) & (liq > 0.0)
        if valid.sum() >= 20:
            q10 = np.nanpercentile(liq[valid], 10.0)
            valid &= (liq >= q10)

    if valid.sum() < 20:
        valid = np.isfinite(mom12) & np.isfinite(mom6) & np.isfinite(ret5) & np.isfinite(vol)
        if np.isfinite(liq).sum() > 20:
            valid &= np.isfinite(liq) & (liq > 0.0)

    idx = np.where(valid)[0]
    if len(idx) < 2:
        return ans

    mom12_s = pd.Series(mom12[idx])
    mom6_s = pd.Series(mom6[idx])
    vol_s = pd.Series(vol[idx])
    rev_s = pd.Series(-ret5[idx])

    r12 = mom12_s.rank(method='average', pct=True).to_numpy()
    r6 = mom6_s.rank(method='average', pct=True).to_numpy()
    rlow = 1.0 - vol_s.rank(method='average', pct=True).to_numpy()
    rrev = rev_s.rank(method='average', pct=True).to_numpy()

    score = 0.45 * r12 + 0.25 * r6 + 0.20 * rlow + 0.10 * rrev
    order = np.argsort(-score, kind='mergesort')
    n_side = min(50, len(idx) // 2)
    if n_side < 1:
        return ans

    long_pos = order[:n_side]
    short_pos = order[-n_side:]
    w = min(0.02, 0.5 / float(n_side))
    weights = np.zeros(n_uni, dtype=float)
    pos_map = {t: i for i, t in enumerate(uni)}

    for loc in long_pos:
        ticker = cols[idx[loc]]
        weights[pos_map[ticker]] = w
    for loc in short_pos:
        ticker = cols[idx[loc]]
        weights[pos_map[ticker]] = -w

    return pd.Series(weights, index=uni, dtype=float)
