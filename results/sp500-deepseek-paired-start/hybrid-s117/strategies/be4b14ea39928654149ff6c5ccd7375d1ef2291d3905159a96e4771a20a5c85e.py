import numpy as np
import pandas as pd


def _zscore(x, mask):
    z = np.zeros_like(x, dtype=float)
    v = x[mask]
    v = v[np.isfinite(v)]
    if v.size < 3:
        return z
    mu = float(v.mean())
    sd = float(v.std(ddof=0))
    if not np.isfinite(sd) or sd < 1e-12:
        return z
    out = (x - mu) / sd
    out[~np.isfinite(out)] = 0.0
    return out


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=tickers, dtype=float)

    ac = history.get('adjusted_close')
    if not isinstance(ac, pd.DataFrame) or ac.shape[0] < 40:
        ac = history.get('close')
    if not isinstance(ac, pd.DataFrame) or ac.shape[0] < 40:
        return zero
    try:
        ac = ac.reindex(columns=tickers)
    except Exception:
        return zero

    dates = ac.index
    last_pos = len(dates) - 1

    # Weekly rebalance: signal is fixed at the most recent Monday close.
    sig = last_pos
    if isinstance(dates, pd.DatetimeIndex) and last_pos >= 5:
        wd = np.asarray(dates.weekday)
        mons = np.where(wd[:last_pos + 1] == 0)[0]
        if mons.size > 0:
            sig = int(mons[-1])

    if sig < 25:
        return zero

    close = ac.values[:sig + 1].astype(float)
    nrow, ncol = close.shape
    cur = close[-1]

    with np.errstate(invalid='ignore', divide='ignore'):
        cnt = np.sum(np.isfinite(close[-120:]), axis=0)
    valid = np.isfinite(cur) & (cur > 3.0) & (cnt >= 60)

    # Point-in-time eligibility (NaN treated as tradable).
    edf = history.get('eligible')
    if isinstance(edf, pd.DataFrame) and edf.shape[0] > 0:
        try:
            er = edf.reindex(columns=tickers)
            try:
                row = er.loc[dates[sig]]
            except Exception:
                row = er.iloc[-1]
            vals = pd.to_numeric(row, errors='coerce')
            elig = (vals == 1).values | vals.isna().values
            valid &= elig
        except Exception:
            pass

    # Liquidity: 20-day median dollar volume, drop bottom 20% of valid names.
    vdf = history.get('volume')
    cdf = history.get('close')
    if isinstance(vdf, pd.DataFrame):
        try:
            win = dates[max(0, sig - 19): sig + 1]
            v = vdf.reindex(index=win, columns=tickers).values.astype(float)
            if isinstance(cdf, pd.DataFrame):
                c = cdf.reindex(index=win, columns=tickers).values.astype(float)
            else:
                c = ac.reindex(index=win).values.astype(float)
            with np.errstate(invalid='ignore'):
                dv = np.nanmedian(v * c, axis=0)
            dvv = dv[np.isfinite(dv)]
            if dvv.size >= 5:
                thr = max(2.0e5, float(np.nanpercentile(dvv, 20)))
                valid &= np.isfinite(dv) & (dv >= thr)
        except Exception:
            pass

    with np.errstate(invalid='ignore', divide='ignore'):
        r21 = close[-1] / close[-22] - 1.0
        if nrow >= 253:
            mom = close[-22] / close[-253] - 1.0
        elif nrow >= 127:
            mom = close[-22] / close[-127] - 1.0
        else:
            mom = np.full(ncol, np.nan)

        seg = close[-min(61, nrow):]
        dret = seg[1:] / seg[:-1] - 1.0

    vol = np.full(ncol, np.nan)
    ok_cols = np.sum(np.isfinite(dret), axis=0) >= 20
    if ok_cols.any():
        sub = dret[:, ok_cols]
        sd = np.nanstd(sub, axis=0)
        vol[ok_cols] = sd
    vol = np.where(np.isfinite(vol) & (vol > 1e-4), vol, np.nan)

    valid &= np.isfinite(r21) & np.isfinite(mom) & np.isfinite(vol)
    valid &= (r21 > -0.85) & (r21 < 3.0) & (mom > -0.95) & (mom < 10.0)

    if valid.sum() < 8:
        return zero

    with np.errstate(invalid='ignore', divide='ignore'):
        mom_adj = mom / (vol * np.sqrt(252.0))
        rev_adj = (-r21) / (vol * np.sqrt(21.0))
        low_vol = -vol * np.sqrt(252.0)

    z_mom = _zscore(mom_adj, valid)
    z_rev = _zscore(rev_adj, valid)
    z_lv = _zscore(low_vol, valid)

    score = 0.60 * z_mom + 0.30 * z_rev + 0.10 * z_lv
    score = np.where(valid & np.isfinite(score), score, np.nan)

    idx = np.where(np.isfinite(score))[0]
    if idx.size < 6:
        return zero

    order = idx[np.argsort(-score[idx])]
    n_side = min(40, idx.size // 2)
    if n_side < 2:
        return zero

    longs = order[:n_side]
    shorts = order[-n_side:]

    w = min(0.02, 0.5 / float(n_side))
    wv = np.zeros(ncol, dtype=float)
    wv[longs] += w
    wv[shorts] -= w

    wv = np.nan_to_num(wv, nan=0.0, posinf=0.0, neginf=0.0)
    wv = np.clip(wv, -0.02, 0.02)

    weights = pd.Series(wv, index=tickers, dtype=float)
    net = weights.sum()
    if abs(net) > 1e-12:
        nz = weights.index[weights != 0.0]
        if len(nz) > 0:
            weights.loc[nz] -= net / len(nz)

    gross = weights.abs().sum()
    if gross > 1.0:
        weights = weights * (1.0 / gross)

    return weights.fillna(0.0).astype(float)