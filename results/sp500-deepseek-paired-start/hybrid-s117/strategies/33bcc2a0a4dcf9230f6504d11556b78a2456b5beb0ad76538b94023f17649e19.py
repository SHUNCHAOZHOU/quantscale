import numpy as np
import pandas as pd


def _z(x, mask):
    out = np.zeros_like(x, dtype=float)
    v = x[mask]
    v = v[np.isfinite(v)]
    if v.size < 3:
        return out
    mu = float(v.mean())
    sd = float(v.std(ddof=0))
    if not np.isfinite(sd) or sd < 1e-12:
        return out
    out = (x - mu) / sd
    out[~np.isfinite(out)] = 0.0
    return out


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=tickers, dtype=float)

    ac = history.get('adjusted_close')
    if not isinstance(ac, pd.DataFrame) or ac.shape[0] < 80:
        ac = history.get('close')
    if not isinstance(ac, pd.DataFrame) or ac.shape[0] < 80:
        return zero
    try:
        ac = ac.reindex(columns=tickers)
    except Exception:
        return zero

    dates = ac.index
    last_pos = len(dates) - 1

    # Weekly rebalance: fix the signal at the most recent Monday close and
    # hold the target through the week. Non-Monday turnover is only drift.
    sig = last_pos
    if isinstance(dates, pd.DatetimeIndex) and last_pos >= 5:
        wd = np.asarray(dates.weekday)
        mons = np.where(wd[:last_pos + 1] == 0)[0]
        if mons.size > 0:
            sig = int(mons[-1])
    if sig < 65:
        return zero

    close = ac.values[:sig + 1].astype(float)
    nrow, ncol = close.shape
    cur = close[-1]

    valid = np.isfinite(cur) & (cur > 2.0)
    with np.errstate(invalid='ignore'):
        cnt = np.sum(np.isfinite(close[-130:]), axis=0)
    valid &= (cnt >= 60)

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
            win = dates[max(0, sig - 19):sig + 1]
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
            mom = close[-22] / close[-253] - 1.0   # 12-1 momentum
        elif nrow >= 127:
            mom = close[-22] / close[-127] - 1.0
        else:
            mom = np.full(ncol, np.nan)
        seg = close[-min(61, nrow):]
        dret = seg[1:] / seg[:-1] - 1.0

    vol = np.full(ncol, np.nan)
    okc = np.sum(np.isfinite(dret), axis=0) >= 20
    if okc.any():
        vol[okc] = np.nanstd(dret[:, okc], axis=0)
    vol = np.where(np.isfinite(vol) & (vol > 1e-4), vol, np.nan)

    valid &= np.isfinite(r21) & np.isfinite(mom) & np.isfinite(vol)
    valid &= (r21 > -0.85) & (r21 < 3.0) & (mom > -0.95) & (mom < 10.0)
    if valid.sum() < 8:
        return zero

    # Cross-sectional dispersion regime: std of 21d returns vs its own recent
    # norm. Higher dispersion -> more weight on reversal; lower -> less.
    disp_series = []
    look = int(min(60, nrow - 22))
    for i in range(look):
        end = nrow - 1 - i
        start = end - 21
        if start < 0:
            break
        rr = close[end] / close[start] - 1.0
        rr = rr[np.isfinite(rr)]
        if rr.size >= 10:
            disp_series.append(float(np.std(rr)))
    if len(disp_series) > 0:
        da = np.asarray(disp_series, dtype=float)
        cur_disp = da[0]
        med_disp = float(np.median(da))
    else:
        cur_disp, med_disp = 0.1, 0.1
    if not np.isfinite(med_disp) or med_disp < 1e-6:
        med_disp = 0.1
    disp_ratio = cur_disp / med_disp
    if not np.isfinite(disp_ratio):
        disp_ratio = 1.0

    # Signal weights: dominant slow momentum, fixed low-vol stabilizer, and a
    # dispersion-gated reversal tilt that is squeezed (never removed entirely).
    mom_w = 0.55
    lv_w = 0.20
    rev_w = float(np.clip(0.25 * disp_ratio, 0.10, 0.35))

    with np.errstate(invalid='ignore', divide='ignore'):
        z_mom = _z(mom / (vol * np.sqrt(252.0)), valid)
        z_rev = _z((-r21) / (vol * np.sqrt(21.0)), valid)
        z_lv = _z(-vol * np.sqrt(252.0), valid)

    score = mom_w * z_mom + rev_w * z_rev + lv_w * z_lv
    score = np.where(valid & np.isfinite(score), score, np.nan)

    idx = np.where(np.isfinite(score))[0]
    if idx.size < 6:
        return zero

    order = idx[np.argsort(-score[idx])]
    n_side = int(min(40, idx.size // 2))
    if n_side < 2:
        return zero

    longs = order[:n_side]
    shorts = order[-n_side:]

    # Fixed gross target (~0.70); exposure is NOT cut to zero in losses and is
    # NOT scaled up by volatility. 2% per-name cap respected.
    gross_target = 0.70
    wt = min(0.02, (gross_target / 2.0) / float(n_side))
    if wt <= 0.0:
        return zero

    wv = np.zeros(ncol, dtype=float)
    wv[longs] += wt
    wv[shorts] -= wt
    wv = np.nan_to_num(wv, nan=0.0, posinf=0.0, neginf=0.0)
    wv = np.clip(wv, -0.02, 0.02)

    weights = pd.Series(wv, index=tickers, dtype=float)

    # Enforce dollar-neutrality across active names.
    net = float(weights.sum())
    if abs(net) > 1e-12:
        nz = weights.index[weights != 0.0]
        if len(nz) > 0:
            weights.loc[nz] -= net / len(nz)

    gross = float(weights.abs().sum())
    if gross > 1.0:
        weights = weights * (1.0 / gross)

    return weights.fillna(0.0).astype(float)