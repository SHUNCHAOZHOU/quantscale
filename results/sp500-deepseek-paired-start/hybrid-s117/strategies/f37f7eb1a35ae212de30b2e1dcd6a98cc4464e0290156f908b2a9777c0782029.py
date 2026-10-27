import numpy as np
import pandas as pd


def _zscore(x, mask):
    z = np.zeros_like(x, dtype=float)
    v = x[mask]
    v = v[np.isfinite(v)]
    if v.size < 3:
        return z
    mu = float(np.mean(v))
    sd = float(np.std(v, ddof=0))
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
    if not isinstance(ac, pd.DataFrame):
        ac = history.get('close')
    if not isinstance(ac, pd.DataFrame) or ac.shape[0] < 65:
        return zero
    try:
        ac = ac.reindex(columns=tickers)
    except Exception:
        return zero

    dates = ac.index
    last_pos = len(dates) - 1
    if last_pos < 64:
        return zero

    # Fix the signal at the most recent Monday close.  On non-Mondays this
    # returns the same target as Monday, so trades occur once per week.
    sig = last_pos
    if isinstance(dates, pd.DatetimeIndex) and last_pos >= 5:
        wd = dates.weekday.values
        mons = np.where(wd[:last_pos + 1] == 0)[0]
        if mons.size > 0:
            sig = int(mons[-1])
    if sig < 64:
        return zero

    close = ac.iloc[:sig + 1].values.astype(float)
    nrow, ncol = close.shape
    cur = close[-1]

    cnt = np.sum(np.isfinite(close[-min(120, nrow):]), axis=0)
    valid = np.isfinite(cur) & (cur > 3.0) & (cnt >= 60)

    edf = history.get('eligible')
    if isinstance(edf, pd.DataFrame):
        try:
            er = edf.reindex(columns=tickers)
            row = er.iloc[sig] if sig < len(er) else er.iloc[-1]
            vals = pd.to_numeric(row, errors='coerce').values
            elig = (vals == 1) | (~np.isfinite(vals))
            valid = valid & elig
        except Exception:
            pass

    vdf = history.get('volume')
    cdf = history.get('close')
    if isinstance(vdf, pd.DataFrame):
        try:
            v = vdf.reindex(columns=tickers).iloc[:sig + 1].tail(20).values.astype(float)
            if isinstance(cdf, pd.DataFrame):
                c = cdf.reindex(columns=tickers).iloc[:sig + 1].tail(20).values.astype(float)
            else:
                c = close[-v.shape[0]:]
            if v.shape == c.shape and v.shape[0] >= 5:
                with np.errstate(invalid='ignore', divide='ignore'):
                    dv = np.nanmedian(v * c, axis=0)
                finite_dv = np.isfinite(dv)
                if finite_dv.sum() >= 10:
                    base = valid & finite_dv
                    if base.sum() >= 10:
                        q = np.nanpercentile(dv[base], 20.0)
                        thr = max(1.0e5, float(q))
                    else:
                        thr = 1.0e5
                    valid = valid & finite_dv & (dv >= thr)
        except Exception:
            pass

    rets = close[1:] / close[:-1] - 1.0
    vol = np.full(ncol, np.nan)
    if rets.shape[0] >= 20:
        look = min(126, rets.shape[0])
        sub = rets[-look:]
        cntr = np.sum(np.isfinite(sub), axis=0)
        sd = np.nanstd(sub, axis=0)
        vol = np.where((cntr >= 20) & np.isfinite(sd) & (sd > 1e-4),
                       sd * np.sqrt(252.0), np.nan)

    if nrow >= 253:
        mom = close[-22] / close[-253] - 1.0
    elif nrow >= 127:
        mom = close[-22] / close[-127] - 1.0
    else:
        mom = close[-22] / close[-64] - 1.0
    mom = np.where(np.isfinite(mom), mom, np.nan)

    if nrow >= 22:
        r21 = close[-1] / close[-22] - 1.0
    else:
        r21 = np.full(ncol, np.nan)

    valid = valid & np.isfinite(vol) & np.isfinite(mom) & np.isfinite(r21)
    valid = valid & (mom > -0.95) & (mom < 10.0) & (r21 > -0.90) & (r21 < 3.0)

    hi_win = min(252, nrow)
    hi = np.nanmax(close[-hi_win:], axis=0)
    dist = cur / hi - 1.0
    dist = np.where(np.isfinite(dist), dist, np.nan)

    pos_frac = np.full(ncol, np.nan)
    if nrow > 22:
        seg = close[-min(127, nrow):]
        if seg.shape[0] > 22:
            rr = seg[21:] / seg[:-21] - 1.0
            pos = (rr > 0.0).astype(float)
            pos[np.isnan(rr)] = np.nan
            cntp = np.sum(np.isfinite(rr), axis=0)
            with np.errstate(invalid='ignore', divide='ignore'):
                pf = np.nanmean(pos, axis=0)
            pos_frac = np.where(cntp >= 20, pf, np.nan)

    valid = valid & np.isfinite(dist)
    if valid.sum() < 8:
        return zero

    mu_mom = np.nanmean(mom[valid])
    mom_res = mom - mu_mom

    z_mom = _zscore(mom_res, valid)
    z_lv = _zscore(-vol, valid)
    z_dist = _zscore(dist, valid)

    if np.isfinite(pos_frac[valid]).sum() >= max(8, int(valid.sum()) // 2):
        z_pos = _zscore(pos_frac, valid)
        z_trend = 0.5 * z_dist + 0.5 * z_pos
    else:
        z_trend = z_dist

    score = 0.55 * z_mom + 0.30 * z_lv + 0.15 * z_trend
    score = np.where(valid & np.isfinite(score), score, np.nan)

    idx = np.where(np.isfinite(score))[0]
    if idx.size < 6:
        return zero

    order = idx[np.argsort(-score[idx])]
    n_side = min(50, idx.size // 2)
    if n_side < 2:
        return zero

    longs = order[:n_side]
    shorts = order[-n_side:]
    w = min(0.02, 0.5 / float(n_side))

    wv = np.zeros(ncol, dtype=float)
    wv[longs] += w
    wv[shorts] -= w

    net = wv.sum()
    if abs(net) > 1e-12:
        nz = np.where(wv != 0.0)[0]
        if nz.size > 0:
            wv[nz] -= net / float(nz.size)

    wv = np.nan_to_num(wv, nan=0.0, posinf=0.0, neginf=0.0)
    wv = np.clip(wv, -0.02, 0.02)
    gross = np.abs(wv).sum()
    if gross > 1.0:
        wv = wv * (1.0 / gross)

    return pd.Series(wv, index=tickers, dtype=float).fillna(0.0)