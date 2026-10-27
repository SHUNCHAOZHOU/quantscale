import numpy as np
import pandas as pd


def _zscore(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    y = x.copy()
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    if np.isfinite(mad) and mad > 1e-12:
        lo = med - 5.0 * 1.4826 * mad
        hi = med + 5.0 * 1.4826 * mad
        y = np.clip(y, lo, hi)
    mu = np.nanmean(y)
    sd = np.nanstd(y)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(y)
    out = (y - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(np.zeros(n, dtype=float), index=tickers)

    ac = history.get('adjusted_close')
    if ac is None:
        return zero
    try:
        ac = ac.reindex(columns=tickers)
    except Exception:
        return zero

    dates = ac.index
    last_pos = len(dates) - 1
    if last_pos < 261:
        return zero

    wd = dates.weekday.values
    mons = np.where(wd == 0)[0]
    mons = mons[mons <= last_pos]
    signal_pos = int(mons[-1]) if len(mons) > 0 else last_pos
    if signal_pos < 261:
        return zero

    close = ac.values.astype(float, copy=False)
    c = close[:signal_pos + 1]
    if c.shape[0] < 261:
        return zero

    p0 = c[-1]
    valid = np.isfinite(p0) & (p0 > 2.0)

    elig = history.get('eligible')
    if elig is not None:
        try:
            e = elig.reindex(columns=tickers)
            le = np.asarray(e.values[signal_pos])
            if le.dtype == bool:
                valid &= le
            else:
                try:
                    valid &= (le.astype(float) == 1.0)
                except Exception:
                    pass
        except Exception:
            pass

    p21 = c[-22]
    p252 = c[-253]
    ret21 = p0 / p21 - 1.0
    mom = p21 / p252 - 1.0
    valid &= np.isfinite(ret21) & np.isfinite(mom)

    R = c[-253:]
    with np.errstate(divide='ignore', invalid='ignore'):
        daily = R[1:] / R[:-1] - 1.0
    daily = np.where(np.isfinite(daily), daily, np.nan)
    vol60 = np.nanstd(daily[-60:], axis=0)
    valid &= np.isfinite(vol60) & (vol60 > 1e-6)

    vol = history.get('volume')
    if vol is not None:
        try:
            v = vol.reindex(columns=tickers).values
            start = max(0, signal_pos - 59)
            vv = v[start:signal_pos + 1]
            cc = c[-vv.shape[0]:]
            dv = np.nanmedian(cc * vv, axis=0)
            valid &= np.isfinite(dv) & (dv > 0)
            if valid.sum() >= 20:
                thresh = np.nanpercentile(dv[valid], 20)
                valid &= dv >= thresh
        except Exception:
            pass

    if valid.sum() >= 40:
        vth = np.nanpercentile(vol60[valid], 75)
        valid &= vol60 <= vth

    idx = np.where(valid)[0]
    if idx.size < 10:
        return zero

    D = daily[-126:]
    m = np.nanmean(D, axis=1)
    mc = m - np.nanmean(m)
    var_m = np.nanmean(mc ** 2)
    fallback = -ret21 / np.maximum(vol60, 1e-6)
    if (not np.isfinite(var_m)) or var_m < 1e-12:
        resid_rev = fallback
    else:
        Dm = D - np.nanmean(D, axis=0)
        cov = np.nanmean(Dm * mc[:, None], axis=0)
        beta = cov / var_m
        beta = np.where(np.isfinite(beta), beta, 1.0)
        resid = D - beta[None, :] * m[:, None]
        resid21 = np.nansum(resid[-21:], axis=0)
        vol_res = np.nanstd(resid[-21:], axis=0)
        resid_rev = -resid21 / np.maximum(vol_res, 1e-6)
        resid_rev = np.where(np.isfinite(resid_rev), resid_rev, fallback)

    if c.shape[0] >= 201:
        eq126 = np.nanmean(c[-1] / c[-127] - 1.0)
        eq200 = np.nanmean(c[-1] / c[-201] - 1.0)
        trend = 0.5 * (eq126 + eq200)
    else:
        trend = np.nanmean(c[-1] / c[-127] - 1.0)

    if trend > 0:
        w_rev = 0.65
        w_mom = 0.35
    else:
        w_rev = 0.30
        w_mom = 0.70

    z_rev = np.zeros(n, dtype=float)
    z_mom = np.zeros(n, dtype=float)
    z_rev[idx] = _zscore(resid_rev[idx])
    z_mom[idx] = _zscore(mom[idx])
    score = w_rev * z_rev + w_mom * z_mom
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    mpos = min(50, idx.size // 2)
    if mpos < 5:
        return zero
    order = np.argsort(score[idx])
    long_sel = idx[order[-mpos:]]
    short_sel = idx[order[:mpos]]

    wt = min(0.02, 0.5 / mpos)
    w = np.zeros(n, dtype=float)
    w[long_sel] = wt
    w[short_sel] = -wt
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, -0.02, 0.02)
    return pd.Series(w, index=tickers)