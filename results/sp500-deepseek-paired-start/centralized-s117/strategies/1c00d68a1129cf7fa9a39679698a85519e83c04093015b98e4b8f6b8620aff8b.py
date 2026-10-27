import numpy as np
import pandas as pd


def _mad_z(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    y = x.copy()
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    if np.isfinite(mad) and mad > 1e-12:
        scale = 1.4826 * mad
        y = np.clip(y, med - 5.0 * scale, med + 5.0 * scale)
    mu = np.nanmean(y)
    sd = np.nanstd(y)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(y)
    out = (y - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _mask_at(eligv, pos):
    if eligv is None:
        return None
    le = eligv[pos]
    if le.dtype == bool:
        return le
    try:
        arr = np.asarray(le, dtype=float)
        return np.isfinite(arr) & (arr == 1.0)
    except Exception:
        out = np.zeros(len(le), dtype=bool)
        for i, x in enumerate(le):
            try:
                out[i] = bool(x is True or x == 1)
            except Exception:
                out[i] = False
        return out


def _score_at(acv, volv, eligv, s):
    n = acv.shape[1]
    score = np.full(n, np.nan)
    if s < 130:
        return score
    c = acv[:s + 1]
    if c.shape[0] < 127:
        return score
    p0 = c[-1]
    valid = np.isfinite(p0) & (p0 > 2.0)
    if eligv is not None:
        em = _mask_at(eligv, s)
        if em is not None:
            valid &= em
    p21 = c[-22]
    with np.errstate(divide='ignore', invalid='ignore'):
        ret21 = p0 / p21 - 1.0
    valid &= np.isfinite(ret21)
    R = c[-127:]
    with np.errstate(divide='ignore', invalid='ignore'):
        daily = R[1:] / R[:-1] - 1.0
    daily = np.where(np.isfinite(daily), daily, np.nan)
    vol60 = np.nanstd(daily[-60:], axis=0)
    valid &= np.isfinite(vol60) & (vol60 > 1e-6)
    if volv is not None:
        v = volv[:s + 1]
        start = max(0, s - 59)
        vv = v[start:s + 1]
        if vv.shape[0] > 0:
            cc = c[-vv.shape[0]:]
            dv = np.nanmedian(cc * vv, axis=0)
            valid &= np.isfinite(dv) & (dv > 0)
            if valid.sum() >= 20:
                thresh = np.nanpercentile(dv[valid], 20)
                valid &= dv >= thresh
    idx = np.where(valid)[0]
    if idx.size < 10:
        return score
    D = daily[-126:]
    m = np.nanmean(D, axis=1)
    mc = m - np.nanmean(m)
    var_m = np.nanmean(mc ** 2)
    fallback = -ret21 / np.maximum(vol60, 1e-6)
    if (not np.isfinite(var_m)) or var_m < 1e-12:
        raw = fallback
    else:
        Dm = D - np.nanmean(D, axis=0)
        cov = np.nanmean(Dm * mc[:, None], axis=0)
        beta = cov / var_m
        beta = np.where(np.isfinite(beta), beta, 1.0)
        resid = D - beta[None, :] * m[:, None]
        resid21 = np.nansum(resid[-21:], axis=0)
        vol_res = np.nanstd(resid[-21:], axis=0)
        raw = -resid21 / np.maximum(vol_res, 1e-6)
        raw = np.where(np.isfinite(raw), raw, fallback)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    z = _mad_z(raw[idx])
    score[idx] = z
    return score


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
    if last_pos < 130:
        return zero
    acv = ac.values.astype(float, copy=False)
    vol = history.get('volume')
    volv = None
    if vol is not None:
        try:
            volv = vol.reindex(index=dates, columns=tickers).values.astype(float, copy=False)
        except Exception:
            volv = None
    elig = history.get('eligible')
    eligv = None
    if elig is not None:
        try:
            eligv = elig.reindex(index=dates, columns=tickers).values
        except Exception:
            eligv = None
    wd = dates.weekday.values
    mons = np.where(wd == 0)[0]
    mons = mons[mons <= last_pos]
    signal_pos = int(mons[-1]) if len(mons) > 0 else last_pos
    if signal_pos < 130:
        return zero
    p0 = acv[last_pos]
    cur_valid = np.isfinite(p0) & (p0 > 2.0)
    if eligv is not None:
        em = _mask_at(eligv, last_pos)
        if em is not None:
            cur_valid &= em
    sig_mons = mons[mons <= signal_pos]
    if len(sig_mons) >= 3:
        sig_list = [(int(sig_mons[-1]), 0.5), (int(sig_mons[-2]), 0.3), (int(sig_mons[-3]), 0.2)]
    elif len(sig_mons) == 2:
        sig_list = [(int(sig_mons[-1]), 0.65), (int(sig_mons[-2]), 0.35)]
    elif len(sig_mons) == 1:
        sig_list = [(int(sig_mons[-1]), 1.0)]
    else:
        sig_list = [(int(signal_pos), 1.0)]
    sum_z = np.zeros(n, dtype=float)
    cnt_z = np.zeros(n, dtype=float)
    for s, wt in sig_list:
        zs = _score_at(acv, volv, eligv, s)
        ok = np.isfinite(zs)
        sum_z[ok] += wt * zs[ok]
        cnt_z[ok] += wt
    z_avg = np.full(n, np.nan)
    ok = cnt_z > 0
    z_avg[ok] = sum_z[ok] / cnt_z[ok]
    valid = cur_valid & np.isfinite(z_avg)
    idx = np.where(valid)[0]
    if idx.size < 8:
        return zero
    z = z_avg[idx].copy()
    z = np.clip(z, -3.0, 3.0)
    z = z - np.mean(z)
    if not np.any(np.isfinite(z)):
        return zero
    target_gross = min(0.98, 0.02 * idx.size * 0.95)
    S = np.sum(np.abs(z))
    if S < 1e-12:
        return zero
    w = z * (target_gross / S)
    for _ in range(6):
        w = np.clip(w, -0.02, 0.02)
        w = w - np.mean(w)
        maxabs = np.max(np.abs(w)) if w.size else 0.0
        if maxabs > 0.02:
            w *= 0.02 / maxabs
        gross = np.sum(np.abs(w))
        if gross > 1.0:
            w *= 1.0 / gross
        elif gross < target_gross * 0.999 and maxabs < 0.02 * 0.999 and gross > 1e-12:
            scale = min(target_gross / gross, 0.02 / maxabs) if maxabs > 1e-12 else target_gross / gross
            if scale > 1.0:
                w *= scale
    w = w - np.mean(w)
    maxabs = np.max(np.abs(w)) if w.size else 0.0
    if maxabs > 0.02:
        w *= 0.02 / maxabs
        w = w - np.mean(w)
    gross = np.sum(np.abs(w))
    if gross > 1.0:
        w *= 1.0 / gross
        w = w - np.mean(w)
    gross = np.sum(np.abs(w))
    if gross > 1.0:
        w *= 1.0 / gross
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w_full = np.zeros(n, dtype=float)
    w_full[idx] = w
    w_full = np.clip(w_full, -0.02, 0.02)
    return pd.Series(w_full, index=tickers)
