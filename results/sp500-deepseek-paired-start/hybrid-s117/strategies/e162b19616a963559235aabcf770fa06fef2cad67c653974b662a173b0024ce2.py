import numpy as np
import pandas as pd
import math


def _zscore(x, mask):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    m = mask & np.isfinite(x)
    if m.sum() < 3:
        return out
    v = x[m]
    mu = float(np.mean(v))
    sd = float(np.std(v, ddof=0))
    if not np.isfinite(sd) or sd < 1e-12:
        return out
    z = (x - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return z


def _month_end_pos(dates, last_pos):
    if not isinstance(dates, pd.DatetimeIndex) or last_pos < 0:
        return last_pos
    start = max(0, last_pos - 25)
    for i in range(last_pos, start - 1, -1):
        d = dates[i]
        try:
            nb = d + pd.offsets.BDay(1)
            if nb.month != d.month:
                return i
        except Exception:
            continue
    for i in range(last_pos - 1, -1, -1):
        if dates[i].month != dates[i + 1].month:
            return i
    return last_pos


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    zero = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return zero

    ac = history.get('adjusted_close')
    if not isinstance(ac, pd.DataFrame) or ac.empty:
        ac = history.get('close')
    if not isinstance(ac, pd.DataFrame) or ac.empty:
        return zero
    try:
        ac = ac.reindex(columns=tickers)
    except Exception:
        return zero

    dates = ac.index
    last_pos = len(dates) - 1
    if last_pos < 65:
        return zero

    asof = _month_end_pos(dates, last_pos)
    if asof < 65:
        return zero

    px = ac.iloc[:asof + 1].tail(300).astype(float)
    if len(px) < 66:
        return zero
    p = px.values
    nrow, ncol = p.shape
    cur = p[-1]

    cnt = np.sum(np.isfinite(p), axis=0)
    valid = np.isfinite(cur) & (cur > 3.0) & (cnt >= 60)

    edf = history.get('eligible')
    if isinstance(edf, pd.DataFrame) and len(edf) > 0:
        try:
            er = edf.reindex(columns=tickers)
            row = er.iloc[asof] if asof < len(er) else er.iloc[-1]
            vals = pd.to_numeric(row, errors='coerce')
            elig = (vals == 1).values | vals.isna().values
            if elig.sum() >= 10:
                valid &= elig
        except Exception:
            pass

    with np.errstate(invalid='ignore', divide='ignore'):
        rets = p[1:] / p[:-1] - 1.0
        r21 = p[-1] / p[-22] - 1.0 if nrow > 21 else np.full(ncol, np.nan)
        r126 = p[-1] / p[-127] - 1.0 if nrow > 126 else np.full(ncol, np.nan)
        if nrow >= 253:
            mom = p[-22] / p[-253] - 1.0
        elif nrow >= 127:
            mom = p[-22] / p[-127] - 1.0
        else:
            mom = np.full(ncol, np.nan)
        vol126_d = np.nanstd(rets[-126:], axis=0) if rets.shape[0] > 0 else np.full(ncol, np.nan)
        vol21_d = np.nanstd(rets[-21:], axis=0) if rets.shape[0] > 0 else np.full(ncol, np.nan)

    vol126_d = np.where(np.isfinite(vol126_d) & (vol126_d > 1e-4), vol126_d, np.nan)
    vol21_d = np.where(np.isfinite(vol21_d) & (vol21_d > 1e-4), vol21_d, np.nan)

    valid = valid & np.isfinite(r21) & np.isfinite(r126) & np.isfinite(mom)
    valid = valid & np.isfinite(vol126_d) & np.isfinite(vol21_d)
    valid = valid & (r21 > -0.95) & (r21 < 5.0) & (r126 > -0.98) & (r126 < 20.0) & (mom > -0.98) & (mom < 20.0)

    if valid.sum() < 8:
        return zero

    vdf = history.get('volume')
    cdf = history.get('close')
    if isinstance(vdf, pd.DataFrame):
        try:
            win = dates[max(0, asof - 19):asof + 1]
            v = vdf.reindex(index=win, columns=tickers).values.astype(float)
            if isinstance(cdf, pd.DataFrame):
                c = cdf.reindex(index=win, columns=tickers).values.astype(float)
            else:
                c = ac.reindex(index=win, columns=tickers).values.astype(float)
            with np.errstate(invalid='ignore'):
                adv = np.nanmedian(v * c, axis=0)
            dvv = adv[valid & np.isfinite(adv)]
            if dvv.size >= 10:
                thr = max(1.0e6, float(np.nanpercentile(dvv, 40)))
                liq = valid & np.isfinite(adv) & (adv >= thr)
                if liq.sum() < 8:
                    liq = valid & np.isfinite(adv) & (adv >= 1.0e6)
                if liq.sum() < 8:
                    liq = valid
                valid = liq
        except Exception:
            pass

    if valid.sum() < 8:
        return zero

    mean_r21 = float(np.nanmean(r21[valid]))
    mean_r126 = float(np.nanmean(r126[valid]))
    mean_mom = float(np.nanmean(mom[valid]))

    r21_res = r21 - mean_r21
    r126_res = r126 - mean_r126
    mom_res = mom - mean_mom

    with np.errstate(invalid='ignore', divide='ignore'):
        rev_raw = -r21_res / (vol21_d * math.sqrt(21.0))
        longrev_raw = -r126_res / (vol126_d * math.sqrt(126.0))
        mom_raw = mom_res / (vol126_d * math.sqrt(252.0))
        lowvol_raw = -vol126_d

    z_rev = _zscore(rev_raw, valid)
    z_longrev = _zscore(longrev_raw, valid)
    z_mom = _zscore(mom_raw, valid)
    z_lv = _zscore(lowvol_raw, valid)

    score = 0.45 * z_rev + 0.30 * z_mom + 0.15 * z_longrev + 0.10 * z_lv
    score = np.where(valid & np.isfinite(score), score, np.nan)

    idx = np.where(np.isfinite(score))[0]
    if idx.size < 8:
        return zero

    order = idx[np.argsort(-score[idx])]
    n_side = min(50, idx.size // 2)
    if n_side < 2:
        return zero

    w = min(0.02, 0.5 / float(n_side))
    wv = np.zeros(ncol, dtype=float)
    wv[order[:n_side]] = w
    wv[order[-n_side:]] = -w
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