import numpy as np
import pandas as pd
import math


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)

    zero = pd.Series(0.0, index=tickers, dtype=float)

    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None:
        return zero

    try:
        ac = ac.reindex(columns=tickers)
    except Exception:
        return zero

    dates = ac.index
    if len(dates) < 70:
        return zero

    sub = ac.iloc[-260:]
    vals = sub.values.astype(float)
    last = vals[-1]

    valid = np.isfinite(last) & (last > 2.0)

    elig = history.get('eligible')
    if elig is not None:
        try:
            e = elig.reindex(index=dates, columns=tickers).iloc[-1]
            e = e.fillna(False).astype(bool).values
            valid = valid & e
        except Exception:
            pass

    vol_df = history.get('volume')
    if vol_df is not None:
        try:
            v = vol_df.reindex(index=dates, columns=tickers).iloc[-60:].values.astype(float)
            c = history.get('close')
            if c is not None:
                c = c.reindex(index=dates, columns=tickers).iloc[-60:].values.astype(float)
            else:
                c = sub.iloc[-60:].values.astype(float)
            adv = np.nanmean(c * v, axis=0)
            valid = valid & np.isfinite(adv) & (adv > 100000.0)
            if valid.sum() >= 20:
                thr = np.nanpercentile(adv[valid], 40.0)
                valid = valid & (adv >= thr)
        except Exception:
            pass

    if valid.sum() < 10:
        return zero

    ret = vals[1:] / vals[:-1] - 1.0

    if len(vals) < 22:
        return zero

    r21 = vals[-1] / vals[-22] - 1.0
    if len(vals) >= 6:
        r5 = vals[-1] / vals[-6] - 1.0
    else:
        r5 = r21.copy()
    r5 = np.where(np.isfinite(r5), r5, r21)

    win60 = min(60, ret.shape[0])
    vol60 = np.nanstd(ret[-win60:], axis=0) * math.sqrt(252.0)
    vol60 = np.where(np.isfinite(vol60) & (vol60 > 0.005), vol60, np.nan)

    valid = valid & np.isfinite(r21) & np.isfinite(r5) & np.isfinite(vol60)
    if valid.sum() < 10:
        return zero

    market_ret = np.nanmean(ret[-min(250, ret.shape[0]):], axis=1)
    market_ret = np.nan_to_num(market_ret, nan=0.0, posinf=0.0, neginf=0.0)
    mkt_level = np.cumprod(1.0 + market_ret)

    if len(mkt_level) >= 200:
        ma200 = np.nanmean(mkt_level[-200:])
    else:
        ma200 = np.nanmean(mkt_level)
    mkt_now = mkt_level[-1]

    if len(mkt_level) >= 22:
        m21 = mkt_now / mkt_level[-22] - 1.0
    else:
        m21 = np.nanmean(r21[valid])

    if len(mkt_level) >= 61:
        m60 = mkt_now / mkt_level[-61] - 1.0
    else:
        m60 = 0.0

    mvol = np.nanstd(market_ret[-min(21, len(market_ret)):]) * math.sqrt(252.0)
    if not np.isfinite(mvol) or mvol < 1e-6:
        mvol = 0.15

    disp = np.nanstd(r21[valid])
    if not np.isfinite(disp) or disp < 1e-6:
        disp = 0.10

    scale = 1.0

    scale *= np.clip(0.20 / max(mvol, 0.08), 0.20, 1.0)
    scale *= np.clip(0.22 / max(disp, 0.08), 0.30, 1.0)

    if np.isfinite(mkt_now) and np.isfinite(ma200) and mkt_now < ma200:
        scale *= 0.30

    if np.isfinite(m21):
        if m21 > 0.07:
            scale *= 0.50
        if m21 > 0.10:
            scale *= 0.0
        if m21 > 0.04 and m60 < 0.0:
            scale *= 0.30

    scale = float(np.clip(scale, 0.0, 1.0))
    if scale < 0.10:
        return zero

    idx = np.where(valid)[0]
    if len(idx) < 10:
        return zero

    mu21 = np.nanmean(r21[idx])
    mu5 = np.nanmean(r5[idx])

    z21 = (r21[idx] - mu21) / (vol60[idx] * math.sqrt(21.0))
    z5 = (r5[idx] - mu5) / (vol60[idx] * math.sqrt(5.0))
    score = -(0.70 * z21 + 0.30 * z5)

    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    order = np.argsort(score)
    m = min(30, len(idx) // 2)
    if m < 1:
        return zero

    long_sel = idx[order[-m:]]
    short_sel = idx[order[:m]]

    gross_target = 0.80 * scale
    wt = min(0.02, (gross_target / 2.0) / float(m))
    if wt <= 0.0:
        return zero

    w = np.zeros(n, dtype=float)
    w[long_sel] = wt
    w[short_sel] = -wt

    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, -0.02, 0.02)

    gross = np.abs(w).sum()
    if gross > 1.0:
        w = w * (1.0 / gross)

    return pd.Series(w, index=tickers, dtype=float)