import numpy as np
import pandas as pd
import math

def _pct_rank(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x
    if n == 1:
        return np.zeros(1, dtype=float)
    order = np.argsort(x, kind='mergesort')
    xs = x[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        avg = (i + j - 1) / 2.0
        ranks[order[i:j]] = avg
        i = j
    return ranks / (n - 1.0) - 0.5

def _to_bool_array(v):
    n = len(v)
    out = np.zeros(n, dtype=bool)
    for i, x in enumerate(v):
        try:
            if pd.isna(x):
                continue
        except Exception:
            pass
        if isinstance(x, (bool, np.bool_)):
            out[i] = bool(x)
        else:
            try:
                out[i] = (float(x) != 0.0)
            except Exception:
                try:
                    out[i] = bool(x)
                except Exception:
                    out[i] = False
    return out

def _mom_from_close(close_sig, lookback, skip):
    if close_sig.shape[0] <= lookback + skip:
        return None
    num = close_sig[-1 - skip]
    den = close_sig[-1 - lookback]
    with np.errstate(divide='ignore', invalid='ignore'):
        r = num / den - 1.0
    return r

def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(np.zeros(n, dtype=float), index=tickers)
    ac_full = history.get('adjusted_close')
    if ac_full is None or len(ac_full.index) == 0:
        return zero
    try:
        ac_full = ac_full.reindex(columns=tickers)
    except Exception:
        return zero
    dates = ac_full.index
    mrows = len(dates)
    if mrows < 80:
        return zero
    try:
        months = dates.month.values
    except Exception:
        months = np.array([d.month for d in dates])
    month_starts = [i for i in range(1, mrows) if months[i] != months[i - 1]]
    if len(month_starts) == 0:
        return zero
    signal_pos = month_starts[-1]
    if signal_pos < 130:
        return zero
    last_pos = mrows - 1
    start = max(0, signal_pos - 400)
    close_all = ac_full.values.astype(float)
    close_sig = close_all[start:signal_pos + 1]
    if close_sig.shape[0] < 130:
        return zero
    last_close = close_all[last_pos]
    valid = np.isfinite(last_close) & (last_close > 1.0)
    elig = history.get('eligible')
    if elig is not None:
        try:
            e = elig.reindex(index=dates, columns=tickers)
            le = e.iloc[last_pos].values
            if len(le) > 0 and not pd.isna(le).all():
                valid &= _to_bool_array(le)
        except Exception:
            pass
    vol_field = history.get('volume')
    if vol_field is not None:
        try:
            v = vol_field.reindex(index=dates, columns=tickers).values.astype(float)
            Lq = min(60, last_pos + 1)
            if Lq >= 20:
                cwin = close_all[last_pos - Lq + 1:last_pos + 1]
                vwin = v[last_pos - Lq + 1:last_pos + 1]
                dv = np.nanmean(cwin * vwin, axis=0)
                valid &= np.isfinite(dv) & (dv > 0)
                if valid.sum() >= 40:
                    thresh = np.nanpercentile(dv[valid], 20)
                    valid &= dv >= thresh
        except Exception:
            pass
    with np.errstate(divide='ignore', invalid='ignore'):
        rets = close_sig[1:] / close_sig[:-1] - 1.0
    if rets.shape[0] < 60:
        return zero
    Lv = min(120, rets.shape[0])
    stock_vol = np.nanstd(rets[-Lv:], axis=0)
    hwin = close_sig[-252:] if close_sig.shape[0] >= 252 else close_sig
    with np.errstate(invalid='ignore'):
        max_close = np.nanmax(hwin, axis=0)
        high_prox = close_sig[-1] / max_close - 1.0
        high_prox = np.where(np.isfinite(max_close) & (max_close > 0), high_prox, np.nan)
    twin = close_sig[-200:] if close_sig.shape[0] >= 200 else close_sig
    with np.errstate(invalid='ignore'):
        sma = np.nanmean(twin, axis=0)
        trend = close_sig[-1] / sma - 1.0
        trend = np.where(np.isfinite(sma) & (sma > 0), trend, np.nan)
    ret = np.full(n, np.nan, dtype=float)
    r = _mom_from_close(close_sig, 252, 21)
    if r is not None:
        ret = r
    r = _mom_from_close(close_sig, 126, 10)
    if r is not None:
        ret = np.where(np.isfinite(ret), ret, r)
    r = _mom_from_close(close_sig, 63, 5)
    if r is not None:
        ret = np.where(np.isfinite(ret), ret, r)
    mret = np.nanmean(rets[-252:], axis=1) if rets.shape[0] >= 252 else np.nanmean(rets, axis=1)
    mret = np.nan_to_num(mret, nan=0.0, posinf=0.0, neginf=0.0)
    level = np.cumprod(1.0 + mret)
    base = np.nanmean(level[-200:]) if len(level) >= 200 else np.nanmean(level)
    mtrend = level[-1] / base - 1.0 if np.isfinite(base) and base > 0 else 0.0
    if len(mret) >= 60:
        mvol = np.nanstd(mret[-60:]) * math.sqrt(252.0)
    else:
        mvol = np.nanstd(mret) * math.sqrt(252.0)
    if not np.isfinite(mvol):
        mvol = 0.20
    peak = np.nanmax(level[-60:]) if len(level) >= 60 else np.nanmax(level)
    recent_dd = level[-1] / peak - 1.0 if np.isfinite(peak) and peak > 0 else 0.0
    beta = np.ones(n, dtype=float)
    Lb = min(120, rets.shape[0])
    if Lb >= 40:
        rwin = rets[-Lb:]
        mwin = mret[-Lb:]
        mean_m = np.nanmean(mwin)
        var_m = np.nanmean((mwin - mean_m) ** 2)
        if np.isfinite(var_m) and var_m > 1e-12:
            mean_s = np.nanmean(rwin, axis=0)
            cov = np.nanmean((rwin - mean_s) * (mwin[:, None] - mean_m), axis=0)
            b = cov / var_m
            beta = np.where(np.isfinite(b), b, 1.0)
    valid &= np.isfinite(ret) & np.isfinite(stock_vol) & (stock_vol > 1e-6) & np.isfinite(high_prox) & np.isfinite(trend) & np.isfinite(beta)
    idx = np.where(valid)[0]
    if len(idx) < 16:
        return zero
    mom = ret[idx]
    volv = stock_vol[idx]
    hp = high_prox[idx]
    tr = trend[idx]
    bet = beta[idx]
    z_lowvol = _pct_rank(-volv)
    z_lowbeta = _pct_rank(-bet)
    z_highvol = _pct_rank(volv)
    z_highbeta = _pct_rank(bet)
    z_mom = _pct_rank(mom)
    z_trend = _pct_rank(tr)
    z_high = _pct_rank(hp)
    score_on = 0.30 * z_highbeta + 0.25 * z_mom + 0.20 * z_trend + 0.15 * z_high + 0.10 * z_highvol
    score_off = 0.35 * z_lowvol + 0.25 * z_lowbeta + 0.20 * z_trend + 0.20 * z_high
    risk_on = (mtrend > 0.03) and (mvol < 0.22) and (recent_dd > -0.06)
    risk_off = (mtrend < -0.02) or (mvol > 0.28) or (recent_dd < -0.10)
    if risk_on:
        score = score_on
        scale = 0.80
    elif risk_off:
        score = score_off
        scale = 1.00
    else:
        score = 0.50 * score_on + 0.50 * score_off
        scale = 0.70
    if mvol > 0.35:
        scale *= 0.85
    if recent_dd < -0.15:
        scale *= 0.85
    scale = float(np.clip(scale, 0.30, 1.00))
    m = min(30, len(idx) // 2)
    if m < 8:
        return zero
    order = np.argsort(score, kind='mergesort')
    long_sel = idx[order[-m:]]
    short_sel = idx[order[:m]]
    wt = min(0.02, 0.5 * scale / m)
    if wt <= 0.0 or not np.isfinite(wt):
        return zero
    w = np.zeros(n, dtype=float)
    w[long_sel] = wt
    w[short_sel] = -wt
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, -0.02, 0.02)
    return pd.Series(w, index=tickers)