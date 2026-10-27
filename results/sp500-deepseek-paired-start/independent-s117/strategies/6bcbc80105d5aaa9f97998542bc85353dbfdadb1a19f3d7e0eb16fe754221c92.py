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
    total = len(ac_full.index)
    start = max(0, total - 420)
    try:
        ac = ac_full.iloc[start:].reindex(columns=tickers)
    except Exception:
        return zero
    dates = ac.index
    mrows = len(dates)
    if mrows < 70:
        return zero
    last_pos = mrows - 1
    try:
        months = dates.month.values
    except Exception:
        months = np.array([d.month for d in dates])
    month_starts = [i for i in range(1, mrows) if months[i] != months[i-1]]
    if len(month_starts) == 0:
        return zero
    signal_pos = month_starts[-1]
    if signal_pos < 60:
        return zero
    close_all = ac.values.astype(float)
    last_close = close_all[last_pos]
    valid = np.isfinite(last_close) & (last_close > 1.0)
    elig = history.get('eligible')
    if elig is not None:
        try:
            e = elig.iloc[start:].reindex(columns=tickers, index=dates)
            le = e.iloc[last_pos].values
            if len(le) > 0 and not pd.isna(le).all():
                valid &= _to_bool_array(le)
        except Exception:
            pass
    vol_field = history.get('volume')
    if vol_field is not None:
        try:
            v = vol_field.iloc[start:].reindex(columns=tickers, index=dates).values.astype(float)
            Lq = min(60, last_pos + 1)
            if Lq >= 20:
                cwin = close_all[last_pos - Lq + 1:last_pos + 1]
                vwin = v[last_pos - Lq + 1:last_pos + 1]
                dv = np.nanmean(cwin * vwin, axis=0)
                valid &= np.isfinite(dv) & (dv > 0)
                if valid.sum() >= 40:
                    thresh = np.nanpercentile(dv[valid], 30)
                    valid &= dv >= thresh
        except Exception:
            pass
    close_sig = close_all[:signal_pos + 1]
    if close_sig.shape[0] < 60:
        return zero
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
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = close_sig[1:] / close_sig[:-1] - 1.0
    if returns.shape[0] >= 20:
        vwin = returns[-126:] if returns.shape[0] >= 126 else returns
        stock_vol = np.nanstd(vwin, axis=0)
    else:
        stock_vol = np.full(n, np.nan, dtype=float)
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
    valid &= np.isfinite(ret) & np.isfinite(stock_vol) & (stock_vol > 1e-6) & np.isfinite(high_prox) & np.isfinite(trend)
    idx = np.where(valid)[0]
    if len(idx) < 10:
        return zero
    mom = ret[idx]
    volv = stock_vol[idx]
    hp = high_prox[idx]
    tr = trend[idx]
    with np.errstate(divide='ignore', invalid='ignore'):
        risk_adj = mom / np.maximum(volv, 1e-4)
    risk_adj = np.nan_to_num(risk_adj, nan=0.0, posinf=0.0, neginf=0.0)
    z_mom = _pct_rank(risk_adj)
    z_high = _pct_rank(hp)
    z_trend = _pct_rank(tr)
    z_lowvol = _pct_rank(-volv)
    if returns.shape[0] >= 60:
        rwin = returns[-200:] if returns.shape[0] >= 200 else returns
        market_ret = np.nanmean(rwin, axis=1)
        market_ret = np.nan_to_num(market_ret, nan=0.0, posinf=0.0, neginf=0.0)
        level = np.cumprod(1.0 + market_ret)
        base = np.nanmean(level[-200:]) if len(level) >= 200 else np.nanmean(level)
        if np.isfinite(base) and base > 0:
            mtrend = level[-1] / base - 1.0
        else:
            mtrend = 0.0
        mvol = np.nanstd(market_ret[-60:]) * math.sqrt(252.0)
        if not np.isfinite(mvol):
            mvol = 0.20
        recent_dd = 0.0
        if len(level) >= 20:
            peak = np.nanmax(level[-60:]) if len(level) >= 60 else np.nanmax(level)
            if np.isfinite(peak) and peak > 0:
                recent_dd = level[-1] / peak - 1.0
    else:
        mtrend = 0.0
        mvol = 0.20
        recent_dd = 0.0
    defensive = (mtrend < 0.0) or (mvol > 0.32) or (recent_dd < -0.08)
    if defensive:
        score = 0.35 * z_mom + 0.20 * z_high + 0.10 * z_trend + 0.35 * z_lowvol
    else:
        score = 0.45 * z_mom + 0.25 * z_high + 0.15 * z_trend + 0.15 * z_lowvol
    scale = 1.0
    if mtrend < 0.0:
        scale *= 0.75
    if mvol > 0.30:
        scale *= 0.85
    if mvol > 0.45:
        scale *= 0.80
    if recent_dd < -0.08:
        scale *= 0.80
    if recent_dd < -0.15:
        scale *= 0.75
    scale = float(np.clip(scale, 0.30, 1.0))
    m = min(50, len(idx) // 2)
    if m < 5:
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