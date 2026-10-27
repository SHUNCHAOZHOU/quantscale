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
    if not isinstance(ac, pd.DataFrame) or ac.empty:
        ac = history.get('close')
    if not isinstance(ac, pd.DataFrame) or ac.empty:
        return zero
    try:
        ac = ac.reindex(columns=tickers).astype(float)
    except Exception:
        return zero
    if ac.shape[0] < 90:
        return zero

    dates = ac.index
    last_pos = ac.shape[0] - 1

    sig_pos = last_pos
    try:
        per = pd.Period(dates[last_pos], freq='M')
        pm = np.array([pd.Period(d, freq='M') != per for d in dates])
        if pm.any():
            sig_pos = int(np.where(pm)[0][-1])
    except Exception:
        sig_pos = last_pos
    if sig_pos < 45:
        sig_pos = last_pos
    if sig_pos < 30:
        return zero

    p = ac.iloc[:sig_pos + 1]
    last_sig = p.iloc[-1].values
    close_now = ac.iloc[last_pos].values

    valid = np.isfinite(close_now) & (close_now > 1.0)
    valid &= np.isfinite(last_sig) & (last_sig > 1.0)
    counts = p.notna().sum(axis=0).values
    valid &= counts >= 60

    elig_df = history.get('eligible')
    if isinstance(elig_df, pd.DataFrame):
        try:
            e = elig_df.reindex(index=ac.index, columns=tickers).iloc[last_pos].values
            e = np.array([(x is True) or (x == 1) for x in e], dtype=bool)
            valid &= e
        except Exception:
            pass

    rets = p.pct_change(fill_method=None)
    vol60 = rets.iloc[-60:].std(ddof=0).values * math.sqrt(252.0)
    vol60 = np.where(np.isfinite(vol60) & (vol60 > 1e-4), vol60, np.nan)
    valid &= np.isfinite(vol60)

    vol_df = history.get('volume')
    raw = history.get('close')
    if not isinstance(raw, pd.DataFrame):
        raw = ac
    try:
        raw = raw.reindex(columns=tickers).astype(float)
    except Exception:
        raw = ac
    if isinstance(vol_df, pd.DataFrame):
        try:
            v = vol_df.reindex(index=ac.index, columns=tickers).iloc[:sig_pos + 1].astype(float)
            r = raw.iloc[:sig_pos + 1]
            win = min(60, len(p))
            dv = (r.iloc[-win:] * v.iloc[-win:]).median(axis=0).values
            dv_ok = np.isfinite(dv) & (dv > 0)
            if dv_ok.sum() >= 20:
                thr = max(5e5, float(np.nanpercentile(dv[dv_ok], 20)))
                valid &= dv_ok & (dv >= thr)
        except Exception:
            pass

    ew_ret = rets.mean(axis=1).fillna(0.0)
    ew_level = (1.0 + ew_ret).cumprod()
    Lp = len(p)

    def mkt_over(offset, look):
        if Lp > offset + look:
            return float(ew_level.iloc[-1 - offset] / ew_level.iloc[-1 - offset - look] - 1.0)
        return np.nan

    mk12 = mkt_over(21, 231)
    mk6 = mkt_over(21, 105)

    def skip_ret(offset, look):
        if Lp > offset + look:
            return p.iloc[-1 - offset].values / p.iloc[-1 - offset - look].values - 1.0
        return np.full(n, np.nan)

    m12 = skip_ret(21, 231)
    m6 = skip_ret(21, 105)

    wlen = min(200, len(rets) - 1)
    R = rets.iloc[-wlen:].values
    M = rets.mean(axis=1).values[-wlen:]
    mask = np.isfinite(R) & np.isfinite(M)[:, None]
    nobs = mask.sum(axis=0)
    Rm = np.where(mask, R, 0.0)
    Mm = np.where(mask, M[:, None], 0.0)
    mR = Rm.sum(axis=0) / np.maximum(nobs, 1)
    mM = Mm.sum(axis=0) / np.maximum(nobs, 1)
    cov = ((Rm - mR) * (Mm - mM)).sum(axis=0) / np.maximum(nobs, 1)
    var = ((Mm - mM) ** 2).sum(axis=0) / np.maximum(nobs, 1)
    beta = np.where((nobs >= 50) & (var > 1e-12), cov / var, 1.0)

    resid12 = (m12 - beta * mk12) if np.isfinite(mk12) else m12
    resid6 = (m6 - beta * mk6) if np.isfinite(mk6) else m6

    max21 = rets.iloc[-21:].max(axis=0).values

    def zs(x):
        x = np.asarray(x, dtype=float)
        sel = valid & np.isfinite(x)
        if sel.sum() < 2:
            return np.zeros(n)
        mu = float(np.mean(x[sel]))
        sd = float(np.std(x[sel]))
        if not np.isfinite(sd) or sd < 1e-12:
            return np.zeros(n)
        return (x - mu) / sd

    z12 = zs(resid12)
    z6 = zs(resid6)
    zv = zs(vol60)
    zx = zs(max21)

    score = 0.45 * z12 + 0.30 * z6 - 0.15 * zv - 0.10 * zx
    score = np.where(np.isfinite(score), score, np.nan)
    finite = valid & np.isfinite(score)
    if finite.sum() < 10:
        return zero

    mask_s = pd.Series(finite, index=tickers)
    s = pd.Series(score, index=tickers).where(mask_s).dropna()
    if len(s) < 10:
        return zero
    n_side = min(25, len(s) // 2)
    if n_side < 3:
        return zero
    ordered = s.sort_values()
    longs = ordered.index[-n_side:]
    shorts = ordered.index[:n_side]

    mv = M[np.isfinite(M)]
    if len(mv) >= 60:
        mvol = float(np.std(mv[-60:], ddof=0) * math.sqrt(252.0))
    else:
        mvol = 0.20
    if not np.isfinite(mvol) or mvol <= 0.01:
        mvol = 0.20
    scale = float(np.clip(0.20 / mvol, 0.4, 1.0))

    w = min(0.02, 0.5 * scale / float(n_side))
    weights = pd.Series(0.0, index=tickers)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.clip(-0.02, 0.02).fillna(0.0)
    return weights.astype(float)