import numpy as np
import pandas as pd
import math

def _get(history, key, tickers):
    if not isinstance(history, dict):
        return None
    df = history.get(key)
    if df is None:
        return None
    try:
        return df.reindex(columns=tickers)
    except Exception:
        return None

def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    weights = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return weights
    adj = _get(history, 'adjusted_close', tickers)
    raw = _get(history, 'close', tickers)
    vol_df = _get(history, 'volume', tickers)
    elig_df = _get(history, 'eligible', tickers)
    close = adj
    if close is None or close.empty or not close.notna().any().any():
        close = raw
    if close is None or close.empty or not close.notna().any().any():
        return weights
    if raw is None or raw.empty or not raw.notna().any().any():
        raw = close
    n = len(close)
    if n < 130:
        return weights
    dates = close.index
    cur = dates[-1]
    years = dates.year.values
    months = dates.month.values
    cmask = (years < cur.year) | ((years == cur.year) & (months < cur.month))
    pos = np.where(cmask)[0]
    if len(pos) == 0:
        return weights
    a = int(pos[-1])
    if a < 125:
        return weights
    c = close.iloc[:a+1]
    r = raw.iloc[:a+1]
    v = vol_df.iloc[:a+1] if vol_df is not None else None
    e = elig_df.iloc[:a+1] if elig_df is not None else None
    last = c.iloc[-1].astype(float)
    raw_last = r.iloc[-1].astype(float)
    valid = last.notna() & np.isfinite(last) & (last > 0.0)
    valid = valid & raw_last.notna() & np.isfinite(raw_last) & (raw_last > 2.0)
    if e is not None and len(e) > 0:
        ev = e.iloc[-1].reindex(tickers)
        try:
            ev = ev.fillna(False).astype(bool)
        except Exception:
            ev = ev.fillna(0).astype(float).astype(bool)
        valid = valid & ev
    if valid.sum() < 20:
        valid = last.notna() & np.isfinite(last) & (last > 0.0)
    if valid.sum() < 10:
        return weights
    rets = c / c.shift(1) - 1.0
    rets = rets.replace([np.inf, -np.inf], np.nan)
    if len(rets) < 25:
        return weights
    win = min(252, len(rets) - 1)
    if win < 30:
        return weights
    ret_win = rets.iloc[-win:]
    mkt = ret_win.mean(axis=1, skipna=True)
    mkt = mkt.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mkt_dm = mkt - mkt.mean()
    mkt_var = float((mkt_dm * mkt_dm).mean())
    if not np.isfinite(mkt_var) or mkt_var < 1e-12:
        beta = pd.Series(1.0, index=tickers)
    else:
        stock_mean = ret_win.mean()
        cov = ret_win.sub(stock_mean, axis=1).mul(mkt_dm, axis=0).mean()
        beta = cov / mkt_var
        beta = beta.replace([np.inf, -np.inf], np.nan)
        med = beta.median()
        if not np.isfinite(med):
            med = 1.0
        beta = beta.fillna(med).clip(0.1, 4.0)
    rev_win = rets.iloc[-21:]
    if len(rev_win) < 10:
        return weights
    mkt_rev = rev_win.mean(axis=1, skipna=True).fillna(0.0)
    sum_ret = rev_win.sum()
    resid = sum_ret - beta * float(mkt_rev.sum())
    vol_short = rev_win.std(ddof=0) * math.sqrt(252.0)
    vol_short = vol_short.replace([np.inf, -np.inf], np.nan)
    score = -resid / vol_short.clip(lower=0.01)
    if v is not None and not v.empty:
        rv = r * v
        dv = rv.iloc[-min(60, len(rv)):].median()
        dv = dv.replace([np.inf, -np.inf], np.nan)
    else:
        dv = pd.Series(np.nan, index=tickers)
    valid = valid & np.isfinite(score) & np.isfinite(vol_short) & (vol_short > 0.01)
    if dv.notna().any():
        dvv = dv.replace([np.inf, -np.inf], np.nan)
        q = dvv.quantile(0.2)
        qv = float(q) if np.isfinite(q) else 0.0
        threshold = max(1000000.0, qv)
        valid_dv = valid & (dvv >= threshold)
        if valid_dv.sum() >= 20:
            valid = valid_dv
        else:
            threshold2 = max(250000.0, qv)
            valid = valid & (dvv >= threshold2)
    basic = last.notna() & np.isfinite(last) & (last > 0.0) & np.isfinite(score)
    if valid.sum() < 20:
        valid = basic
    if valid.sum() < 10:
        return weights
    idx = score[valid].dropna().index
    if len(idx) < 10:
        return weights
    s = score.loc[idx]
    b = beta.loc[idx].astype(float)
    if len(idx) >= 50:
        nb = 5
    elif len(idx) >= 30:
        nb = 4
    else:
        nb = 2
    try:
        bq = pd.qcut(b, nb, labels=False, duplicates='drop')
        bq = pd.Series(bq, index=b.index)
        if bq.isna().any():
            bq = bq.fillna(0)
        actual_nb = int(bq.max()) + 1 if bq.notna().any() else 1
    except Exception:
        bq = pd.Series(0, index=b.index)
        actual_nb = 1
    per_bucket = max(1, int(np.floor(25.0 / max(1, actual_nb))))
    longs = []
    shorts = []
    for g in range(actual_nb):
        cand = s[bq == g].sort_values()
        if len(cand) < 2:
            continue
        k = min(per_bucket, len(cand) // 2)
        if k < 1:
            continue
        shorts.extend(list(cand.index[:k]))
        longs.extend(list(cand.index[-k:]))
    if len(longs) < 5 or len(shorts) < 5:
        ordered = s.sort_values()
        k = min(25, len(ordered) // 2)
        if k < 5:
            return weights
        shorts = list(ordered.index[:k])
        longs = list(ordered.index[-k:])
    longs = list(dict.fromkeys(longs))
    shorts = list(dict.fromkeys(shorts))
    m = min(len(longs), len(shorts))
    if m < 1:
        return weights
    longs = longs[:m]
    shorts = shorts[:m]
    wt = min(0.02, 0.5 / float(m))
    if not np.isfinite(wt) or wt <= 0.0:
        return weights
    weights.loc[longs] = wt
    weights.loc[shorts] = -wt
    return weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
