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


def _z(s):
    s = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    weights = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return weights

    adj = _get(history, 'adjusted_close', tickers)
    raw = _get(history, 'close', tickers)
    vol_df = _get(history, 'volume', tickers)
    elig = _get(history, 'eligible', tickers)

    close = adj if adj is not None and adj.notna().any().any() else raw
    if close is None or close.empty or not close.notna().any().any():
        return weights
    if raw is None or raw.empty or not raw.notna().any().any():
        raw = close

    n = len(close)
    if n < 260:
        return weights

    dates = close.index
    cur = dates[-1]
    qstart_month = ((cur.month - 1) // 3) * 3 + 1
    qstart = pd.Timestamp(year=cur.year, month=qstart_month, day=1)
    pos = np.where(dates < qstart)[0]
    if len(pos) == 0:
        return weights
    a = int(pos[-1])
    if a < 252:
        return weights

    c = close.iloc[:a + 1]
    r = raw.iloc[:a + 1]
    v = vol_df.iloc[:a + 1] if vol_df is not None else None
    e = elig.iloc[:a + 1] if elig is not None else None

    last = c.iloc[-1].astype(float)
    raw_last = r.iloc[-1].astype(float)
    valid = last.notna() & np.isfinite(last) & (last > 0.0)
    valid = valid & raw_last.notna() & np.isfinite(raw_last) & (raw_last > 3.0)

    if e is not None and len(e) > 0:
        ev = e.iloc[-1].reindex(tickers)
        try:
            ev = ev.fillna(False).astype(bool)
        except Exception:
            ev = ev.fillna(0).astype(float).astype(bool)
        valid = valid & ev

    if valid.sum() < 20:
        valid = last.notna() & np.isfinite(last) & (last > 0.0)
    if valid.sum() < 12:
        return weights

    rets = c.pct_change().replace([np.inf, -np.inf], np.nan)
    if len(rets) < 150:
        return weights

    win = min(252, len(rets) - 1)
    ret_win = rets.iloc[-win:]
    mkt = ret_win.mean(axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
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

    R = ret_win.to_numpy(dtype=float)
    m = mkt.to_numpy(dtype=float)
    b = beta.to_numpy(dtype=float)
    resid_arr = R - m[:, None] * b[None, :]
    resid = pd.DataFrame(resid_arr, index=ret_win.index, columns=tickers)

    rv_win = min(126, len(resid))
    rv = resid.iloc[-rv_win:].std(ddof=0) * math.sqrt(252.0)
    rv = rv.replace([np.inf, -np.inf], np.nan)

    rev_win = min(21, len(resid))
    rev = resid.iloc[-rev_win:].sum()
    rev_vol = resid.iloc[-rev_win:].std(ddof=0) * math.sqrt(252.0)
    rev_adj = rev / rev_vol.clip(lower=0.01)
    rev_adj = rev_adj.replace([np.inf, -np.inf], np.nan)

    hi_win = min(252, len(c))
    hi = c.iloc[-hi_win:].max()
    trend = last / hi - 1.0
    trend = trend.replace([np.inf, -np.inf], np.nan)

    if v is not None and not v.empty:
        dv = (r * v).iloc[-min(60, len(r)):].median()
        dv = dv.replace([np.inf, -np.inf], np.nan)
    else:
        dv = pd.Series(np.nan, index=tickers)

    logrv = np.log(rv.clip(lower=0.005))
    score = (-0.60 * _z(logrv)
             - 0.25 * _z(rev_adj)
             + 0.15 * _z(trend))

    valid = valid & np.isfinite(score) & np.isfinite(rv) & (rv > 0.005)

    if dv.notna().any():
        dvv = dv.replace([np.inf, -np.inf], np.nan)
        q = dvv.quantile(0.20)
        qv = float(q) if np.isfinite(q) else 0.0
        th = max(1000000.0, qv)
        valid_dv = valid & (dvv >= th)
        if valid_dv.sum() >= 24:
            valid = valid_dv
        else:
            valid = valid & (dvv >= max(250000.0, qv))

    basic = last.notna() & np.isfinite(last) & (last > 0.0) & np.isfinite(score)
    if valid.sum() < 24:
        valid = basic
    if valid.sum() < 12:
        return weights

    idx = score[valid].dropna().index
    if len(idx) < 12:
        return weights

    s = score.loc[idx]
    b = beta.loc[idx].astype(float)

    if len(idx) >= 90:
        nb = 5
    elif len(idx) >= 60:
        nb = 4
    elif len(idx) >= 40:
        nb = 3
    else:
        nb = 2

    try:
        bq = pd.qcut(b, nb, labels=False, duplicates='drop')
        bq = pd.Series(bq, index=b.index).fillna(0)
        actual_nb = int(bq.max()) + 1 if bq.notna().any() else 1
    except Exception:
        bq = pd.Series(0, index=b.index)
        actual_nb = 1

    target_side = min(40, len(idx) // 2)
    per_bucket = max(1, target_side // max(1, actual_nb))
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

    longs = list(dict.fromkeys(longs))
    shorts = list(dict.fromkeys(shorts))
    mside = min(len(longs), len(shorts))

    if mside < 5:
        ordered = s.sort_values()
        k = min(40, len(ordered) // 2)
        if k < 5:
            return weights
        shorts = list(ordered.index[:k])
        longs = list(ordered.index[-k:])
        mside = min(len(longs), len(shorts))

    if mside < 1:
        return weights

    longs = longs[:mside]
    shorts = shorts[:mside]
    wt = min(0.02, 0.5 / float(mside))
    if not np.isfinite(wt) or wt <= 0.0:
        return weights

    weights.loc[longs] = wt
    weights.loc[shorts] = -wt
    return weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
