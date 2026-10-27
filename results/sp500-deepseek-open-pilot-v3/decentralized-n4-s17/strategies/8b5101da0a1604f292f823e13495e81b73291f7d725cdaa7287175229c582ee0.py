import numpy as np
import pandas as pd


def _zeros(universe):
    return pd.Series(0.0, index=universe, dtype=float)


def _zscore(s):
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 2:
        return pd.Series(0.0, index=s.index, dtype=float)
    mu = float(s.mean())
    sd = float(s.std(ddof=0))
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=s.index, dtype=float)
    return ((s - mu) / sd).clip(-3.0, 3.0)


def generate_weights(history, universe):
    u = pd.Index(universe)
    if len(u) == 0:
        return pd.Series(dtype=float)
    empty = _zeros(u)

    adj = history.get('adjusted_close')
    if not isinstance(adj, pd.DataFrame):
        adj = history.get('close')
    if not isinstance(adj, pd.DataFrame) or adj.empty:
        frames = {}
        for t in u:
            d = history.get(t)
            if isinstance(d, pd.DataFrame) and 'adjusted_close' in d.columns:
                frames[t] = d['adjusted_close']
        if not frames:
            return empty
        adj = pd.DataFrame(frames)

    adj = adj.reindex(columns=u)
    if len(adj) < 40:
        return empty

    idx = adj.index
    last_pos = len(idx) - 1
    fridays = np.where(idx.dayofweek == 4)[0]
    if len(fridays) > 0:
        base_pos = int(fridays[-1])
    else:
        base_pos = last_pos
    if base_pos < 130:
        return empty

    start = max(0, base_pos - 259)
    p = adj.iloc[start:base_pos + 1].copy()
    if len(p) < 130:
        return empty
    p = p.ffill()

    lastp = p.iloc[-1]
    valid = lastp.notna() & (lastp.to_numpy(dtype=float) > 1.0)
    valid = pd.Series(valid, index=u)
    valid &= p.notna().sum() >= min(120, len(p))

    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame):
        e = elig.reindex(index=idx, columns=u)
        if base_pos < len(e):
            es = e.iloc[base_pos]
        else:
            es = e.iloc[-1]
        valid &= es.fillna(False).astype(bool)

    vol = history.get('volume')
    close = history.get('close')
    if isinstance(vol, pd.DataFrame):
        v = vol.reindex(index=idx, columns=u).iloc[start:base_pos + 1]
        if isinstance(close, pd.DataFrame):
            c = close.reindex(index=idx, columns=u).iloc[start:base_pos + 1]
        else:
            c = p
        if len(v) >= 5:
            vv = v.iloc[-20:]
            cc = c.iloc[-20:]
            dollar = (vv * cc).median()
            valid &= dollar.notna() & (dollar.to_numpy(dtype=float) > 1.0e6)

    if int(valid.sum()) < 10:
        return empty

    rets = p.pct_change()
    vol20 = rets.iloc[-20:].std(ddof=0)
    vol20 = vol20.replace([np.inf, -np.inf], np.nan)
    vol20 = vol20.where(vol20 > 1e-8)

    if len(p) >= 6:
        r5 = p.iloc[-1] / p.iloc[-6] - 1.0
    else:
        r5 = pd.Series(np.nan, index=u)
    if len(p) >= 22:
        r21 = p.iloc[-1] / p.iloc[-22] - 1.0
    else:
        r21 = pd.Series(np.nan, index=u)

    rev5 = -r5 / (vol20 * np.sqrt(5.0))
    rev21 = -r21 / (vol20 * np.sqrt(21.0))
    rev5 = rev5.replace([np.inf, -np.inf], np.nan)
    rev21 = rev21.replace([np.inf, -np.inf], np.nan)

    score = 0.7 * _zscore(rev5) + 0.3 * _zscore(rev21)
    score = score.replace([np.inf, -np.inf], np.nan)
    score = score.where(valid)

    cand = score.dropna()
    if len(cand) < 10:
        return empty

    beta = pd.Series(1.0, index=u, dtype=float)
    if len(rets) >= 30:
        win = rets.iloc[-120:]
        mkt = win.mean(axis=1, skipna=True)
        mkt_d = mkt - mkt.mean()
        denom = float((mkt_d * mkt_d).sum())
        if np.isfinite(denom) and denom > 1e-12:
            demean = win.sub(win.mean(axis=0), axis=1)
            cov = (demean.mul(mkt_d, axis=0)).sum() / denom
            beta = pd.Series(cov, index=u, dtype=float)
    beta = beta.replace([np.inf, -np.inf], np.nan)
    if beta.notna().any():
        beta = beta.fillna(float(beta.median()))
    else:
        beta = beta.fillna(1.0)
    beta = beta.reindex(u).fillna(1.0)

    cand_beta = beta.loc[cand.index]
    br = cand_beta.rank(method='first', pct=True)
    buckets = np.minimum((br.to_numpy(dtype=float) * 5.0).astype(int), 4)

    long_names = []
    short_names = []
    for b in range(5):
        names = cand.index[buckets == b]
        if len(names) < 2:
            continue
        ordered = cand.loc[names].sort_values(ascending=False, kind='mergesort')
        n = min(5, len(ordered) // 2)
        if n < 1:
            continue
        long_names.extend(list(ordered.index[:n]))
        short_names.extend(list(ordered.index[-n:]))

    if len(long_names) == 0 or len(long_names) != len(short_names):
        return empty

    side_count = len(long_names)
    w = min(0.02, 0.5 / float(side_count))
    out = pd.Series(0.0, index=u, dtype=float)
    for t in long_names:
        if t in out.index:
            out.loc[t] = w
    for t in short_names:
        if t in out.index:
            out.loc[t] = -w
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out