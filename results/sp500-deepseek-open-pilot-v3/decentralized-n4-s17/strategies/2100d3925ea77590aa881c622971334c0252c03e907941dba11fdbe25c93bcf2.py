import numpy as np
import pandas as pd


def _rank_pct(s):
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.rank(method='average', pct=True)


def generate_weights(history, universe):
    u = pd.Index(universe)
    zero = pd.Series(0.0, index=u, dtype=float)
    if len(u) == 0:
        return zero

    adj = history.get('adjusted_close')
    if not isinstance(adj, pd.DataFrame) or adj.empty:
        adj = history.get('close')
    if not isinstance(adj, pd.DataFrame) or adj.empty:
        return zero

    cols = [c for c in u if c in adj.columns]
    if len(cols) < 20:
        return zero

    adj = adj.loc[:, cols].sort_index()
    n = len(adj)
    if n < 260:
        return zero

    last_pos = n - 1
    reb_pos = last_pos - (last_pos % 21)
    if reb_pos < 252:
        return zero

    start = max(0, reb_pos - 260)
    hist = adj.iloc[start:reb_pos + 1].ffill()
    if len(hist) < 253:
        return zero
    p = hist.iloc[-253:].copy()
    p = p.where(p > 0.0)

    last = p.iloc[-1]
    valid = last.notna() & np.isfinite(last) & (last > 0.5)
    valid = valid & (p.notna().sum() >= 230)

    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame) and not elig.empty:
        e = elig.reindex(index=p.index, columns=cols).ffill().iloc[-1]
        valid = valid & e.fillna(False).astype(bool)

    vol = history.get('volume')
    close = history.get('close')
    if isinstance(vol, pd.DataFrame) and not vol.empty:
        v = vol.reindex(index=p.index, columns=cols).ffill().iloc[-20:]
        if isinstance(close, pd.DataFrame) and not close.empty:
            c = close.reindex(index=p.index, columns=cols).ffill().iloc[-20:]
        else:
            c = p.iloc[-20:]
        dv = (v * c).median()
        valid = valid & dv.notna() & np.isfinite(dv) & (dv > 1.0e6)

    keep_cols = valid[valid].index
    if len(keep_cols) < 20:
        return zero

    pv = p.loc[:, keep_cols]
    logp = np.log(pv)
    rets = logp.diff().iloc[1:]

    counts = rets.notna().sum()
    keep2 = counts >= 200
    keep_cols = keep_cols[keep2]
    if len(keep_cols) < 20:
        return zero

    rets = rets.loc[:, keep_cols]
    mkt = rets.mean(axis=1)
    resid = rets.sub(mkt, axis=0)
    volr = resid.std(ddof=0)
    volr = volr.replace([np.inf, -np.inf], np.nan)
    volr = volr.where(volr > 1e-8)
    resid0 = resid.fillna(0.0)

    n_all = len(resid0)
    n12 = max(1, n_all - 21)
    if n_all > 21:
        mom12 = resid0.iloc[:-21].sum(axis=0)
    else:
        mom12 = pd.Series(np.nan, index=keep_cols)

    if n_all >= 126:
        mom6 = resid0.iloc[-126:-21].sum(axis=0)
        n6 = 105
    else:
        mom6 = pd.Series(np.nan, index=keep_cols)
        n6 = 1

    if n_all >= 21:
        rev = resid0.iloc[-21:].sum(axis=0)
        nrev = 21
    else:
        rev = pd.Series(np.nan, index=keep_cols)
        nrev = 1

    z12 = mom12 / (volr * np.sqrt(float(n12)))
    z6 = mom6 / (volr * np.sqrt(float(n6)))
    zrev = rev / (volr * np.sqrt(float(nrev)))
    z12 = z12.replace([np.inf, -np.inf], np.nan)
    z6 = z6.replace([np.inf, -np.inf], np.nan)
    zrev = zrev.replace([np.inf, -np.inf], np.nan)

    r12 = _rank_pct(z12)
    r6 = _rank_pct(z6)
    rrev = _rank_pct(-zrev)
    rlv = _rank_pct(-volr)

    score = 0.50 * r12 + 0.20 * r6 + 0.20 * rrev + 0.10 * rlv
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 20:
        return zero

    ordered = score.sort_values(ascending=False, kind='mergesort')
    n_each = min(30, len(ordered) // 2)
    if n_each < 1:
        return zero

    weight = min(0.02, 0.5 / float(n_each))
    w = zero.copy()
    longs = ordered.index[:n_each]
    shorts = ordered.index[-n_each:]
    w.loc[longs] = weight
    w.loc[shorts] = -weight
    return w.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)