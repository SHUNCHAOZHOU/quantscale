import numpy as np
import pandas as pd


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    adj = history.get('adjusted_close')
    if not isinstance(adj, pd.DataFrame) or adj.empty:
        adj = history.get('close')
    if not isinstance(adj, pd.DataFrame) or adj.empty:
        return zero

    cols = [c for c in universe if c in adj.columns]
    if len(cols) < 20:
        return zero

    adj = adj[cols].sort_index()
    n = len(adj)
    if n < 120:
        return zero

    rebalance = 21
    anchor = ((n - 1) // rebalance) * rebalance
    if anchor < 100:
        return zero

    px = adj.iloc[:anchor + 1].ffill()
    if len(px) < 100:
        return zero

    last = px.iloc[-1]
    valid = last.notna() & (last > 1.0)
    valid = valid & np.isfinite(last.astype(float))

    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame) and not elig.empty:
        e = elig.reindex(index=adj.index, columns=cols)
        if anchor < len(e):
            ev = e.iloc[anchor].fillna(False).astype(bool)
        else:
            ev = e.iloc[-1].fillna(False).astype(bool)
        valid = valid & ev

    vol = history.get('volume')
    close = history.get('close')
    if isinstance(vol, pd.DataFrame) and not vol.empty:
        v = vol.reindex(index=adj.index, columns=cols).iloc[:anchor + 1]
        if isinstance(close, pd.DataFrame) and not close.empty:
            c = close.reindex(index=adj.index, columns=cols).iloc[:anchor + 1]
        else:
            c = px
        dv = v * c
        med = dv.tail(60).median()
        valid = valid & med.notna() & (med >= 1.0e6)

    ret = px.pct_change()
    ret = ret.replace([np.inf, -np.inf], np.nan)
    mkt = ret.mean(axis=1, skipna=True)
    if mkt.notna().sum() < 60:
        return zero

    window = min(252, len(ret) - 1)
    if window < 60:
        return zero
    r = ret.tail(window)
    m = mkt.tail(window)
    mm = m.dropna()
    if len(mm) < 60:
        return zero
    r = r.loc[mm.index]
    m = mm

    R = r.to_numpy(dtype=float, copy=False)
    M = m.to_numpy(dtype=float, copy=False)
    betas = np.full(R.shape[1], np.nan, dtype=float)
    for j in range(R.shape[1]):
        x = R[:, j]
        good = np.isfinite(x) & np.isfinite(M)
        if good.sum() < 60:
            continue
        xg = x[good]
        mg = M[good]
        varm = np.var(mg, ddof=1)
        if not np.isfinite(varm) or varm <= 0.0:
            continue
        cov = np.cov(xg, mg, ddof=1)[0, 1]
        if np.isfinite(cov):
            betas[j] = cov / varm

    beta = pd.Series(betas, index=cols)
    valid = valid & beta.notna() & np.isfinite(beta)

    mkt_index = (1.0 + mkt.fillna(0.0)).cumprod()
    ma_len = min(200, len(mkt_index))
    if ma_len < 20:
        return zero
    ma = mkt_index.tail(ma_len).mean()
    if not np.isfinite(ma) or ma <= 0.0:
        return zero
    trend = mkt_index.iloc[-1] / ma - 1.0
    risk_on = bool(trend >= 0.0)

    cand = beta[valid].dropna().sort_values()
    if len(cand) < 20:
        return zero
    n_side = min(50, len(cand) // 2)
    if n_side < 5:
        return zero

    if risk_on:
        shorts = cand.index[:n_side]
        longs = cand.index[-n_side:]
    else:
        longs = cand.index[:n_side]
        shorts = cand.index[-n_side:]

    w = min(0.02, 0.5 / float(n_side))
    weights = pd.Series(0.0, index=universe, dtype=float)
    for t in longs:
        weights.loc[t] = weights.loc[t] + w
    for t in shorts:
        weights.loc[t] = weights.loc[t] - w

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
    net = float(weights.sum())
    if abs(net) > 1e-12:
        weights = weights - net / float(len(universe))
        weights = weights.clip(-0.02, 0.02)
    gross = float(weights.abs().sum())
    if gross > 1.0 and gross > 0.0:
        weights = weights * (1.0 / gross)
    return weights.astype(float)