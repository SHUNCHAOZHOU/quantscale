import numpy as np
import pandas as pd


def _rank(s):
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return s
    return s.rank(method='average', pct=True)


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    zero = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return zero

    adj = history.get('adjusted_close')
    if adj is None:
        adj = history.get('close')
    if not isinstance(adj, pd.DataFrame) or adj.shape[0] == 0:
        return zero
    if not adj.index.is_monotonic_increasing:
        adj = adj.sort_index()

    cols = [c for c in tickers if c in adj.columns]
    if len(cols) < 10:
        return zero

    px = adj[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    m = int(px.shape[0])

    # Quarterly rebalance grid: weights are stale by design within a quarter.
    reb = ((m - 1) // 63) * 63
    if reb < 273:
        return zero

    data = px.iloc[:reb + 1]
    last = data.iloc[-1]
    p_1m = data.iloc[-1 - 21]
    p_12m = data.iloc[-1 - 252]
    mom = p_1m / p_12m - 1.0
    mom = mom.replace([np.inf, -np.inf], np.nan)

    rets = data.pct_change()
    vol = rets.iloc[-126:].std(ddof=1).replace([np.inf, -np.inf], np.nan)

    valid = last.notna() & (last > 0) & mom.notna() & vol.notna() & (vol > 0)

    # Point-in-time eligibility at the rebalance row.
    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame) and elig.shape[0] > 0:
        e2 = elig.reindex(columns=cols)
        if not e2.index.is_monotonic_increasing:
            e2 = e2.sort_index()
        if len(e2) > reb:
            row = e2.iloc[reb]
        else:
            row = e2.iloc[-1]
        row = row.reindex(cols).fillna(False).astype(bool)
        if row.sum() >= 10:
            valid = valid & row

    # Liquidity screen: recent median dollar volume must be positive / not bottom decile.
    vdf = history.get('volume')
    if isinstance(vdf, pd.DataFrame) and vdf.shape[0] > 0:
        v2 = vdf.reindex(columns=cols)
        if not v2.index.is_monotonic_increasing:
            v2 = v2.sort_index()
        vsub = v2.reindex(index=data.index[-60:], columns=cols)
        p60 = data.iloc[-60:]
        dollar = vsub * p60
        liq = dollar.median(axis=0).reindex(cols)
        valid = valid & liq.notna() & (liq > 0)
        if liq.notna().sum() >= 20:
            thr = liq.quantile(0.10)
            valid = valid & (liq >= thr)

    valid = valid.fillna(False)
    names = valid[valid].index
    if len(names) < 10:
        return zero

    mom_v = mom.reindex(names)
    vol_v = vol.reindex(names)

    # Positive 12-1 momentum with a small low-volatility risk tilt.
    score = _rank(mom_v) - 0.20 * _rank(vol_v)
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 8:
        return zero

    n_side = min(25, len(score) // 2)
    if n_side < 4:
        return zero

    order = score.sort_values(ascending=False)
    longs = order.index[:n_side]
    shorts = order.index[-n_side:]

    wmag = min(0.02, 0.5 / float(n_side))

    weights = pd.Series(0.0, index=tickers, dtype=float)
    weights.loc[longs] = wmag
    weights.loc[shorts] = -wmag

    # Dollar-neutral by construction (equal counts each side), but enforce anyway.
    net = float(weights.sum())
    if abs(net) > 1e-12:
        active = weights[weights != 0].index
        if len(active) > 0:
            weights.loc[active] -= net / float(len(active))

    weights = weights.clip(-0.02, 0.02)
    gross = float(weights.abs().sum())
    if gross > 1.0 + 1e-12:
        weights = weights * (1.0 / gross)

    net = float(weights.sum())
    if abs(net) > 0.005:
        active = weights[weights != 0].index
        if len(active) > 0:
            weights.loc[active] -= net / float(len(active))
        weights = weights.clip(-0.02, 0.02)
        gross = float(weights.abs().sum())
        if gross > 1.0 + 1e-12:
            weights = weights * (1.0 / gross)

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights.reindex(tickers).fillna(0.0).astype(float)