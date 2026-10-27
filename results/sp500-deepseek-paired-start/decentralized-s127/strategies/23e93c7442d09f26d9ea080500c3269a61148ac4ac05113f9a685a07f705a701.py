import numpy as np
import pandas as pd


def _df(history, key):
    try:
        x = history.get(key)
    except Exception:
        return None
    if isinstance(x, pd.DataFrame) and x.shape[0] > 0:
        return x
    return None


def _rank_pct(s):
    return s.rank(method='average', pct=True, na_option='keep')


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    adj = _df(history, 'adjusted_close')
    if adj is None:
        adj = _df(history, 'close')
    if adj is None:
        return zero

    cols = [c for c in universe if c in adj.columns]
    if len(cols) < 10:
        return zero

    adj = adj[cols].sort_index()
    n_rows = len(adj)
    if n_rows < 260:
        return zero

    rebalance = 21
    anchor = ((n_rows - 1) // rebalance) * rebalance
    if anchor < 259:
        return zero
    n_use = anchor + 1

    px = adj.iloc[:n_use]
    last = px.iloc[-1]
    valid = last.notna() & (last > 0.0)

    obs = px.notna().sum()
    valid = valid & (obs >= 120)

    elig = _df(history, 'eligible')
    if elig is not None:
        ecols = [c for c in cols if c in elig.columns]
        if len(ecols) > 0:
            e = elig[ecols].iloc[:n_use]
            if len(e) > 0:
                ev = e.iloc[-1].reindex(cols).fillna(True).astype(bool)
                valid = valid & ev

    vol_df = _df(history, 'volume')
    close_df = _df(history, 'close')
    if vol_df is not None and close_df is not None:
        vcols = [c for c in cols if c in vol_df.columns and c in close_df.columns]
        if len(vcols) > 0:
            vv = vol_df[vcols].iloc[max(0, n_use - 60):n_use]
            cc = close_df[vcols].iloc[max(0, n_use - 60):n_use]
            adv = (vv * cc).median()
            adv = adv.reindex(cols).fillna(0.0)
            valid = valid & (adv >= 1000000.0)

    ret = px / px.shift(1) - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)

    vol126 = ret.tail(126).std()
    tail = ret.tail(126)
    down = tail.where(tail < 0.0).std()
    max21 = ret.tail(21).max()
    trend = px.iloc[-1] / px.tail(200).mean() - 1.0

    mkt = ret.mean(axis=1)
    excess = ret.sub(mkt, axis=0)
    n_ex = len(excess)
    if n_ex > 63:
        end = max(1, n_ex - 21)
        start = max(0, n_ex - 252)
        if end > start:
            resmom = excess.iloc[start:end].sum()
        else:
            resmom = excess.sum()
    else:
        resmom = excess.sum()

    f = pd.DataFrame({
        'vol': vol126,
        'down': down,
        'max': max21,
        'res': resmom,
        'trend': trend
    }).reindex(cols)
    f = f.replace([np.inf, -np.inf], np.nan)

    valid = valid & f['vol'].notna() & (f['vol'] > 0.0) & f['res'].notna()
    if int(valid.sum()) < 20:
        return zero

    fv = f.loc[valid]
    r_vol = 1.0 - _rank_pct(fv['vol'])
    r_down = 1.0 - _rank_pct(fv['down'])
    r_max = 1.0 - _rank_pct(fv['max'])
    r_res = _rank_pct(fv['res'])
    r_trend = _rank_pct(fv['trend'])

    score = (0.30 * r_vol.fillna(0.5)
             + 0.20 * r_down.fillna(0.5)
             + 0.15 * r_max.fillna(0.5)
             + 0.25 * r_res.fillna(0.5)
             + 0.10 * r_trend.fillna(0.5))
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 20:
        return zero

    n_side = min(50, len(score) // 2)
    if n_side < 1:
        return zero

    order = score.sort_values(ascending=False)
    longs = order.index[:n_side]
    shorts = order.index[-n_side:]
    w = min(0.02, 0.5 / float(n_side))

    weights = pd.Series(0.0, index=universe, dtype=float)
    for t in longs:
        weights.loc[t] = weights.loc[t] + w
    for t in shorts:
        weights.loc[t] = weights.loc[t] - w

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
    net = float(weights.sum())
    if abs(net) > 1e-12:
        adj = net / float(len(universe))
        weights = (weights - adj).clip(-0.02, 0.02)
    gross = float(weights.abs().sum())
    if gross > 1.0 and gross > 0.0:
        weights = weights * (1.0 / gross)
    return weights