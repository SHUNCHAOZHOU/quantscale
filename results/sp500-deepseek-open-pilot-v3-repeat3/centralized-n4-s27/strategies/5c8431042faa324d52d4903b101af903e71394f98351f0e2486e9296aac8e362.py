import numpy as np
import pandas as pd
import math


def _get_df(history, key):
    if isinstance(history, dict):
        df = history.get(key)
        if isinstance(df, pd.DataFrame):
            return df
    return None


def _latest_bool(df, universe):
    if df is None or df.empty:
        return pd.Series(True, index=universe)
    row = df.reindex(columns=universe).iloc[-1]
    return row.astype(float).fillna(0.0).astype(bool)


def generate_weights(history, universe):
    universe = pd.Index(pd.unique(pd.Index(universe).astype(str)))
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    adj = _get_df(history, 'adjusted_open')
    if adj is None or adj.empty:
        adj = _get_df(history, 'open')
    if adj is None or adj.empty:
        return zero

    adj = adj.reindex(columns=universe).astype(float)
    adj_ff = adj.ffill()
    if len(adj_ff) == 0:
        return zero

    last = adj_ff.iloc[-1]
    lookback = 60
    if len(adj_ff) <= lookback:
        prev = adj_ff.iloc[0]
    else:
        prev = adj_ff.iloc[-lookback - 1]

    ret = last / prev - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)

    elig = _latest_bool(_get_df(history, 'eligible'), universe)

    vol = _get_df(history, 'volume')
    liq = pd.Series(True, index=universe)
    if vol is not None and not vol.empty:
        vol = vol.reindex(columns=universe).astype(float)
        m = min(60, len(vol), len(adj))
        if m > 0:
            v = vol.iloc[-m:].copy()
            p = adj.iloc[-m:].copy()
            dv = v * p
            med = dv.median(axis=0, skipna=True)
            liq = (med > 1000000.0) | med.isna()

    price_ok = (last > 0) & last.notna()
    mask = elig & price_ok & ret.notna() & np.isfinite(ret) & liq
    tickers = universe[mask].tolist()
    if len(tickers) < 2:
        return zero

    s = ret.reindex(tickers).astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return zero

    mu = s.mean()
    sd = s.std(ddof=0)
    if np.isfinite(sd) and sd > 0:
        z = (s - mu) / sd
    else:
        z = s - mu

    tmp = pd.DataFrame({'t': z.index, 's': z.values})
    tmp = tmp.sort_values(['s', 't'], ascending=[False, True]).reset_index(drop=True)

    n_good = len(tmp)
    min_side = 25
    k = max(min_side, int(math.floor(0.12 * n_good)))
    k = min(k, n_good // 2)
    if k < 1:
        return zero

    longs = tmp.iloc[:k]['t'].tolist()
    shorts = tmp.iloc[-k:]['t'].tolist()

    base = min(0.5 / k, 0.02)
    w = pd.Series(0.0, index=universe, dtype=float)
    for t in longs:
        w[t] = base
    for t in shorts:
        w[t] = -base

    w = w.clip(-0.02, 0.02)
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return w