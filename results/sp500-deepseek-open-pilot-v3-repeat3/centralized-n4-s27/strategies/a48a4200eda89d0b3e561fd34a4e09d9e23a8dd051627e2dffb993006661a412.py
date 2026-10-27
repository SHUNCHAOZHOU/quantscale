import numpy as np
import pandas as pd


def _get_df(history, key):
    if isinstance(history, dict):
        df = history.get(key)
        if isinstance(df, pd.DataFrame):
            return df
    return None


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
    n_needed = 130
    adj_s = adj.iloc[-n_needed:].ffill()
    if len(adj_s) < 62:
        return zero

    # Smoothed 60-session return: average of 60-day returns over last 5 sessions
    rets = []
    max_lag = 5
    lookback = 60
    for lag in range(max_lag):
        end_idx = len(adj_s) - 1 - lag
        start_idx = end_idx - lookback
        if start_idx < 0:
            continue
        r = adj_s.iloc[end_idx] / adj_s.iloc[start_idx] - 1.0
        rets.append(r)
    if len(rets) == 0:
        return zero
    ret_smooth = pd.concat(rets, axis=1).mean(axis=1, skipna=True)
    ret_smooth = ret_smooth.replace([np.inf, -np.inf], np.nan)

    last = adj_s.iloc[-1]
    elig_df = _get_df(history, 'eligible')
    if elig_df is not None and not elig_df.empty:
        elig = elig_df.reindex(columns=universe).iloc[-1].astype(float).fillna(0.0).astype(bool)
    else:
        elig = pd.Series(True, index=universe)

    vol = _get_df(history, 'volume')
    liq = pd.Series(True, index=universe)
    if vol is not None and not vol.empty:
        vol_s = vol.reindex(columns=universe).astype(float).iloc[-60:]
        price_s = adj.reindex(columns=universe).astype(float).iloc[-60:]
        dv = vol_s * price_s
        med = dv.median(axis=0, skipna=True)
        liq = (med > 1e6) | med.isna()

    price_ok = last.notna() & (last > 0)
    mask = elig & price_ok & ret_smooth.notna() & np.isfinite(ret_smooth) & liq
    tickers = universe[mask].tolist()
    if len(tickers) < 2:
        return zero

    s = ret_smooth.reindex(tickers).astype(float).dropna()
    if len(s) < 2:
        return zero

    mu = s.mean()
    sd = s.std(ddof=0)
    if np.isfinite(sd) and sd > 0:
        z = (s - mu) / sd
    else:
        z = s - mu

    # Reversal: long lowest z (losers), short highest z (winners)
    tmp = pd.DataFrame({'t': z.index, 'z': z.values})
    tmp = tmp.sort_values(['z', 't'], ascending=[True, True]).reset_index(drop=True)
    n = len(tmp)
    k = max(25, int(np.floor(0.15 * n)))
    k = min(k, n // 2)
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
