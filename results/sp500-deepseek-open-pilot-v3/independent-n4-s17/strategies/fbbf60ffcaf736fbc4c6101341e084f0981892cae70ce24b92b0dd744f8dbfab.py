import pandas as pd
import numpy as np

def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)
    adj_all = history.get('adjusted_close')
    vol_all = history.get('volume')
    if adj_all is None or vol_all is None:
        return pd.Series(0.0, index=universe)
    common = universe.intersection(adj_all.columns).intersection(vol_all.columns)
    if len(common) == 0:
        return pd.Series(0.0, index=universe)
    adj = adj_all[common]
    vol = vol_all[common]
    idx = adj.index
    if len(idx) == 0:
        return pd.Series(0.0, index=universe)
    if not idx.is_monotonic_increasing:
        adj = adj.sort_index()
        vol = vol.sort_index()
        idx = adj.index
    periods = idx.to_period('M')
    first_mask = ~periods.duplicated(keep='first')
    rebal_dates = idx[first_mask]
    if len(rebal_dates) == 0:
        return pd.Series(0.0, index=universe)
    rebal_date = rebal_dates[-1]
    adj_r = adj.loc[:rebal_date]
    vol_r = vol.loc[:rebal_date]
    n = len(adj_r)
    if n < 30:
        return pd.Series(0.0, index=universe)
    ret = adj_r.pct_change()
    if n >= 22:
        ret21 = adj_r.iloc[-1] / adj_r.iloc[-22] - 1.0
    else:
        ret21 = pd.Series(np.nan, index=common)
    if n >= 6:
        ret5 = adj_r.iloc[-1] / adj_r.iloc[-6] - 1.0
    else:
        ret5 = pd.Series(np.nan, index=common)
    if n >= 127:
        mom126 = adj_r.iloc[-22] / adj_r.iloc[-127] - 1.0
    else:
        mom126 = pd.Series(np.nan, index=common)
    if n >= 60:
        vol60 = ret.iloc[-60:].std()
    else:
        vol60 = ret.std()
    if n >= 20:
        dv = (adj_r.iloc[-20:] * vol_r.iloc[-20:]).median()
    else:
        dv = (adj_r * vol_r).median()
    df = pd.DataFrame({'ret21': ret21, 'ret5': ret5, 'mom126': mom126, 'vol': vol60, 'dv': dv})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df['vol'] > 0]
    if len(df) < 20:
        return pd.Series(0.0, index=universe)
    dv_cut = df['dv'].quantile(0.2)
    df = df[df['dv'] >= dv_cut]
    if len(df) < 20:
        return pd.Series(0.0, index=universe)
    df = df[df['ret21'].abs() < 0.75]
    if len(df) < 20:
        return pd.Series(0.0, index=universe)
    def zscore(s):
        std = s.std(ddof=0)
        if std == 0 or not np.isfinite(std):
            return s * 0.0
        return (s - s.mean()) / std
    score = (-1.0 * zscore(df['ret21']) - 0.25 * zscore(df['vol']) + 0.15 * zscore(df['mom126']) - 0.2 * zscore(df['ret5']))
    df = df.assign(score=score)
    k = min(30, len(df) // 2)
    if k < 1:
        return pd.Series(0.0, index=universe)
    df_sorted = df.sort_values('score', ascending=False)
    longs = df_sorted.index[:k]
    shorts = df_sorted.index[-k:]
    w = min(0.02, 0.96 / (2.0 * k))
    weights = pd.Series(0.0, index=universe)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    return weights.fillna(0.0)