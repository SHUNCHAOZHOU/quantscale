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
    mkt_ret = ret.mean(axis=1)
    mkt = (1.0 + mkt_ret.fillna(0.0)).cumprod()
    if n >= 100:
        ma = mkt.iloc[-100:].mean()
    elif n >= 60:
        ma = mkt.iloc[-60:].mean()
    else:
        ma = mkt.iloc[-1]
    risk_on = mkt.iloc[-1] >= ma * 0.99
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
    df = pd.DataFrame({'mom126': mom126, 'vol': vol60, 'dv': dv})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df['vol'] > 0]
    if len(df) < 20:
        return pd.Series(0.0, index=universe)
    dv_cut = df['dv'].quantile(0.2)
    df = df[df['dv'] >= dv_cut]
    if len(df) < 20:
        return pd.Series(0.0, index=universe)
    df = df[df['mom126'] > 0]
    if len(df) < 10:
        return pd.Series(0.0, index=universe)
    def zscore(s):
        std = s.std(ddof=0)
        if std == 0 or not np.isfinite(std):
            return s * 0.0
        return (s - s.mean()) / std
    df['score'] = zscore(df['mom126'] / df['vol'])
    k = min(50, len(df))
    if k < 1:
        return pd.Series(0.0, index=universe)
    top = df.sort_values('score', ascending=False).index[:k]
    gross_target = 0.95 if risk_on else 0.25
    w = min(0.02, gross_target / k)
    weights = pd.Series(0.0, index=universe)
    weights.loc[top] = w
    return weights.fillna(0.0)