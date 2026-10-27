import pandas as pd
import numpy as np

def generate_weights(history, universe):
    # Extract adjusted close and volume
    adj = history['adjusted_close']
    vol = history['volume']
    # Align to universe columns
    tickers = universe.intersection(adj.columns)
    if len(tickers) == 0:
        return pd.Series(0.0, index=universe)
    adj = adj[tickers]
    vol = vol[tickers]
    idx = adj.index
    if len(idx) == 0:
        return pd.Series(0.0, index=universe)
    last_date = idx[-1]
    # Find last rebalance date: first trading day of month
    periods = idx.to_period('M')
    first_mask = ~periods.duplicated(keep='first')
    rebal_dates = idx[first_mask]
    if len(rebal_dates) == 0:
        return pd.Series(0.0, index=universe)
    rebal_date = rebal_dates[-1]
    # Slice data up to rebal_date
    adj_r = adj.loc[:rebal_date]
    vol_r = vol.loc[:rebal_date]
    n = len(adj_r)
    if n < 2:
        return pd.Series(0.0, index=universe)
    ret = adj_r.pct_change()
    # Momentum: 120-day return skipping last 5 days
    if n >= 126:
        mom = adj_r.iloc[-6] / adj_r.iloc[-126] - 1.0
    else:
        mom = pd.Series(np.nan, index=tickers)
    # Short-term reversal: 5-day return
    if n >= 6:
        ret5 = adj_r.iloc[-1] / adj_r.iloc[-6] - 1.0
    else:
        ret5 = pd.Series(np.nan, index=tickers)
    # Volatility: std of daily returns over last 60 days
    if n >= 60:
        vol60 = ret.iloc[-60:].std()
    else:
        vol60 = ret.std()
    # Dollar volume: median over last 20 days
    if n >= 20:
        dv = (adj_r.iloc[-20:] * vol_r.iloc[-20:]).median()
    else:
        dv = (adj_r * vol_r).median()
    # Build dataframe
    df = pd.DataFrame({'mom': mom, 'ret5': ret5, 'vol': vol60, 'dv': dv})
    df = df.dropna(subset=['mom', 'ret5', 'vol', 'dv'])
    df = df[df['vol'] > 0]
    if len(df) < 10:
        return pd.Series(0.0, index=universe)
    # Liquidity filter: keep top 80% by dollar volume
    dv_cut = df['dv'].quantile(0.2)
    df = df[df['dv'] >= dv_cut]
    if len(df) < 10:
        return pd.Series(0.0, index=universe)
    # Risk-adjusted momentum
    df['mom_adj'] = df['mom'] / df['vol']
    # Composite score
    def zscore(s):
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            return s * 0.0
        return (s - s.mean()) / std
    df['z_mom'] = zscore(df['mom_adj'])
    df['z_ret5'] = zscore(df['ret5'])
    df['score'] = df['z_mom'] - 0.5 * df['z_ret5']
    # Select top and bottom
    k = min(50, len(df) // 2)
    if k < 1:
        return pd.Series(0.0, index=universe)
    df_sorted = df.sort_values('score', ascending=False)
    long_tickers = df_sorted.index[:k]
    short_tickers = df_sorted.index[-k:]
    # Weight per position
    w = min(0.02, 1.0 / (2.0 * k))
    weights = pd.Series(0.0, index=universe)
    weights.loc[long_tickers] = w
    weights.loc[short_tickers] = -w
    return weights
