import numpy as np
import pandas as pd

def generate_weights(history, universe):
    universe = pd.Index(universe)
    n_univ = len(universe)
    if n_univ == 0:
        return pd.Series(dtype=float)

    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None:
        return pd.Series(0.0, index=universe)

    cols = [t for t in universe if t in ac.columns]
    if len(cols) < 4:
        return pd.Series(0.0, index=universe)

    prices = ac.loc[:, cols].tail(500).astype(float).copy()
    if len(prices) < 30:
        return pd.Series(0.0, index=universe)

    idx = prices.index
    last_date = idx[-1]
    periods = idx.to_period('M')
    cur_period = last_date.to_period('M')
    prev_mask = periods < cur_period
    if prev_mask.any():
        formation_date = idx[prev_mask][-1]
    else:
        formation_date = idx[-1]

    p = prices.loc[:formation_date].copy()
    if len(p) < 30:
        return pd.Series(0.0, index=universe)

    rets = p.pct_change(fill_method=None)
    mkt = rets.mean(axis=1, skipna=True)

    L = len(p)
    if L >= 273:
        mom12 = p.iloc[-21] / p.iloc[-273] - 1.0
    elif L >= 252:
        mom12 = p.iloc[-1] / p.iloc[-252] - 1.0
    else:
        mom12 = p.iloc[-1] / p.iloc[0] - 1.0

    if L >= 273:
        seg = mkt.iloc[-273:-21]
    elif L >= 252:
        seg = mkt.iloc[-252:]
    else:
        seg = mkt.iloc[1:]
    market_mom12 = float((1.0 + seg.fillna(0.0)).prod() - 1.0)

    beta_window = min(252, max(20, L - 1))
    R = rets.tail(beta_window).values
    m = mkt.tail(beta_window).values
    m = np.nan_to_num(m, nan=0.0)
    m_dm = m - m.mean()
    m_var = float(np.mean(m_dm ** 2))
    R_mean = np.nanmean(R, axis=0)
    R_dm = R - R_mean
    prod = np.where(np.isfinite(R_dm), R_dm * m_dm[:, None], 0.0)
    counts = np.isfinite(R_dm).sum(axis=0)
    denom = counts * m_var
    beta_vals = np.divide(prod.sum(axis=0), denom, out=np.zeros(len(cols)), where=denom > 0)
    beta = pd.Series(beta_vals, index=cols)

    resmom = mom12 - beta * market_mom12

    high252 = p.tail(252).max(skipna=True)
    prox = p.iloc[-1] / high252
    vol60 = rets.tail(60).std(ddof=0)

    vol_df = history.get('volume')
    if vol_df is not None:
        vcols = [t for t in cols if t in vol_df.columns]
        v = vol_df.loc[:, vcols].tail(20).astype(float)
        close_df = history.get('close')
        if close_df is not None:
            c = close_df.loc[:, vcols].tail(20).astype(float)
        else:
            c = prices.loc[:, vcols].tail(20)
        adv = (v * c).mean(skipna=True).reindex(cols)
    else:
        adv = pd.Series(np.nan, index=cols)

    eligible_df = history.get('eligible')
    if eligible_df is not None and len(eligible_df):
        erow = eligible_df.iloc[-1].reindex(cols)
        elig = erow.fillna(False).astype(bool)
    else:
        elig = pd.Series(True, index=cols)

    last_price = prices.iloc[-1]
    df = pd.DataFrame({
        'prox': prox,
        'resmom': resmom,
        'vol': vol60,
        'adv': adv,
        'eligible': elig,
        'last': last_price,
    })
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df['eligible'] & df['last'].notna() & df['prox'].notna() & df['resmom'].notna() & df['vol'].notna() & (df['vol'] > 0)]

    if len(df) < 4:
        return pd.Series(0.0, index=universe)

    if df['adv'].notna().sum() >= 5:
        df = df[df['adv'].notna() & (df['adv'] > 0)]
        if len(df) >= 5:
            q = df['adv'].quantile(0.20)
            df = df[df['adv'] >= q]

    if len(df) < 4:
        return pd.Series(0.0, index=universe)

    rank_prox = df['prox'].rank(method='average', pct=True)
    rank_resmom = df['resmom'].rank(method='average', pct=True)
    rank_lowvol = (-df['vol']).rank(method='average', pct=True)
    df['score'] = 0.45 * rank_prox + 0.40 * rank_resmom + 0.15 * rank_lowvol

    n_side = min(50, len(df) // 2)
    if n_side < 1:
        return pd.Series(0.0, index=universe)

    ordered = df.sort_values('score', ascending=False)
    longs = ordered.index[:n_side]
    shorts = ordered.index[-n_side:]

    side_gross = min(0.5, 0.02 * n_side)
    mkt_level = (1.0 + mkt.fillna(0.0)).cumprod()
    if len(mkt_level) >= 200:
        ma = mkt_level.tail(200).mean()
        if mkt_level.iloc[-1] < ma:
            side_gross *= 0.5
    w = side_gross / n_side

    weights = pd.Series(0.0, index=universe)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
    return weights.astype(float)