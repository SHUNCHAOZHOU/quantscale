import numpy as np
import pandas as pd

def generate_weights(history, universe):
    universe = pd.Index(universe)
    n = len(universe)
    if n == 0:
        return pd.Series(dtype=float)

    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None:
        return pd.Series(0.0, index=universe)

    cols = [t for t in universe if t in ac.columns]
    if len(cols) < 2:
        return pd.Series(0.0, index=universe)

    prices = ac.loc[:, cols].tail(300).astype(float).copy()
    if len(prices) < 25:
        return pd.Series(0.0, index=universe)

    rets = prices.pct_change(fill_method=None)
    last = prices.iloc[-1]
    vol = rets.tail(60).std(ddof=0)

    if len(prices) >= 22:
        r21 = prices.iloc[-1] / prices.iloc[-22] - 1.0
    else:
        r21 = pd.Series(np.nan, index=cols)

    if len(prices) >= 252:
        mom = prices.iloc[-22] / prices.iloc[-252] - 1.0
    elif len(prices) >= 126:
        mom = prices.iloc[-22] / prices.iloc[-126] - 1.0
    else:
        mom = pd.Series(np.nan, index=cols)

    vol_df = history.get('volume')
    if vol_df is not None:
        vcols = [t for t in cols if t in vol_df.columns]
        v = vol_df.loc[:, vcols].tail(20).astype(float)
        close_df = history.get('close')
        if close_df is not None:
            c = close_df.loc[:, vcols].tail(20).astype(float)
        else:
            c = prices.loc[:, vcols].tail(20)
        adv = (v * c).mean(skipna=True)
        adv = adv.reindex(cols)
    else:
        adv = pd.Series(np.nan, index=cols)

    eligible_df = history.get('eligible')
    if eligible_df is not None and len(eligible_df):
        erow = eligible_df.iloc[-1].reindex(cols)
        elig = erow.fillna(False).astype(bool)
    else:
        elig = pd.Series(True, index=cols)

    df = pd.DataFrame({'r21': r21, 'mom': mom, 'vol': vol, 'adv': adv, 'eligible': elig, 'last': last})
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df['eligible'] & df['last'].notna() & df['r21'].notna() & df['vol'].notna() & (df['vol'] > 0)]

    if df['adv'].notna().sum() >= 5:
        df = df[df['adv'].notna() & (df['adv'] > 0)]
        if len(df) >= 5:
            q = df['adv'].quantile(0.20)
            df = df[df['adv'] >= q]

    if len(df) < 4:
        return pd.Series(0.0, index=universe)

    rank_rev = (-df['r21']).rank(method='average', pct=True)
    rank_mom = df['mom'].rank(method='average', pct=True).fillna(0.5)
    rank_lowvol = (-df['vol']).rank(method='average', pct=True)

    df = df.copy()
    df['score'] = 0.55 * rank_rev + 0.30 * rank_mom + 0.15 * rank_lowvol

    n_side = min(50, len(df) // 2)
    if n_side < 1:
        return pd.Series(0.0, index=universe)

    ordered = df.sort_values('score', ascending=False)
    longs = ordered.index[:n_side]
    shorts = ordered.index[-n_side:]

    side_gross = min(0.5, 0.02 * n_side)
    w = side_gross / n_side

    weights = pd.Series(0.0, index=universe)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
    return weights.astype(float)