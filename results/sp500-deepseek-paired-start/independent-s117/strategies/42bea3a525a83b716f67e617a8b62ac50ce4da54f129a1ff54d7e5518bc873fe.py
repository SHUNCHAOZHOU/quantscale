import numpy as np
import pandas as pd


def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)

    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None:
        return pd.Series(0.0, index=universe)

    cols = [t for t in universe if t in ac.columns]
    if len(cols) < 4:
        return pd.Series(0.0, index=universe)

    # Limit history for efficiency while keeping enough for 12-1 momentum.
    ac = ac.loc[:, cols].tail(400).astype(float)
    if len(ac) < 25:
        return pd.Series(0.0, index=universe)

    idx = ac.index
    last_date = idx[-1]
    cur_period = last_date.to_period('M')
    periods = idx.to_period('M')
    prev_mask = periods < cur_period

    # Rebalance at the start of each month using the previous month-end close.
    if prev_mask.any():
        formation_date = idx[prev_mask][-1]
    else:
        formation_date = idx[-1]

    p = ac.loc[:formation_date].copy()
    if len(p) < 25:
        return pd.Series(0.0, index=universe)

    def momentum(days):
        # Skip the most recent month to avoid short-term reversal contamination.
        if len(p) >= days + 21:
            base = p.iloc[-days - 21]
        else:
            base = p.iloc[0]
        skip = p.iloc[-21]
        return skip / base - 1.0

    mom3 = momentum(63)
    mom6 = momentum(126)
    mom12 = momentum(252)
    mom_df = pd.concat([mom3, mom6, mom12], axis=1)
    mom_ranks = mom_df.rank(axis=0, pct=True, method='average')
    score = mom_ranks.mean(axis=1, skipna=True)

    rets = p.pct_change(fill_method=None)
    vol = rets.tail(60).std(ddof=0)

    vol_df = history.get('volume')
    if vol_df is not None:
        vcols = [t for t in cols if t in vol_df.columns]
        v = vol_df.loc[:, vcols].tail(20).astype(float)
        close_df = history.get('close')
        if close_df is not None:
            c = close_df.loc[:, vcols].tail(20).astype(float)
        else:
            c = ac.loc[:, vcols].tail(20)
        adv = (v * c).mean(skipna=True).reindex(cols)
    else:
        adv = pd.Series(np.nan, index=cols)

    eligible_df = history.get('eligible')
    if eligible_df is not None and len(eligible_df):
        erow = eligible_df.iloc[-1].reindex(cols)
        elig = erow.fillna(False).astype(bool)
    else:
        elig = pd.Series(True, index=cols)

    last_price = ac.iloc[-1]
    df = pd.DataFrame({
        'score': score,
        'vol': vol,
        'adv': adv,
        'eligible': elig,
        'last': last_price,
    })
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[
        df['eligible']
        & df['last'].notna()
        & df['score'].notna()
        & df['vol'].notna()
        & (df['vol'] > 0)
    ]

    if len(df) < 4:
        return pd.Series(0.0, index=universe)

    # Liquidity filter: keep the more liquid portion when liquidity data is available.
    if df['adv'].notna().sum() >= 5:
        df = df[df['adv'].notna() & (df['adv'] > 0)]
        if len(df) >= 5:
            q = df['adv'].quantile(0.20)
            df = df[df['adv'] >= q]

    if len(df) < 4:
        return pd.Series(0.0, index=universe)

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