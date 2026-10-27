import numpy as np
import pandas as pd


def generate_weights(history, universe):
    empty = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return empty

    ac = history.get('adjusted_close')
    if not isinstance(ac, pd.DataFrame) or ac.empty:
        ac = history.get('close')
    if not isinstance(ac, pd.DataFrame) or ac.empty:
        return empty

    cols = [c for c in universe if c in ac.columns]
    if len(cols) < 4:
        return empty

    index = ac.index
    n = len(index)
    if n < 260:
        return empty

    last_pos = n - 1
    reb_pos = last_pos - (last_pos % 21)
    if reb_pos < 252:
        return empty

    prices = ac.loc[:, cols].iloc[:reb_pos + 1].copy().ffill()
    last = prices.iloc[-1]
    valid = last.notna() & (last > 0.5)

    vol_df = history.get('volume')
    if isinstance(vol_df, pd.DataFrame):
        v = vol_df.reindex(index=index, columns=cols).iloc[:reb_pos + 1].ffill()
        if len(v) >= 20:
            dollar_vol = (v.iloc[-20:] * prices.iloc[-20:]).median()
            valid = valid & dollar_vol.notna() & (dollar_vol > 1e6)

    eligible_df = history.get('eligible')
    if isinstance(eligible_df, pd.DataFrame):
        e = eligible_df.reindex(index=index, columns=cols).iloc[:reb_pos + 1].ffill().iloc[-1]
        valid = valid & e.fillna(False).astype(bool)

    prices = prices.loc[:, valid]
    if prices.shape[1] < 4:
        return empty

    p = prices.replace([np.inf, -np.inf], np.nan)

    mom12 = p.iloc[-21] / p.iloc[-252] - 1.0
    mom6 = p.iloc[-21] / p.iloc[-126] - 1.0
    rev21 = -(p.iloc[-1] / p.iloc[-21] - 1.0)

    rets = p.pct_change().iloc[-60:]
    vol60 = rets.std()

    r_mom12 = mom12.rank(method='average', pct=True)
    r_mom6 = mom6.rank(method='average', pct=True)
    r_rev = rev21.rank(method='average', pct=True)
    r_lowvol = (-vol60).rank(method='average', pct=True)

    score = (
        0.45 * r_mom12
        + 0.25 * r_mom6
        + 0.20 * r_rev
        + 0.10 * r_lowvol
    )
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 4:
        return empty

    n_each = min(25, len(score) // 2)
    if n_each < 1:
        return empty

    ranked = pd.DataFrame({'score': score})
    ranked['ticker'] = ranked.index.astype(str)
    ranked = ranked.sort_values(['score', 'ticker'], ascending=[False, True])

    longs = ranked.index[:n_each].tolist()
    shorts = ranked.index[-n_each:].tolist()

    weight = min(0.02, 1.0 / (2.0 * n_each))
    if weight * 2.0 * n_each > 1.0:
        weight = (1.0 - 1e-12) / (2.0 * n_each)

    w = pd.Series(0.0, index=universe, dtype=float)
    for t in longs:
        if t in w.index:
            w.loc[t] = weight
    for t in shorts:
        if t in w.index:
            w.loc[t] = -weight

    return w