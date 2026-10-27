def generate_weights(history, universe):
    import numpy as np
    import pandas as pd

    tickers = pd.Index(universe)
    zero = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return zero

    close_all = history.get('adjusted_close')
    if close_all is None:
        return zero
    close_all = close_all.reindex(columns=tickers)
    if close_all.shape[0] == 0:
        return zero

    dates = close_all.index
    current = dates[-1]
    wd = current.weekday()
    target = current - pd.Timedelta(days=(wd - 4) % 7)
    pos = dates.searchsorted(target, side='right') - 1
    if pos < 0:
        pos = 0
    ref = dates[pos]
    start = max(0, pos - 299)
    c = close_all.iloc[start:pos + 1]
    if c.shape[0] < 3:
        return zero

    if 'eligible' in history and history['eligible'] is not None:
        elig_all = history['eligible'].reindex(columns=tickers)
        if len(elig_all.index) > 0:
            try:
                e = elig_all.loc[ref]
            except KeyError:
                past = elig_all.loc[:ref]
                e = past.iloc[-1] if len(past) > 0 else pd.Series(True, index=tickers)
        else:
            e = pd.Series(True, index=tickers)
    else:
        e = pd.Series(True, index=tickers)
    e = e.reindex(tickers).fillna(False).astype(bool)

    last = pd.to_numeric(c.iloc[-1], errors='coerce')
    valid = last.notna() & (last > 0) & e
    if valid.sum() < 6:
        return zero

    n = c.shape[0]
    if n >= 252:
        mom = c.iloc[-22] / c.iloc[-252] - 1.0
    elif n >= 22:
        mom = c.iloc[-22] / c.iloc[0] - 1.0
    else:
        mom = pd.Series(np.nan, index=tickers)

    if n >= 22:
        rev = last / c.iloc[-22] - 1.0
    else:
        rev = last / c.iloc[0] - 1.0

    rets = c.pct_change()
    if n >= 61:
        vol = rets.iloc[-60:].std()
    else:
        vol = rets.std()

    mom = pd.to_numeric(mom, errors='coerce').reindex(tickers)
    rev = pd.to_numeric(rev, errors='coerce').reindex(tickers)
    vol = pd.to_numeric(vol, errors='coerce').reindex(tickers)
    valid = valid & mom.notna() & rev.notna() & vol.notna()
    valid = valid & np.isfinite(mom) & np.isfinite(rev) & np.isfinite(vol)
    if valid.sum() < 6:
        return zero

    idx = valid.index[valid.to_numpy(dtype=bool)]
    scores = (0.55 * mom.loc[idx].rank(pct=True)
              - 0.35 * rev.loc[idx].rank(pct=True)
              - 0.10 * vol.loc[idx].rank(pct=True))

    n_valid = len(idx)
    n_side = min(30, n_valid // 2)
    if n_side < 3:
        return zero

    order = scores.sort_values(ascending=False).index
    top = order[:n_side]
    bottom = order[-n_side:]

    w = min(0.02, 0.5 / float(n_side))
    if 2.0 * n_side * w > 1.0:
        w = 0.5 / float(n_side) * (1.0 - 1e-12)

    weights = pd.Series(0.0, index=tickers, dtype=float)
    weights.loc[top] = w
    weights.loc[bottom] = -w
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights