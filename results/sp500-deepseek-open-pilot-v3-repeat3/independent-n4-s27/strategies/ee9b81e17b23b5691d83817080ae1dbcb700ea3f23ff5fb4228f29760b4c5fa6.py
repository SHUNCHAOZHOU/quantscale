import numpy as np
import pandas as pd

_CACHE = {}

def _zero_weights(universe):
    return pd.Series(0.0, index=universe, dtype=float)

def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)

    price_key = 'adjusted_close' if 'adjusted_close' in history else 'close'
    px_all = history[price_key]
    if px_all is None or px_all.empty:
        return _zero_weights(universe)

    idx = px_all.index
    if len(idx) == 0:
        return _zero_weights(universe)

    # Rebalance on the first trading day of each month, using only data through that close.
    months = idx.month.values
    is_first = np.empty(len(idx), dtype=bool)
    is_first[0] = True
    if len(idx) > 1:
        is_first[1:] = months[1:] != months[:-1]
    pos = np.where(is_first)[0][-1]
    reb_date = idx[pos]

    cache_key = (reb_date, tuple(universe))
    if cache_key in _CACHE:
        return _CACHE[cache_key].copy()

    px = px_all.loc[:reb_date]
    if len(px) < 253:
        w = _zero_weights(universe)
        _CACHE[cache_key] = w
        return w.copy()

    px = px.reindex(columns=universe).astype(float)
    n = len(px)

    # 12-1 momentum: price 21 sessions ago divided by price 252 sessions ago.
    p_start = px.iloc[n - 253]
    p_end = px.iloc[n - 22]
    mom = p_end / p_start - 1.0

    # Daily volatility over the last 126 sessions.
    rets = px.pct_change()
    vol = rets.iloc[-126:].std(ddof=0)

    last_px = px.iloc[-1]
    valid = (
        mom.notna()
        & vol.notna()
        & (vol > 1e-8)
        & last_px.notna()
        & p_start.notna()
        & p_end.notna()
    )

    score = mom / vol
    score = score[valid]
    if len(score) < 2:
        w = _zero_weights(universe)
        _CACHE[cache_key] = w
        return w.copy()

    score = score.sort_values(ascending=False)
    k = min(25, len(score) // 2)
    if k == 0:
        w = _zero_weights(universe)
        _CACHE[cache_key] = w
        return w.copy()

    longs = score.index[:k]
    shorts = score.index[-k:]

    w = pd.Series(0.0, index=universe, dtype=float)
    w.loc[longs] = 0.02
    w.loc[shorts] = -0.02

    # Equal long and short counts make the portfolio dollar neutral.
    _CACHE[cache_key] = w
    return w.copy()
