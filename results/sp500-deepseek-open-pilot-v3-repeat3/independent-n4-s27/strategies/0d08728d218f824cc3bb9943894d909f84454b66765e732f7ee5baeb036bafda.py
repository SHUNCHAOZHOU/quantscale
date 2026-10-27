import numpy as np
import pandas as pd

_CACHE = {}

def _zero_weights(universe):
    return pd.Series(0.0, index=universe, dtype=float)

def _find_weekly_rebalance_date(idx):
    """Return the most recent trading date that is the first trading day of its week."""
    if len(idx) == 0:
        return None
    iso = idx.isocalendar()
    weeks = iso.year * 100 + iso.week
    diff = np.empty(len(idx), dtype=bool)
    diff[0] = True
    diff[1:] = weeks[1:] != weeks[:-1]
    pos = np.where(diff)[0][-1]
    return idx[pos]

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

    reb_date = _find_weekly_rebalance_date(idx)
    if reb_date is None:
        return _zero_weights(universe)

    cache_key = (reb_date, tuple(universe))
    if cache_key in _CACHE:
        return _CACHE[cache_key].copy()

    px = px_all.loc[:reb_date].reindex(columns=universe).astype(float)
    n = len(px)
    if n < 10:
        w = _zero_weights(universe)
        _CACHE[cache_key] = w
        return w.copy()

    last = px.iloc[-1]
    # 5-day return
    if n >= 6:
        ret5 = last / px.iloc[-6] - 1.0
    else:
        ret5 = pd.Series(np.nan, index=universe)

    # 20-day volatility
    rets = px.pct_change()
    vol_window = min(20, n - 1)
    if vol_window >= 5:
        vol = rets.iloc[-vol_window:].std(ddof=0)
    else:
        vol = pd.Series(np.nan, index=universe)

    valid = last.notna() & (last > 0) & ret5.notna() & vol.notna() & (vol > 1e-8)
    valid_idx = valid[valid].index
    if len(valid_idx) < 4:
        w = _zero_weights(universe)
        _CACHE[cache_key] = w
        return w.copy()

    # Reversal score: negative 5-day return divided by volatility.
    # Higher score = more negative return (loser) and lower volatility.
    score = -ret5 / vol
    score = score.reindex(valid_idx).replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 4:
        w = _zero_weights(universe)
        _CACHE[cache_key] = w
        return w.copy()

    k = min(25, len(score) // 2)
    if k < 1:
        w = _zero_weights(universe)
        _CACHE[cache_key] = w
        return w.copy()

    score = score.sort_values(ascending=False)
    longs = score.index[:k]
    shorts = score.index[-k:]

    w = _zero_weights(universe)
    w.loc[longs] = 0.02
    w.loc[shorts] = -0.02

    _CACHE[cache_key] = w
    return w.copy()
