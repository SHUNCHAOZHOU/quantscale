import numpy as np
import pandas as pd

def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe)

    adj = history.get('adjusted_close')
    if adj is None:
        return zero

    cols = [c for c in universe if c in adj.columns]
    if len(cols) < 10:
        return zero

    if not adj.index.is_monotonic_increasing:
        adj = adj.sort_index()

    n_total = len(adj)
    if n_total < 300:
        return zero

    current_i = n_total - 1
    # Rebalance every 20 trading days starting from index 300
    last_reb_i = 300 + ((current_i - 300) // 20) * 20

    # Slice data up to the last rebalance date (inclusive)
    adj_slice = adj.iloc[:last_reb_i + 1][cols].copy()
    if len(adj_slice) < 273:  # need at least 252 + 21 days for momentum
        return zero

    t = len(adj_slice) - 1
    if t < 252:
        return zero

    # Prices for momentum
    px_t = adj_slice.iloc[t]
    px_t21 = adj_slice.iloc[t - 21]
    px_t126 = adj_slice.iloc[t - 126]
    px_t252 = adj_slice.iloc[t - 252]

    mom12 = px_t21 / px_t252 - 1.0
    mom6 = px_t21 / px_t126 - 1.0

    # Volatility over past 126 days
    rets = adj_slice.pct_change().iloc[-126:]
    vol = rets.std() * np.sqrt(252.0)

    mom12 = mom12.replace([np.inf, -np.inf], np.nan)
    mom6 = mom6.replace([np.inf, -np.inf], np.nan)
    vol = vol.replace([np.inf, -np.inf], np.nan)

    valid = px_t.notna() & (px_t > 0) & mom12.notna() & mom6.notna() & vol.notna() & (vol > 0)

    # Eligible filter
    elig = history.get('eligible')
    if elig is not None:
        if not elig.index.is_monotonic_increasing:
            elig = elig.sort_index()
        if len(elig) > last_reb_i:
            e_last = elig.iloc[last_reb_i]
            e_last = e_last.reindex(cols).fillna(False).astype(bool)
            valid = valid & e_last

    valid_stocks = valid[valid].index
    if len(valid_stocks) < 10:
        return zero

    mom12_v = mom12.loc[valid_stocks]
    mom6_v = mom6.loc[valid_stocks]
    vol_v = vol.loc[valid_stocks]

    def rp(s):
        return s.rank(method='average', pct=True)

    # Sign-flipped momentum: long losers, short winners; prefer low vol
    score = -0.5 * rp(mom12_v) - 0.3 * rp(mom6_v) - 0.2 * rp(vol_v)
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 10:
        return zero

    n_avail = len(score)
    n_long = min(30, max(5, n_avail // 3))
    n_short = min(60, n_avail - n_long)
    if n_long < 1 or n_short < 1:
        return zero

    sorted_score = score.sort_values(ascending=False)
    longs = sorted_score.index[:n_long]
    shorts = sorted_score.index[-n_short:]

    w_long = 0.5 / n_long
    w_short = 0.5 / n_short

    # Enforce max absolute weight 0.02
    if w_long > 0.02:
        w_long = 0.02
        total_long = n_long * w_long
        w_short = total_long / n_short
        if w_short > 0.02:
            w_short = 0.02
            total_short = n_short * w_short
            w_long = total_short / n_long

    weights = pd.Series(0.0, index=universe)
    weights.loc[longs] = w_long
    weights.loc[shorts] = -w_short

    # Final safety: ensure gross <= 1 and net near zero
    gross = weights.abs().sum()
    net = weights.sum()
    if gross > 1.0 + 1e-12:
        weights = weights / gross
        net = weights.sum()
    if abs(net) > 0.01:
        active = weights[weights != 0]
        if len(active) > 0:
            adj = net / len(active)
            weights.loc[active.index] = active - adj

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights.reindex(universe).fillna(0.0)