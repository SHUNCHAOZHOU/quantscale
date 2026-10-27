import pandas as pd
import numpy as np


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)

    try:
        adj_close = history['adjusted_close']
    except Exception:
        return zeros

    if adj_close is None or adj_close.shape[0] == 0 or adj_close.shape[1] == 0:
        return zeros

    dates = adj_close.index
    if len(dates) < 61:
        return zeros

    current_date = dates[-1]

    # Last month-end trading date in the available history.
    s = pd.Series(dates, index=dates)
    month_ends = s.groupby([dates.year, dates.month]).max()
    candidates = month_ends[month_ends <= current_date]
    rebal_date = current_date if len(candidates) == 0 else candidates.iloc[-1]

    idx = adj_close.index.get_loc(rebal_date)
    if idx < 60:
        return zeros

    close_hist = adj_close.iloc[:idx + 1]
    prices = close_hist.iloc[-61:]
    rets = prices.pct_change().iloc[1:]
    vol = rets.std(axis=0, ddof=1) * np.sqrt(252.0)

    vol_hist = history['volume'].iloc[:idx + 1].iloc[-20:]
    close_20 = close_hist.iloc[-20:]
    dollar = (close_20 * vol_hist).mean(axis=0)
    price = close_hist.iloc[-1]

    if 'eligible' in history:
        try:
            elig = history['eligible'].iloc[idx]
        except Exception:
            elig = pd.Series(True, index=adj_close.columns)
    else:
        elig = pd.Series(True, index=adj_close.columns)

    v = vol.reindex(universe)
    d = dollar.reindex(universe)
    p = price.reindex(universe)
    e = elig.reindex(universe).fillna(False).astype(bool)

    mask = (
        v.notna()
        & d.notna()
        & p.notna()
        & (v > 0.0)
        & (p > 1.0)
        & (d > 1_000_000.0)
        & e
    )

    candidates = universe[mask]
    if len(candidates) < 10:
        return zeros

    n_side = min(25, len(candidates) // 2)
    if n_side < 5:
        return zeros

    w = min(0.02, 0.5 / n_side)
    vals = v.loc[candidates].values
    order = np.argsort(vals)
    sorted_cands = candidates[order]

    long_names = sorted_cands[:n_side]
    short_names = sorted_cands[-n_side:]

    weights = pd.Series(0.0, index=universe, dtype=float)
    weights.loc[long_names] = w
    weights.loc[short_names] = -w

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    weights = weights.clip(-0.02, 0.02)

    gross = weights.abs().sum()
    if gross > 1.0:
        weights = weights / gross

    return weights