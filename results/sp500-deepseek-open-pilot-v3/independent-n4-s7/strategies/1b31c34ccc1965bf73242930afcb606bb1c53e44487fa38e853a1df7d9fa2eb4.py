import numpy as np
import pandas as pd


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)

    adj = history.get('adjusted_close')
    if adj is None or len(adj.index) == 0:
        return zeros

    dates = adj.index
    latest = dates[-1]

    # Rebalance on the last trading day of each week, so the new portfolio
    # is executed at the next open.  On other days hold the prior week's
    # signal to avoid unnecessary trading.
    try:
        current_week = latest.to_period('W')
        week_dates = dates[dates.to_period('W') == current_week]
        if len(week_dates) > 0 and latest == week_dates[-1]:
            rebal = latest
        else:
            prev_dates = dates[dates.to_period('W') < current_week]
            if len(prev_dates) == 0:
                return zeros
            rebal = prev_dates[-1]
    except Exception:
        rebal = latest

    adj_r = adj.loc[:rebal]
    if len(adj_r) < 6:
        return zeros

    tickers = universe.intersection(adj_r.columns)
    if len(tickers) == 0:
        return zeros

    adj_sub = adj_r[tickers]
    last = adj_sub.iloc[-1]

    # Five-session reversal.
    ret5 = last / adj_sub.iloc[-6] - 1.0

    # Intermediate trend: price relative to a 100-day mean.
    look = min(100, len(adj_sub))
    if look >= 40:
        sma = adj_sub.iloc[-look:].mean()
        trend = last / sma - 1.0
    else:
        trend = pd.Series(np.nan, index=tickers)

    # Liquidity filter using median 20-day dollar volume.
    vol = history.get('volume')
    if vol is not None:
        vol_r = vol.loc[:rebal].reindex(columns=tickers)
        dv = vol_r * adj_sub
        if len(dv) >= 20:
            liq = dv.iloc[-20:].median()
        else:
            liq = dv.median()
    else:
        liq = pd.Series(1.0, index=tickers)

    # Require both rebalance-date and current eligibility.
    elig = history.get('eligible')
    if elig is not None:
        el_r = elig.loc[:rebal].reindex(columns=tickers)
        if len(el_r) > 0:
            el = el_r.iloc[-1].fillna(False).astype(bool)
        else:
            el = pd.Series(False, index=tickers)
        el_now_df = elig.reindex(columns=tickers)
        if len(el_now_df) > 0:
            el_now = el_now_df.iloc[-1].fillna(False).astype(bool)
        else:
            el_now = pd.Series(False, index=tickers)
    else:
        el = pd.Series(True, index=tickers)
        el_now = pd.Series(True, index=tickers)

    valid = ret5.notna() & trend.notna() & liq.notna() & (liq > 0) & el & el_now
    valid = valid & np.isfinite(ret5) & np.isfinite(trend) & np.isfinite(liq)
    if valid.sum() < 10:
        return zeros

    # Drop the least liquid names, more aggressively when the candidate set is large.
    if valid.sum() >= 100:
        q = liq[valid].quantile(0.2)
        valid = valid & (liq >= q)
    elif valid.sum() >= 50:
        q = liq[valid].quantile(0.1)
        valid = valid & (liq >= q)

    candidates = valid[valid].index
    if len(candidates) < 10:
        return zeros

    r_ret = ret5[candidates].rank(pct=True)
    r_trend = trend[candidates].rank(pct=True)

    # Long: low recent return, high trend.  Short: high recent return, low trend.
    score = -r_ret + 0.5 * r_trend
    score = score.sort_values(ascending=False)

    n_side = min(50, len(score) // 2)
    if n_side < 10:
        return zeros

    longs = score.index[:n_side]
    shorts = score.index[-n_side:]

    # Guard against any accidental overlap (should not occur with unique scores).
    overlap = set(longs).intersection(set(shorts))
    if len(overlap) > 0:
        shorts = [t for t in shorts if t not in overlap]
        n_side = min(len(longs), len(shorts))
        if n_side < 10:
            return zeros
        longs = longs[:n_side]
        shorts = shorts[-n_side:]

    total = min(0.5, 0.02 * n_side)
    w_long = total / n_side
    w_short = -total / n_side

    weights = pd.Series(0.0, index=universe, dtype=float)
    weights.loc[longs] = w_long
    weights.loc[shorts] = w_short
    weights = weights.fillna(0.0).clip(-0.02, 0.02)

    return weights
