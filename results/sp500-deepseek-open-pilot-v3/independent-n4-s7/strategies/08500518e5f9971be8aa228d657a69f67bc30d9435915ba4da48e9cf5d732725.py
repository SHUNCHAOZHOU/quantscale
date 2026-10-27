import numpy as np
import pandas as pd


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)

    adj = history.get('adjusted_close')
    if adj is None or adj.empty or len(universe) == 0:
        return zeros

    dates = adj.index
    if len(dates) < 30:
        return zeros

    latest = dates[-1]

    # Rebalance on the first trading day of each month.  On later days in the
    # month we recompute the same signal from that first-day close, so the
    # portfolio is effectively held until the next month.
    try:
        cur_period = latest.to_period('M')
        cur_dates = dates[dates.to_period('M') == cur_period]
        if len(cur_dates) == 0:
            return zeros
        rebal = cur_dates[0]
    except Exception:
        rebal = latest

    adj_r = adj.loc[:rebal]
    if len(adj_r) < 30:
        return zeros

    tickers = universe.intersection(adj_r.columns)
    if len(tickers) < 20:
        return zeros

    # Keep a bounded window for speed.
    win = adj_r.iloc[-160:].reindex(columns=tickers).ffill()
    n = len(win)
    if n < 30:
        return zeros

    last = win.iloc[-1]

    # One-month reversal.
    if n >= 22:
        r21 = last / win.iloc[-22] - 1.0
    else:
        r21 = pd.Series(np.nan, index=tickers)

    # Intermediate trend filter: price relative to 100-day mean.
    look = min(100, n)
    if look >= 40:
        sma = win.iloc[-look:].mean()
        trend = last / sma - 1.0
    else:
        trend = pd.Series(np.nan, index=tickers)

    # Liquidity: median 20-day dollar volume.
    vol_df = history.get('volume')
    if vol_df is not None and not vol_df.empty:
        vol_win = vol_df.reindex(index=win.index, columns=tickers)
        dv = vol_win * win
        if len(dv) >= 20:
            liq = dv.iloc[-20:].median()
        else:
            liq = dv.median()
    else:
        liq = pd.Series(1.0, index=tickers)

    # Require eligibility at both the rebalance date and the current close.
    elig = history.get('eligible')
    if elig is not None and not elig.empty:
        el_r = elig.loc[:rebal].reindex(columns=tickers)
        if len(el_r) > 0:
            el = el_r.iloc[-1].fillna(False).astype(bool)
        else:
            el = pd.Series(False, index=tickers)

        el_now_df = elig.reindex(columns=tickers)
        if len(el_now_df) > 0:
            el_now = el_now_df.iloc[-1].fillna(False).astype(bool)
        else:
            el_now = pd.Series(True, index=tickers)
    else:
        el = pd.Series(True, index=tickers)
        el_now = pd.Series(True, index=tickers)

    valid = (
        r21.notna()
        & trend.notna()
        & liq.notna()
        & (liq > 0)
        & el
        & el_now
        & np.isfinite(r21)
        & np.isfinite(trend)
        & np.isfinite(liq)
    )

    if valid.sum() < 20:
        return zeros

    # Drop the least liquid names, more aggressively when many candidates exist.
    if valid.sum() >= 100:
        q = liq[valid].quantile(0.2)
        valid = valid & (liq >= q)
    elif valid.sum() >= 50:
        q = liq[valid].quantile(0.1)
        valid = valid & (liq >= q)

    cand = valid[valid].index
    if len(cand) < 20:
        return zeros

    def rk(s):
        return s.rank(pct=True)

    # Long: low one-month return but relatively strong trend.
    # Short: high one-month return but relatively weak trend.
    score = -rk(r21[cand]) + 0.5 * rk(trend[cand])
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 20:
        return zeros

    score = score.sort_values(ascending=False)
    n_side = min(50, len(score) // 2)
    if n_side < 10:
        return zeros

    longs = list(score.index[:n_side])
    shorts = list(score.index[-n_side:])

    overlap = set(longs).intersection(set(shorts))
    if overlap:
        longs = [t for t in longs if t not in overlap]
        shorts = [t for t in shorts if t not in overlap]
        n_side = min(len(longs), len(shorts))
        if n_side < 10:
            return zeros
        longs = longs[:n_side]
        shorts = shorts[-n_side:]

    # Dollar-neutral, low turnover, with gross exposure slightly below the limit.
    w = min(0.02, 0.49 / n_side)
    weights = pd.Series(0.0, index=universe, dtype=float)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.fillna(0.0).clip(-0.02, 0.02)

    return weights
