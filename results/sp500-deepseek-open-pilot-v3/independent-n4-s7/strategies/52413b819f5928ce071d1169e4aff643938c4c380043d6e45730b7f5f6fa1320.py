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

    # Rebalance on the first trading day of each month.  On all later days
    # in the month, hold the portfolio selected from that first-day close.
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
    if len(tickers) == 0:
        return zeros

    # Keep the computation window bounded for speed.
    win = adj_r.iloc[-300:].reindex(columns=tickers).ffill()
    n = len(win)
    if n < 25:
        return zeros

    last = win.iloc[-1]

    def get_ret(lookback, skip=21):
        end_idx = n - 1 - skip
        if end_idx < 1:
            return pd.Series(np.nan, index=tickers)
        start_idx = end_idx - lookback
        if start_idx < 0:
            start_idx = 0
        if start_idx >= end_idx:
            return pd.Series(np.nan, index=tickers)
        p0 = win.iloc[start_idx]
        p1 = win.iloc[end_idx]
        return p1 / p0 - 1.0

    r3 = get_ret(63, 21)
    r6 = get_ret(126, 21)
    r12 = get_ret(252, 21)

    mom = r6.copy()
    mom = mom.fillna(r12).fillna(r3)

    look_trend = min(200, n)
    sma = win.iloc[-look_trend:].mean()
    trend = last / sma - 1.0

    rets = win.pct_change()
    vol = rets.iloc[-120:].std()

    vol_df = history.get('volume')
    if vol_df is not None and not vol_df.empty:
        vol_win = vol_df.reindex(index=win.index, columns=tickers)
        dv = vol_win * win
        liq = dv.iloc[-60:].median()
    else:
        liq = pd.Series(1.0, index=tickers)

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

    valid = mom.notna() & trend.notna() & liq.notna() & (liq > 0) & el & el_now
    valid = valid & np.isfinite(mom) & np.isfinite(trend) & np.isfinite(liq)

    if valid.sum() >= 100:
        q = liq[valid].quantile(0.3)
        valid = valid & (liq >= q)
    elif valid.sum() >= 50:
        q = liq[valid].quantile(0.15)
        valid = valid & (liq >= q)

    if valid.sum() < 20:
        return zeros

    cand = valid[valid].index

    def rk(s):
        return s.rank(pct=True, na_option='keep')

    vol_filled = vol.copy()
    if vol_filled.notna().any():
        vol_filled = vol_filled.fillna(vol_filled.median())
    else:
        vol_filled = pd.Series(0.0, index=tickers)

    score = (
        0.45 * rk(mom[cand]).fillna(0.5)
        + 0.25 * rk(r12[cand]).fillna(0.5)
        + 0.15 * rk(r3[cand]).fillna(0.5)
        + 0.15 * rk(trend[cand]).fillna(0.5)
        - 0.10 * rk(vol_filled[cand]).fillna(0.5)
    )

    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 20:
        return zeros

    score = score.sort_values(ascending=False)
    n_side = min(40, len(score) // 2)
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

    w = min(0.02, 0.5 / n_side)
    weights = pd.Series(0.0, index=universe, dtype=float)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.fillna(0.0).clip(-0.02, 0.02)

    return weights
