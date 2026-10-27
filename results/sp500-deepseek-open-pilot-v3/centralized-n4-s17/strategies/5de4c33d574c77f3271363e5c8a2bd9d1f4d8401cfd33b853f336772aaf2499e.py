import numpy as np
import pandas as pd


def generate_weights(history, universe):
    universe = pd.Index(universe)
    weights = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return weights

    close = history.get('close')
    adj = history.get('adjusted_close')
    if adj is None:
        adj = close
    elif close is not None:
        adj = adj.reindex(columns=universe)
        close = close.reindex(columns=universe)
        adj = adj.where(adj.notna(), close)
    if adj is None:
        return weights

    adj = adj.reindex(columns=universe)
    if adj.shape[0] < 25:
        return weights

    # Rebalance every 5 trading days from the start of the history.
    n_rows = adj.shape[0]
    rebal_idx = n_rows - (n_rows % 5) - 1  # zero-based index of last rebalance row
    if rebal_idx < 19:  # need at least 20 rows for vol and 5 for R5
        return weights

    # Use only recent rows (up to rebal_idx) for speed; all data is point-in-time.
    start_idx = max(0, rebal_idx - 120)
    adj_sub = adj.iloc[start_idx:rebal_idx+1].copy()
    if adj_sub.shape[0] < 21:
        return weights

    # Daily returns and 20-day volatility series
    ret1 = adj_sub / adj_sub.shift(1) - 1.0
    vol_series = ret1.rolling(20).std(ddof=1).clip(lower=0.005)

    # Compute smoothed score: average of last 3 daily -R5/vol scores up to rebal_idx
    lookback = 5
    score_days = []
    for i in range(1, 4):  # 1, 2, 3 days back from rebal_idx
        idx = adj_sub.shape[0] - i
        if idx < lookback or idx < 19:
            continue
        ret_l = adj_sub.iloc[idx] / adj_sub.iloc[idx - lookback] - 1.0
        vol_t = vol_series.iloc[idx]
        if np.isfinite(ret_l) and np.isfinite(vol_t):
            score_days.append(-ret_l / vol_t)
    if len(score_days) == 0:
        return weights
    score = sum(score_days) / len(score_days)

    # Volume data for liquidity filter (use data up to rebal_idx)
    volume = history.get('volume')
    if volume is not None:
        volume = volume.reindex(columns=universe).iloc[start_idx:rebal_idx+1]
        dollar_vol = (adj_sub * volume).tail(20).median()
    else:
        dollar_vol = pd.Series(1.0, index=universe, dtype=float)

    # Eligible flag at rebal_idx
    eligible = history.get('eligible')
    if eligible is not None:
        eligible = eligible.reindex(columns=universe).iloc[rebal_idx]
        eligible = eligible.fillna(False).astype(bool)
    else:
        eligible = pd.Series(True, index=universe, dtype=bool)

    last_adj = adj_sub.iloc[-1]
    valid = (
        eligible
        & np.isfinite(last_adj)
        & (last_adj > 0)
        & np.isfinite(score)
        & np.isfinite(dollar_vol)
    )
    if valid.sum() < 4:
        return weights

    # Liquidity filter: keep names above the 40th percentile of dollar volume.
    liq_ok = valid.copy()
    if valid.sum() >= 20:
        thresh = dollar_vol[valid].quantile(0.4)
        if np.isfinite(thresh):
            liq_ok = valid & (dollar_vol >= thresh)
            if liq_ok.sum() < 10:
                liq_ok = valid

    score = score.where(liq_ok)
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    n_cand = len(score)
    if n_cand < 2:
        return weights

    max_side = 20
    n_side = min(max_side, n_cand // 2)
    if n_side <= 0:
        return weights

    ordered = score.sort_values(kind='mergesort')
    short_names = ordered.index[:n_side]
    long_names = ordered.index[-n_side:]

    w = min(0.02, 0.95 / (2.0 * n_side))
    weights.loc[long_names] = w
    weights.loc[short_names] = -w

    # Ensure dollar neutrality
    long_sum = weights[weights > 0].sum()
    short_sum = -weights[weights < 0].sum()
    diff = long_sum - short_sum
    if abs(diff) > 1e-12 and n_side > 0:
        adj_w = diff / (2.0 * n_side)
        weights.loc[long_names] = weights.loc[long_names] - adj_w
        weights.loc[short_names] = weights.loc[short_names] + adj_w

    return weights
