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
    if adj.shape[0] < 6:
        return weights

    # Use only recent rows for speed; all calculations remain point-in-time.
    adj = adj.tail(120).copy()
    if adj.shape[0] < 6:
        return weights

    ret1 = adj / adj.shift(1) - 1.0
    vol = ret1.tail(20).std(ddof=1)
    vol = vol.clip(lower=0.005)

    lookback = 5
    if adj.shape[0] > lookback:
        ret_l = adj.iloc[-1] / adj.iloc[-1 - lookback] - 1.0
    else:
        ret_l = pd.Series(np.nan, index=universe, dtype=float)

    volume = history.get('volume')
    if volume is not None:
        volume = volume.reindex(columns=universe).tail(120)
        dollar_vol = (adj * volume).tail(20).median()
    else:
        dollar_vol = pd.Series(1.0, index=universe, dtype=float)

    eligible = history.get('eligible')
    if eligible is not None:
        eligible = eligible.reindex(columns=universe).iloc[-1]
        eligible = eligible.fillna(False).astype(bool)
    else:
        eligible = pd.Series(True, index=universe, dtype=bool)

    last_adj = adj.iloc[-1]
    valid = (
        eligible
        & np.isfinite(last_adj)
        & (last_adj > 0)
        & np.isfinite(ret_l)
        & np.isfinite(vol)
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

    score = -ret_l / vol
    score = score.where(liq_ok)
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    n_cand = len(score)
    if n_cand < 2:
        return weights

    max_side = 25
    n_side = min(max_side, n_cand // 2)
    if n_side <= 0:
        return weights

    ordered = score.sort_values(kind='mergesort')
    short_names = ordered.index[:n_side]
    long_names = ordered.index[-n_side:]

    w = min(0.02, 0.95 / (2.0 * n_side))
    weights.loc[long_names] = w
    weights.loc[short_names] = -w

    long_sum = weights[weights > 0].sum()
    short_sum = -weights[weights < 0].sum()
    diff = long_sum - short_sum
    if abs(diff) > 1e-12 and n_side > 0:
        adj_w = diff / (2.0 * n_side)
        weights.loc[long_names] = weights.loc[long_names] - adj_w
        weights.loc[short_names] = weights.loc[short_names] + adj_w

    return weights
