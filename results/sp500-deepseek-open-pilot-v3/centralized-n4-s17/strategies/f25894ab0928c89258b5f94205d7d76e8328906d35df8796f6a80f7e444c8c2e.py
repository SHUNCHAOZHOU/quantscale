import numpy as np
import pandas as pd


def generate_weights(history, universe):
    universe = pd.Index(universe)
    weights = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return weights

    # Get adjusted close, fallback to close
    adj = history.get('adjusted_close')
    if adj is None:
        adj = history.get('close')
    else:
        close = history.get('close')
        if close is not None:
            adj = adj.reindex(columns=universe)
            close = close.reindex(columns=universe)
            adj = adj.where(adj.notna(), close)
    if adj is None:
        return weights
    adj = adj.reindex(columns=universe)
    n = adj.shape[0]
    if n < 61:
        return weights

    # Rebalance every 10 trading days
    N = 10
    if n % N == 0:
        rebal_idx = n - 1
    else:
        rebal_idx = (n // N) * N - 1
    if rebal_idx < 60:
        return weights

    # Use recent window for speed
    start_idx = max(0, rebal_idx - 120)
    adj_sub = adj.iloc[start_idx:rebal_idx+1].copy()
    if adj_sub.shape[0] < 61:
        return weights

    # Volume for liquidity
    volume = history.get('volume')
    if volume is not None:
        volume_sub = volume.reindex(columns=universe).iloc[start_idx:rebal_idx+1]
        dollar_vol = (adj_sub * volume_sub).tail(20).median()
    else:
        dollar_vol = pd.Series(1.0, index=universe, dtype=float)

    # Eligible flag at rebalance
    eligible = history.get('eligible')
    if eligible is not None:
        eligible = eligible.reindex(columns=universe).iloc[rebal_idx]
        eligible = eligible.fillna(False).astype(bool)
    else:
        eligible = pd.Series(True, index=universe, dtype=bool)

    # Daily returns and volatility
    ret1 = adj_sub / adj_sub.shift(1) - 1.0
    vol20 = ret1.tail(20).std(ddof=1).clip(lower=0.005)

    # Returns over formation periods
    def calc_ret(lookback):
        if adj_sub.shape[0] > lookback:
            return adj_sub.iloc[-1] / adj_sub.iloc[-1-lookback] - 1.0
        else:
            return pd.Series(np.nan, index=universe, dtype=float)

    ret5 = calc_ret(5)
    ret20 = calc_ret(20)
    ret60 = calc_ret(60)

    last_adj = adj_sub.iloc[-1]
    valid = (
        eligible
        & np.isfinite(last_adj)
        & (last_adj > 0)
        & np.isfinite(ret5)
        & np.isfinite(ret20)
        & np.isfinite(ret60)
        & np.isfinite(vol20)
        & np.isfinite(dollar_vol)
    )
    if valid.sum() < 10:
        return weights

    # Liquidity filter: top half by dollar volume among valid
    if valid.sum() >= 20:
        thresh = dollar_vol[valid].quantile(0.5)
        if np.isfinite(thresh):
            liq_ok = valid & (dollar_vol >= thresh)
            if liq_ok.sum() < 10:
                liq_ok = valid
        else:
            liq_ok = valid
    else:
        liq_ok = valid

    # Combined reversal signal: average of 5-day and 20-day returns
    rev = 0.5 * ret5 + 0.5 * ret20

    # Residualize rev against 60-day momentum cross-sectionally
    y = rev[liq_ok].values
    x = ret60[liq_ok].values
    if len(y) >= 10:
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta
            resid_series = pd.Series(resid, index=rev[liq_ok].index)
        except np.linalg.LinAlgError:
            resid_series = rev[liq_ok]
    else:
        resid_series = rev[liq_ok]

    # Score: negative residual reversal scaled by volatility
    score = -resid_series / vol20[resid_series.index]
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    n_cand = len(score)
    if n_cand < 4:
        return weights

    # Select top and bottom names
    max_side = 20
    n_side = min(max_side, n_cand // 2)
    if n_side <= 0:
        return weights

    ordered = score.sort_values(kind='mergesort')
    short_names = ordered.index[:n_side]   # low score -> short
    long_names = ordered.index[-n_side:]   # high score -> long

    w = min(0.02, 1.0 / (2.0 * n_side))
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

    # Clip to max absolute weight
    weights = weights.clip(-0.02, 0.02)

    return weights
