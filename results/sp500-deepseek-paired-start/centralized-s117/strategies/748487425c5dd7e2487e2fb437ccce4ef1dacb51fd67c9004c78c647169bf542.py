import numpy as np
import pandas as pd
import math


def _get_df(history, names):
    for name in names:
        obj = history.get(name)
        if isinstance(obj, pd.DataFrame):
            return obj
    return None


def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)

    price = _get_df(history, ['adjusted_close', 'close'])
    if price is None:
        return pd.Series(0.0, index=universe)

    price = price.reindex(columns=universe).iloc[-260:]
    if price.shape[0] < 25:
        return pd.Series(0.0, index=universe)

    last = price.iloc[-1]
    cnt = price.notna().sum()
    valid = last.notna() & (last > 2.0) & (cnt >= 60)

    vol_df = _get_df(history, ['volume'])
    if vol_df is not None:
        vol_df = vol_df.reindex(columns=universe).iloc[-60:]
        vmed = vol_df.tail(20).median()
        valid = valid & (vmed > 100000)

    elig_df = _get_df(history, ['eligible'])
    if elig_df is not None:
        e = elig_df.reindex(columns=universe).iloc[-1]
        valid = valid & e.fillna(True).astype(bool)

    rets = price / price.shift(1) - 1.0
    vol = rets.tail(60).std()
    vol = vol.replace(0.0, np.nan).clip(lower=0.005)

    def form(k):
        if price.shape[0] >= k + 1:
            return price.iloc[-1] / price.iloc[-k - 1] - 1.0
        return pd.Series(np.nan, index=universe)

    ret21 = form(21)
    ret5 = form(5)

    valid = valid & ret21.notna() & (ret21 < 1.5) & (ret21 > -0.75)
    if valid.sum() < 4:
        return pd.Series(0.0, index=universe)

    m21 = ret21[valid].mean()
    m5 = ret5[valid].mean()
    z21 = (ret21 - m21) / (vol * math.sqrt(21.0))
    z5 = (ret5 - m5) / (vol * math.sqrt(5.0))

    score = -(0.8 * z21 + 0.2 * z5)
    score = score.where(valid).replace([np.inf, -np.inf], np.nan).dropna()

    if len(score) < 10:
        score = (-(0.8 * ret21 + 0.2 * ret5)).where(valid)
        score = score.replace([np.inf, -np.inf], np.nan).dropna()

    if len(score) < 2:
        return pd.Series(0.0, index=universe)

    m = len(score)
    n_side = min(50, m // 2)
    if n_side < 1:
        n_side = 1

    sorted_score = score.sort_values(ascending=False)
    long_names = sorted_score.index[:n_side]
    short_names = sorted_score.index[-n_side:]

    w = min(0.5 / n_side, 0.02)
    weights = pd.Series(0.0, index=universe)
    weights.loc[long_names] += w
    weights.loc[short_names] -= w

    net = weights.sum()
    if abs(net) > 1e-12:
        idx = weights.index[weights != 0.0]
        if len(idx) > 0:
            weights.loc[idx] -= net / len(idx)

    weights = weights.clip(-0.02, 0.02)
    gross = weights.abs().sum()
    if gross > 1.0:
        weights = weights * (1.0 / gross)

    return weights.fillna(0.0).astype(float)