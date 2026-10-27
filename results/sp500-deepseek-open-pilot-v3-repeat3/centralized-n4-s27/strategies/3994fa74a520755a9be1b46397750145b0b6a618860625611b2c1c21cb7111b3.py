import numpy as np
import pandas as pd

def generate_weights(history, universe):
    universe = pd.Index(universe)
    n = len(universe)
    weights = pd.Series(0.0, index=universe, dtype=float)
    if n == 0:
        return weights

    price_field = None
    df = None
    for f in ('adjusted_close', 'close', 'adjusted_open', 'open'):
        if f in history and isinstance(history[f], pd.DataFrame):
            trial = history[f].reindex(columns=universe)
            if trial.shape[0] > 0:
                last_trial = trial.iloc[-1].to_numpy(dtype=float)
                if np.isfinite(last_trial).any():
                    price_field = f
                    df = trial
                    break
    if df is None or df.shape[0] < 6:
        return weights

    if 'eligible' in history and isinstance(history['eligible'], pd.DataFrame):
        el = history['eligible'].reindex(columns=universe).iloc[-1]
        elig = el.fillna(False).astype(bool).to_numpy()
    else:
        elig = np.ones(n, dtype=bool)

    last = df.iloc[-1].to_numpy(dtype=float)
    valid = elig & np.isfinite(last) & (last > 0.0)
    if valid.sum() < 2:
        return weights

    def ret(L):
        if df.shape[0] <= L:
            return np.full(n, np.nan)
        a = df.iloc[-1].to_numpy(dtype=float)
        b = df.iloc[-1 - L].to_numpy(dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = a / b - 1.0
        r[~np.isfinite(r)] = np.nan
        return r

    mom120 = ret(120)
    mom60 = ret(60)
    rev5 = ret(5)

    def rank_pct(x):
        return pd.Series(x, index=universe, dtype=float).rank(pct=True, method='average')

    r120 = rank_pct(mom120)
    r60 = rank_pct(mom60)
    r5 = rank_pct(rev5)

    mom_score = 0.6 * r120.fillna(r60) + 0.4 * r60.fillna(r120)
    score = mom_score - 0.25 * r5

    score_arr = score.to_numpy(dtype=float)
    cand = valid & np.isfinite(score_arr)
    idx = np.flatnonzero(cand)
    if idx.size < 2:
        return weights

    N = min(25, idx.size // 2)
    if N <= 0:
        return weights

    order = idx[np.argsort(score_arr[idx], kind='mergesort')]
    short_idx = order[:N]
    long_idx = order[-N:]
    w = min(0.02, 1.0 / (2.0 * N))

    weights.iloc[long_idx] = w
    weights.iloc[short_idx] = -w
    return weights