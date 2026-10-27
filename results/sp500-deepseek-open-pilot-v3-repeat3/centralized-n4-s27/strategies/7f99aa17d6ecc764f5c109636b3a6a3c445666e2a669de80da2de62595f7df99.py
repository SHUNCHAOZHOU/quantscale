import numpy as np
import pandas as pd

_cache = {}

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

    K = 3
    last_date = df.index[-1]
    cache_key = (last_date, n, universe[0] if n > 0 else '')
    if cache_key in _cache:
        return _cache[cache_key].copy()

    if df.shape[0] % K != 0 and _cache:
        prev_dates = [k[0] for k in _cache.keys() if k[0] < last_date]
        if prev_dates:
            prev_date = max(prev_dates)
            prev_key = (prev_date, n, universe[0] if n > 0 else '')
            if prev_key in _cache:
                w_prev = _cache[prev_key].copy()
                if 'eligible' in history and isinstance(history['eligible'], pd.DataFrame):
                    el = history['eligible'].reindex(columns=universe).iloc[-1]
                    elig = el.fillna(False).astype(bool).to_numpy()
                else:
                    elig = np.ones(n, dtype=bool)
                last_px = df.iloc[-1].to_numpy(dtype=float)
                valid_now = elig & np.isfinite(last_px) & (last_px > 0.0)
                w_prev[~valid_now] = 0.0
                _cache[cache_key] = w_prev.copy()
                return w_prev

    if 'eligible' in history and isinstance(history['eligible'], pd.DataFrame):
        el = history['eligible'].reindex(columns=universe).iloc[-1]
        elig = el.fillna(False).astype(bool).to_numpy()
    else:
        elig = np.ones(n, dtype=bool)

    last = df.iloc[-1].to_numpy(dtype=float)
    valid = elig & np.isfinite(last) & (last > 0.0)
    if valid.sum() < 2:
        _cache[cache_key] = weights.copy()
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

    r1 = ret(1)
    r3 = ret(3)
    r5 = ret(5)

    def rank_pct(x):
        return pd.Series(x, index=universe, dtype=float).rank(pct=True, method='average')

    rev_score = -(0.5 * rank_pct(r5).fillna(0.5) +
                  0.3 * rank_pct(r3).fillna(0.5) +
                  0.2 * rank_pct(r1).fillna(0.5))

    if df.shape[0] > 21:
        rets = df.iloc[-21:].pct_change().iloc[1:]
        vol = rets.std()
        vol_rank = rank_pct(vol).fillna(0.5)
        rev_score = rev_score - 0.3 * vol_rank

    score = rev_score
    score_arr = score.to_numpy(dtype=float)
    cand = valid & np.isfinite(score_arr)
    idx = np.flatnonzero(cand)
    if idx.size < 2:
        _cache[cache_key] = weights.copy()
        return weights

    N = min(25, idx.size // 2)
    if N <= 0:
        _cache[cache_key] = weights.copy()
        return weights

    order = idx[np.argsort(score_arr[idx], kind='mergesort')]
    short_idx = order[:N]
    long_idx = order[-N:]
    w = min(0.02, 0.5 / N)

    weights.iloc[long_idx] = w
    weights.iloc[short_idx] = -w
    _cache[cache_key] = weights.copy()
    return weights