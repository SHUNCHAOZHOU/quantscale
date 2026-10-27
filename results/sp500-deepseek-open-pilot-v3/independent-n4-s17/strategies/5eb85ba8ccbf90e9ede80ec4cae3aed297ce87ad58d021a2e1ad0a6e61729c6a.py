import numpy as np
import pandas as pd


def _rank(arr):
    """Cross-sectional rank mapped to (-0.5, 0.5), average ties."""
    s = pd.Series(np.asarray(arr, dtype=float))
    r = s.rank(method='average').to_numpy(dtype=float)
    n = len(r)
    if n == 0:
        return r
    return r / n - 0.5


def generate_weights(history, universe):
    uni = pd.Index(universe)
    n = len(uni)
    if n == 0:
        return pd.Series(dtype=float)

    empty = pd.Series(0.0, index=uni, dtype=float)

    try:
        adj = history['adjusted_close']
    except Exception:
        return empty
    if adj is None:
        return empty

    try:
        adj = adj.reindex(columns=uni)
        px = adj.to_numpy(dtype=float)
    except Exception:
        return empty

    if px.ndim != 2 or px.shape[1] != n:
        return empty
    rows = px.shape[0]
    if rows < 64 or n < 10:
        return empty

    last = px[-1]

    # Multi-horizon momentum: 1, 3, 6, 12 months (whichever fit in history).
    lookbacks = [lb for lb in (21, 63, 126, 252) if lb < rows]
    if not lookbacks:
        return empty

    with np.errstate(invalid='ignore', divide='ignore'):
        rets = [last / px[-1 - lb] - 1.0 for lb in lookbacks]

    valid = np.isfinite(last) & (last > 0.0)
    for r in rets:
        valid = valid & np.isfinite(r)

    # Respect tradability flag when available.
    try:
        elig_raw = history['eligible'].reindex(columns=uni).iloc[-1]
        elig = pd.to_numeric(elig_raw, errors='coerce').fillna(1.0).to_numpy(dtype=float) > 0.0
        valid = valid & elig
    except Exception:
        pass

    idx = np.flatnonzero(valid)
    if len(idx) < 10:
        return empty

    L = len(rets)
    ranks = np.empty((L, len(idx)), dtype=float)
    for j in range(L):
        ranks[j] = _rank(rets[j][idx])
    score = np.nanmean(ranks, axis=0)

    order = np.argsort(-score, kind='mergesort')
    k = min(40, len(idx) // 2)
    if k < 5:
        return empty

    w = min(0.02, 0.5 / k)
    longs = idx[order[:k]]
    shorts = idx[order[-k:]]

    vals = np.zeros(n, dtype=float)
    vals[longs] += w
    vals[shorts] -= w

    out = pd.Series(vals, index=uni, dtype=float)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out
