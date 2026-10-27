import numpy as np
import pandas as pd


def _cs_rank(arr):
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

    weights = pd.Series(0.0, index=uni, dtype=float)

    # Pull adjusted close prices; bail out defensively on any data problem.
    try:
        adj = history['adjusted_close']
    except Exception:
        return weights
    if adj is None:
        return weights
    try:
        if adj.shape[0] < 65 or n < 6:
            return weights
        adj = adj.reindex(columns=uni)
    except Exception:
        return weights

    try:
        px = adj.to_numpy(dtype=float)
    except Exception:
        return weights
    if px.ndim != 2 or px.shape[1] != n or px.shape[0] < 65:
        return weights

    last = px[-1]

    with np.errstate(invalid='ignore', divide='ignore'):
        r5 = last / px[-6] - 1.0
        r60 = last / px[-61] - 1.0
        recent = px[-21:]
        rets = recent[1:] / recent[:-1] - 1.0
        vol = np.nanstd(rets, axis=0)

    finite_rets = np.isfinite(rets)
    cnt = finite_rets.sum(axis=0)

    valid = (np.isfinite(last) & (last > 0.0)
             & np.isfinite(r5) & np.isfinite(r60)
             & np.isfinite(vol) & (vol > 0.0)
             & (cnt >= 15))

    # Respect the tradability flag if present.
    try:
        elig_raw = history['eligible'].reindex(columns=uni).iloc[-1]
        elig = pd.to_numeric(elig_raw, errors='coerce').fillna(1.0).to_numpy(dtype=float) > 0.0
        valid = valid & elig
    except Exception:
        pass

    idx = np.flatnonzero(valid)
    if len(idx) < 6:
        return weights

    r5v = r5[idx]
    r60v = r60[idx]
    volv = vol[idx]

    rank_r5 = _cs_rank(r5v)
    rank_vol = _cs_rank(volv)
    rank_mom = _cs_rank(r60v)

    # Short-horizon reversal (short recent winners), low-vol tilt, light momentum.
    score = -1.0 * rank_r5 - 0.75 * rank_vol + 0.40 * rank_mom

    order = np.argsort(-score, kind='mergesort')
    k = min(30, len(idx) // 2)
    if k < 3:
        return weights

    long_pos = order[:k]
    short_pos = order[-k:]

    target_gross_per_side = 0.45
    w = min(0.02, target_gross_per_side / k)

    long_tickers = idx[long_pos]
    short_tickers = idx[short_pos]

    vals = np.zeros(n, dtype=float)
    vals[long_tickers] = w
    vals[short_tickers] = -w

    weights = pd.Series(vals, index=uni, dtype=float)
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights
