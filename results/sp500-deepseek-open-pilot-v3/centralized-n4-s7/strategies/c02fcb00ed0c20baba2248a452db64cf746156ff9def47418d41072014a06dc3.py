import numpy as np
import pandas as pd


def _capped_inv_vol(inv, target, cap):
    n = len(inv)
    if n == 0 or target <= 0:
        return np.zeros(n, dtype=float)
    target = min(float(target), cap * n)
    inv = np.asarray(inv, dtype=float)
    inv = np.where(np.isfinite(inv) & (inv > 0), inv, 0.0)
    if inv.sum() <= 0:
        w = np.full(n, target / n, dtype=float)
        return np.minimum(w, cap)
    w = np.zeros(n, dtype=float)
    active = np.ones(n, dtype=bool)
    remaining = float(target)
    for _ in range(n + 2):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        s = inv[idx].sum()
        if s <= 0:
            w[idx] += remaining / idx.size
            remaining = 0.0
            break
        lam = remaining / s
        raw = lam * inv[idx]
        over = raw >= cap - 1e-15
        if not np.any(over):
            w[idx] = raw
            remaining = 0.0
            break
        over_idx = idx[over]
        w[over_idx] = cap
        active[over_idx] = False
        remaining -= cap * over_idx.size
        if remaining <= 1e-15:
            remaining = 0.0
            break
    if remaining > 1e-12:
        idx = np.flatnonzero(active & (w < cap - 1e-15))
        if idx.size:
            room = cap - w[idx]
            add = min(remaining, room.sum())
            if room.sum() > 0:
                w[idx] += add * room / room.sum()
    return np.minimum(w, cap)


def generate_weights(history, universe):
    universe = pd.Index(universe)
    n_assets = len(universe)
    if n_assets == 0:
        return pd.Series(dtype=float)

    px = None
    if 'adjusted_close' in history:
        px = history['adjusted_close']
    elif 'close' in history:
        px = history['close']
    else:
        return pd.Series(0.0, index=universe)

    if not isinstance(px, pd.DataFrame):
        return pd.Series(0.0, index=universe)

    px = px.reindex(columns=universe)
    if px.shape[0] < 25:
        return pd.Series(0.0, index=universe)
    if not px.index.is_monotonic_increasing:
        px = px.sort_index()

    vals = px.to_numpy(dtype=float, copy=False)
    n_dates = vals.shape[0]

    # Rebalance monthly: use the most recent prior month-end observation.
    dates = px.index
    current = dates[-1]
    same_month = (dates.year == current.year) & (dates.month == current.month)
    first_current = int(np.argmax(same_month))
    ref_idx = first_current - 1 if first_current > 0 else n_dates - 1
    if ref_idx < 21:
        return pd.Series(0.0, index=universe)

    last = vals[ref_idx]
    valid = np.isfinite(last) & (last > 0)

    if 'eligible' in history:
        e = history['eligible']
        if isinstance(e, pd.DataFrame):
            e = e.reindex(index=px.index, columns=universe)
            last_e = e.iloc[ref_idx].to_numpy(dtype=float, copy=False)
            valid &= np.isfinite(last_e) & (last_e > 0.5)

    rets = px.pct_change(fill_method=None).to_numpy(dtype=float, copy=False)
    start = max(0, ref_idx - 60)
    rwin = rets[start:ref_idx + 1]
    vol = np.nanstd(rwin, axis=0, ddof=1)
    obs = np.sum(np.isfinite(rwin), axis=0)
    valid &= np.isfinite(vol) & (vol > 1e-8) & (obs >= 20)

    if valid.sum() < 10:
        return pd.Series(0.0, index=universe)

    denom = np.maximum(vol, 1e-8)
    signal_sum = np.zeros(n_assets, dtype=float)
    signal_w = np.zeros(n_assets, dtype=float)

    # Reversal score: negative of market-relative past return, z-scored.
    for L, wt in ((20, 0.2), (60, 0.5), (120, 0.3)):
        if ref_idx <= L:
            continue
        past = vals[ref_idx - L]
        past_ok = np.isfinite(past) & (past > 0)
        ret = np.where(past_ok & valid, last / past - 1.0, np.nan)
        finite = np.isfinite(ret)
        if finite.sum() < 10:
            continue
        mu = ret[finite].mean()
        rel = ret - mu
        sd = rel[finite].std(ddof=0)
        if not np.isfinite(sd) or sd <= 1e-12:
            continue
        z = np.where(finite, rel / sd, np.nan)
        score = -np.clip(z, -3.0, 3.0)
        good = np.isfinite(score)
        signal_sum[good] += wt * score[good]
        signal_w[good] += wt

    signal = np.full(n_assets, np.nan, dtype=float)
    has = signal_w > 0
    signal[has] = signal_sum[has] / signal_w[has]
    valid &= np.isfinite(signal)

    candidates = np.flatnonzero(valid)
    if candidates.size < 10:
        return pd.Series(0.0, index=universe)

    K = min(40, candidates.size // 2)
    if K < 5:
        return pd.Series(0.0, index=universe)

    order = candidates[np.argsort(signal[candidates])]
    short_idx = order[:K]
    long_idx = order[-K:]

    cap = 0.02
    side_target = min(0.5, cap * K)
    inv = 1.0 / denom
    w = np.zeros(n_assets, dtype=float)
    w[long_idx] = _capped_inv_vol(inv[long_idx], side_target, cap)
    w[short_idx] = -_capped_inv_vol(inv[short_idx], side_target, cap)

    pos = w > 0
    neg = w < 0
    long_sum = w[pos].sum()
    short_sum = -w[neg].sum()
    if long_sum > 0 and short_sum > 0:
        side = min(long_sum, short_sum)
        w[pos] *= side / long_sum
        w[neg] *= side / short_sum
    else:
        return pd.Series(0.0, index=universe)

    w = np.clip(w, -cap, cap)
    gross = np.abs(w).sum()
    if gross > 1.0:
        w *= 1.0 / gross

    return pd.Series(w, index=universe, dtype=float)
