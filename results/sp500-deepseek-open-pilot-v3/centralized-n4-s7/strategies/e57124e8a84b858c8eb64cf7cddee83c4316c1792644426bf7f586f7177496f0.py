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
    if px.shape[0] == 0:
        return pd.Series(0.0, index=universe)
    if not px.index.is_monotonic_increasing:
        px = px.sort_index()
    vals = px.to_numpy(dtype=float, copy=False)
    n_dates = vals.shape[0]
    if n_dates < 6:
        return pd.Series(0.0, index=universe)
    last = vals[-1]
    valid = np.isfinite(last) & (last > 0)
    if 'eligible' in history:
        e = history['eligible']
        if isinstance(e, pd.DataFrame):
            e = e.reindex(index=px.index, columns=universe)
            last_e = e.iloc[-1].to_numpy(dtype=float, copy=False)
            valid &= np.isfinite(last_e) & (last_e > 0.5)
    vol_win = min(60, n_dates - 1)
    if vol_win < 20:
        return pd.Series(0.0, index=universe)
    rets = px.pct_change(fill_method=None).to_numpy(dtype=float, copy=False)
    rwin = rets[-vol_win:]
    vol = np.nanstd(rwin, axis=0, ddof=1)
    obs = np.sum(np.isfinite(rwin), axis=0)
    valid &= np.isfinite(vol) & (vol > 1e-8) & (obs >= 20)
    denom = np.maximum(vol, 1e-8)
    sig_sum = np.zeros(n_assets, dtype=float)
    sig_cnt = np.zeros(n_assets, dtype=float)
    for L in (20, 60, 120):
        if n_dates > L:
            past = vals[-1 - L]
            retL = np.where(np.isfinite(past) & (past > 0), last / past - 1.0, np.nan)
            z = retL / (denom * np.sqrt(float(L)))
            z = np.where(np.isfinite(z), np.clip(z, -5.0, 5.0), np.nan)
            good = np.isfinite(z)
            sig_sum[good] += z[good]
            sig_cnt[good] += 1.0
    signal = np.full(n_assets, np.nan, dtype=float)
    has = sig_cnt > 0
    signal[has] = sig_sum[has] / sig_cnt[has]
    if n_dates > 5:
        past5 = vals[-6]
        ret5 = np.where(np.isfinite(past5) & (past5 > 0), last / past5 - 1.0, np.nan)
        z5 = ret5 / (denom * np.sqrt(5.0))
        z5 = np.where(np.isfinite(z5), np.clip(z5, -5.0, 5.0), np.nan)
        overlay = np.where(np.isfinite(z5), z5, 0.0)
        signal = signal - 0.25 * overlay
    valid &= np.isfinite(signal)
    candidates = np.flatnonzero(valid)
    if candidates.size < 2:
        return pd.Series(0.0, index=universe)
    order = candidates[np.argsort(signal[candidates])]
    K = min(40, order.size // 2)
    if K <= 0:
        return pd.Series(0.0, index=universe)
    short_idx = order[:K]
    long_idx = order[-K:]
    cap = 0.02
    side_target = min(0.5, cap * K)
    inv = 1.0 / np.maximum(vol, 1e-8)
    w = np.zeros(n_assets, dtype=float)
    w[long_idx] = _capped_inv_vol(inv[long_idx], side_target, cap)
    w[short_idx] = -_capped_inv_vol(inv[short_idx], side_target, cap)
    pos = w > 0
    neg = w < 0
    long_sum = w[pos].sum()
    short_sum = -w[neg].sum()
    if long_sum > 0 and short_sum > 0:
        if long_sum > short_sum:
            w[pos] *= short_sum / long_sum
        elif short_sum > long_sum:
            w[neg] *= long_sum / short_sum
    w = np.clip(w, -cap, cap)
    gross = np.abs(w).sum()
    if gross > 1.0:
        w *= 1.0 / gross
    return pd.Series(w, index=universe, dtype=float)
