import numpy as np
import pandas as pd


def _rank_pct(values, valid):
    out = np.full(len(values), np.nan, dtype=float)
    idx = np.where(valid & np.isfinite(values))[0]
    if idx.size == 0:
        return out
    ranks = pd.Series(values[idx]).rank(method='average').to_numpy(dtype=float)
    out[idx] = (ranks - 0.5) / float(idx.size)
    return out


def _cap_normalize(raw, target, cap):
    raw = np.asarray(raw, dtype=float)
    n = raw.size
    if n == 0:
        return raw
    raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
    if raw.sum() <= 0.0:
        return np.full(n, target / float(n), dtype=float)
    w = raw / raw.sum() * target
    for _ in range(50):
        over = w > cap + 1e-15
        if not np.any(over):
            break
        excess = np.sum(w[over] - cap)
        w[over] = cap
        under = ~over
        if not np.any(under):
            break
        r = raw[under]
        if r.sum() <= 0.0:
            w[under] += excess / float(under.sum())
        else:
            w[under] += excess * r / r.sum()
    for _ in range(50):
        diff = target - w.sum()
        if abs(diff) < 1e-12:
            break
        if diff > 0.0:
            room = cap - w
            if room.sum() <= 0.0:
                break
            w += diff * room / room.sum()
        else:
            pos = w > 0.0
            if not np.any(pos):
                break
            w[pos] += diff * w[pos] / w[pos].sum()
    return np.minimum(w, cap)


def generate_weights(history, universe):
    universe = pd.Index(universe)
    n_univ = len(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if n_univ == 0:
        return zero

    try:
        ac_all = history['adjusted_close']
    except Exception:
        return zero
    if not isinstance(ac_all, pd.DataFrame):
        return zero

    cols = [c for c in universe if c in ac_all.columns]
    if len(cols) == 0:
        return zero
    ac = ac_all.loc[:, cols]
    n_rows = ac.shape[0]
    if n_rows < 80:
        return zero

    # Deterministic monthly refresh: use the most recent 21-session block
    # boundary, and hold that signal until the next boundary.
    anchor = ((n_rows - 1) // 21) * 21
    if anchor < 63:
        return zero

    prices = ac.iloc[:anchor + 1].to_numpy(dtype=float, copy=True)
    n_cols = len(cols)
    last_p = prices[-1]

    elig_df = history.get('eligible')
    if isinstance(elig_df, pd.DataFrame):
        e = elig_df.reindex(index=ac.index, columns=cols)
        last_e = e.iloc[anchor].fillna(False).astype(bool).to_numpy()
    else:
        last_e = np.ones(n_cols, dtype=bool)

    valid = np.isfinite(last_p) & (last_p > 0.0) & last_e

    vol_df = history.get('volume')
    close_df = history.get('close')
    if isinstance(vol_df, pd.DataFrame):
        v_full = vol_df.reindex(index=ac.index, columns=cols).to_numpy(dtype=float, copy=True)
        v = v_full[:anchor + 1]
        if isinstance(close_df, pd.DataFrame):
            px_full = close_df.reindex(index=ac.index, columns=cols).to_numpy(dtype=float, copy=True)
            px = px_full[:anchor + 1]
        else:
            px = prices
        w = min(60, anchor + 1)
        dollar = v[-w:] * px[-w:]
        med_dv = np.nanmedian(dollar, axis=0)
        valid &= np.isfinite(med_dv) & (med_dv >= 1000000.0)

    rets = np.full_like(prices, np.nan, dtype=float)
    rets[1:] = prices[1:] / prices[:-1] - 1.0
    rw = min(60, anchor)
    if rw >= 2:
        vola = np.nanstd(rets[-rw:], axis=0)
    else:
        vola = np.full(n_cols, np.nan)
    valid &= np.isfinite(vola) & (vola > 0.0)

    # One-month reversal: prior-month losers are expected winners.
    if anchor >= 21:
        p_end = prices[-1]
        p_start = prices[-22]
        rev = -(p_end / p_start - 1.0)
        rev[~np.isfinite(rev)] = np.nan
    else:
        rev = np.full(n_cols, np.nan)

    f_rev = _rank_pct(rev, valid)
    f_vol = _rank_pct(-vola, valid)

    signal = np.full(n_cols, np.nan, dtype=float)
    ssum = np.zeros(n_cols, dtype=float)
    wsum = np.zeros(n_cols, dtype=float)
    for vals, wgt in ((f_rev, 0.65), (f_vol, 0.35)):
        has = np.isfinite(vals)
        ssum[has] += wgt * vals[has]
        wsum[has] += wgt
    good = wsum > 0.0
    signal[good] = ssum[good] / wsum[good]

    cand = np.where(valid & np.isfinite(signal))[0]
    if cand.size < 10:
        return zero
    n_side = int(min(50, cand.size // 2))
    if n_side < 5:
        return zero

    order = cand[np.argsort(signal[cand], kind='mergesort')]
    shorts = order[:n_side]
    longs = order[-n_side:]

    target_side = min(0.5, 0.02 * n_side)
    raw_long = 1.0 / np.maximum(vola[longs], 1e-8)
    raw_short = 1.0 / np.maximum(vola[shorts], 1e-8)
    w_long = _cap_normalize(raw_long, target_side, 0.02)
    w_short = _cap_normalize(raw_short, target_side, 0.02)

    weights = np.zeros(n_cols, dtype=float)
    weights[longs] = w_long
    weights[shorts] = -w_short

    out = zero.copy()
    for i, ticker in enumerate(cols):
        val = weights[i]
        if val != 0.0:
            out.loc[ticker] = float(val)
    return out.astype(float)
