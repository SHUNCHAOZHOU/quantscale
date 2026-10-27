import numpy as np
import pandas as pd


def _zscore(x, mask):
    xm = np.where(mask, x, np.nan)
    if not np.any(mask):
        return np.zeros_like(x)
    with np.errstate(all='ignore'):
        mu = np.nanmean(xm)
        sd = np.nanstd(xm)
    if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 1e-12:
        return np.zeros_like(x)
    z = (x - mu) / sd
    z[~mask] = np.nan
    return z


def generate_weights(history, universe):
    universe = pd.Index(universe)
    weights = pd.Series(0.0, index=universe)
    if len(universe) == 0:
        return weights

    adj_df = history.get('adjusted_close')
    if adj_df is None or adj_df.shape[0] < 30:
        return weights

    cols = [c for c in universe if c in adj_df.columns]
    if len(cols) < 20:
        return weights

    A = adj_df[cols].to_numpy(dtype=float)
    T = A.shape[0]
    end = T - 1

    # Eligibility flag on the decision close
    elig_mask = np.ones(len(cols), dtype=bool)
    if 'eligible' in history:
        edf = history['eligible']
        if len([c for c in cols if c in edf.columns]) == len(cols):
            e_last = edf[cols].to_numpy()[-1]
            elig_mask = np.where(pd.isna(e_last), True, e_last.astype(bool))

    last_px = A[end]
    price_ok = np.isfinite(last_px) & (last_px > 1.0)

    # Momentum: 12-1 month, with adaptive fallback for short histories
    if T > 260:
        look, skip = 252, 21
    elif T > 140:
        look, skip = 126, 21
    else:
        look, skip = max(20, T - 10), 5
    s_idx = end - look
    k_idx = end - skip
    mom = np.full(len(cols), np.nan)
    if s_idx >= 0 and k_idx > s_idx:
        with np.errstate(all='ignore'):
            mom = A[k_idx] / A[s_idx] - 1.0
    mom_ok = np.isfinite(mom)

    # Short-horizon reversal over last 10 sessions
    r_idx = end - 10
    st_rev = np.full(len(cols), np.nan)
    if r_idx >= 0:
        with np.errstate(all='ignore'):
            st_rev = -(A[end] / A[r_idx] - 1.0)
    rev_ok = np.isfinite(st_rev)

    # Realized volatility (60d)
    with np.errstate(all='ignore'):
        R = A[1:] / A[:-1] - 1.0
    vol = np.full(len(cols), np.nan)
    wv = min(60, R.shape[0])
    if wv > 5:
        vol = np.nanstd(R[-wv:], axis=0)
    vol_ok = np.isfinite(vol) & (vol > 1e-6)

    # Liquidity floor: 20d median dollar volume > $1M
    liq_ok = np.ones(len(cols), dtype=bool)
    if 'volume' in history:
        vdf = history['volume']
        if len([c for c in cols if c in vdf.columns]) == len(cols):
            V = vdf[cols].to_numpy(dtype=float)
            ww = min(20, V.shape[0])
            with np.errstate(all='ignore'):
                dv = np.nanmedian(V[-ww:] * A[-ww:], axis=0)
            liq_ok = np.isfinite(dv) & (dv > 1e6)

    base_mask = price_ok & mom_ok & rev_ok & vol_ok & elig_mask & liq_ok

    # Volatility ceiling: drop the top-decile vol names
    if base_mask.sum() > 10:
        with np.errstate(all='ignore'):
            vp = np.nanpercentile(vol[base_mask], 90)
        base_mask = base_mask & (vol <= vp)

    if base_mask.sum() < 10:
        return weights

    z_mom = _zscore(mom, base_mask)
    z_rev = _zscore(st_rev, base_mask)
    with np.errstate(all='ignore'):
        score = 0.5 * z_mom + 0.5 * z_rev
    score_ok = base_mask & np.isfinite(score)

    idx = np.where(score_ok)[0]
    if len(idx) < 10:
        return weights

    order = idx[np.argsort(-score[idx], kind='stable')]
    n_side = min(40, len(order) // 2)
    if n_side < 5:
        return weights

    long_sel = order[:n_side]
    short_sel = order[-n_side:]
    wt = min(0.02, 0.5 / n_side)

    for i in long_sel:
        weights[cols[i]] = wt
    for i in short_sel:
        weights[cols[i]] = -wt

    return weights