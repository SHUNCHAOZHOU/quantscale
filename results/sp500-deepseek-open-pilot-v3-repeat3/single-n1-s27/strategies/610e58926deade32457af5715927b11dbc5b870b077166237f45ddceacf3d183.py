import numpy as np
import pandas as pd


def _rank_centered(s):
    """Cross-sectional percentile rank mapped to [-0.5, 0.5]; NaN stays NaN."""
    s = pd.Series(s, dtype=float)
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.rank(pct=True) - 0.5


def _mom(px, skip, look):
    """Series of (price[t-skip] / price[t-skip-look] - 1) using positional rows."""
    n = px.shape[0]
    skip = int(min(skip, max(0, n - 2)))
    look = int(min(look, max(1, n - 1 - skip)))
    num = px.iloc[n - 1 - skip]
    den = px.iloc[n - 1 - skip - look]
    return num / den - 1.0


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)

    px = history.get('adjusted_close')
    if px is None or px.shape[1] == 0:
        px = history.get('close')
    if px is None or px.shape[1] == 0:
        return zeros

    try:
        px = px.reindex(columns=universe).astype(float)
    except Exception:
        return zeros

    n = px.shape[0]
    if n < 130:
        return zeros

    last = px.iloc[-1]
    valid = last.notna() & (last > 0.0) & (px.notna().sum() >= 120)

    # --- signals (all cross-sectional Series over tickers) ---
    mom6 = _mom(px, 5, 126)
    mom12 = _mom(px, 21, 252)
    rev5 = _mom(px, 0, 5)

    ret = px.pct_change()
    vol = ret.tail(60).std()

    hi_win = px.tail(min(252, n))
    dhigh = last / hi_win.max() - 1.0

    valid = valid & mom6.notna() & np.isfinite(mom6.fillna(np.inf))

    # liquidity screen
    volm = history.get('volume')
    if volm is not None:
        try:
            volm = volm.reindex(columns=universe).reindex(index=px.index).astype(float)
            dollar = (px * volm).tail(20).median()
            dollar = dollar.reindex(universe)
            liq = dollar.fillna(0.0) > 5e5
            if liq.sum() >= 8:
                valid = valid & liq
        except Exception:
            pass

    # eligibility screen (last known flag)
    el = history.get('eligible')
    if el is not None:
        try:
            el = el.reindex(columns=universe).reindex(index=px.index)
            el_last = el.ffill().iloc[-1]
            if el_last.notna().sum() > 0:
                flag = el_last.fillna(1.0).astype(float) > 0.5
                if flag.sum() >= 8:
                    valid = valid & flag
        except Exception:
            pass

    if int(valid.sum()) < 6:
        return zeros

    z_mom6 = _rank_centered(mom6)
    z_mom12 = _rank_centered(mom12)
    z_rev = _rank_centered(rev5)
    z_vol = _rank_centered(vol)
    z_dh = _rank_centered(dhigh)

    score = (0.70 * z_mom6.fillna(0.0)
             + 0.50 * z_mom12.fillna(0.0)
             - 0.50 * z_rev.fillna(0.0)
             - 0.90 * z_vol.fillna(0.0)
             + 0.25 * z_dh.fillna(0.0))

    score = score[valid].replace([np.inf, -np.inf], np.nan).dropna()
    k = int(len(score))
    if k < 6:
        return zeros

    n_side = min(50, k // 2)
    if n_side < 2:
        return zeros

    order = score.sort_values(kind='mergesort')
    shorts = order.index[:n_side]
    longs = order.index[-n_side:]

    w = min(0.5 / n_side, 0.02)

    weights = zeros.copy()
    weights.loc[longs] = w
    weights.loc[shorts] = -w

    gross = float(weights.abs().sum())
    if gross > 1.0 + 1e-12:
        weights = weights * (1.0 / gross)

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights.astype(float)
