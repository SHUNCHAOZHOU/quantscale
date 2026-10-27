import numpy as np
import pandas as pd


def _rank_score(s):
    r = s.rank(pct=True)
    return (r - 0.5).fillna(0.0)


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)

    px = history.get('adjusted_close')
    if px is None:
        px = history.get('close')
    if px is None:
        return zeros

    px = px.reindex(columns=universe)
    if px.shape[0] < 30:
        return zeros

    vol = history.get('volume')
    if vol is not None:
        vol = vol.reindex(columns=universe).reindex(index=px.index)

    last = px.iloc[-1]
    valid = last.notna() & (last > 0)
    n_hist = px.notna().sum()
    valid = valid & (n_hist >= 130)

    mom6 = px.shift(5) / px.shift(126) - 1.0
    valid = valid & mom6.iloc[-1].notna()

    el = history.get('eligible')
    if el is not None:
        try:
            el = el.reindex(columns=universe).reindex(index=px.index)
            el_last = el.ffill().iloc[-1]
            if el_last.notna().sum() > 0:
                valid = valid & el_last.fillna(False).astype(bool)
        except Exception:
            pass

    if vol is not None:
        try:
            dollar = (px * vol).tail(20).median()
            valid = valid & (dollar > 5e5)
        except Exception:
            pass

    if valid.sum() < 4:
        return zeros

    ret = px / px.shift(1) - 1.0
    mom12 = px.shift(21) / px.shift(252) - 1.0
    rev5 = px / px.shift(5) - 1.0
    vol60 = ret.tail(60).std()
    dhigh = px.iloc[-1] / px.tail(252).max() - 1.0

    s_mom6 = _rank_score(mom6.iloc[-1])
    s_mom12 = _rank_score(mom12.iloc[-1])
    s_rev = _rank_score(rev5.iloc[-1])
    s_vol = _rank_score(vol60.iloc[-1])
    s_dh = _rank_score(dhigh)

    score = (0.70 * s_mom6 + 0.50 * s_mom12 - 0.50 * s_rev
             - 0.90 * s_vol + 0.25 * s_dh)
    score = score[valid].replace([np.inf, -np.inf], np.nan).dropna()

    n = len(score)
    if n < 4:
        return zeros

    n_side = min(50, n // 2)
    if n_side < 2:
        return zeros

    order = score.sort_values(kind='mergesort')
    short_names = order.index[:n_side]
    long_names = order.index[-n_side:]

    w = 0.5 / n_side
    if w > 0.02:
        w = 0.02

    weights = zeros.copy()
    weights.loc[long_names] = w
    weights.loc[short_names] = -w

    gross = weights.abs().sum()
    if gross > 1.0 + 1e-12:
        weights = weights * (1.0 / gross)

    return weights.astype(float)
