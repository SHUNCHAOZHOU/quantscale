import numpy as np
import pandas as pd


def _price_frame(history):
    px = history.get('adjusted_close')
    if px is None or not isinstance(px, pd.DataFrame) or px.empty:
        px = history.get('close')
    if px is None or not isinstance(px, pd.DataFrame) or px.empty:
        return None
    return px


def generate_weights(history, universe):
    u = pd.Index(universe)
    w = pd.Series(0.0, index=u, dtype=float)
    if len(u) == 0:
        return w

    px = _price_frame(history)
    if px is None:
        return w

    cols = [t for t in u if t in px.columns]
    if len(cols) < 8:
        return w

    # Only a short window is required for 5d/20d signals.
    sub = px.loc[:, cols].astype(float).tail(70)
    if len(sub) < 22:
        return w

    cols2 = list(sub.columns)
    last = sub.iloc[-1]

    def ret_n(n):
        if len(sub) > n:
            return sub.iloc[-1] / sub.iloc[-1 - n] - 1.0
        return pd.Series(np.nan, index=cols2)

    ret5 = ret_n(5)
    ret20 = ret_n(20)

    # Eligibility mask (point-in-time).
    elig_last = None
    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame) and not elig.empty:
        ecols = [t for t in cols2 if t in elig.columns]
        if ecols:
            e = elig.loc[:, ecols].tail(1)
            elig_last = e.iloc[-1].reindex(cols2).fillna(True).astype(bool)

    valid = last.notna() & (last > 0) & np.isfinite(ret5) & np.isfinite(ret20)
    if elig_last is not None:
        valid = valid & elig_last

    cand = [t for t in cols2 if bool(valid.get(t, False))]
    if len(cand) < 8:
        return w

    r5 = ret5.reindex(cand).astype(float)
    r20 = ret20.reindex(cand).astype(float)

    # Momentum score: higher recent relative strength -> long.
    s5 = r5.rank(pct=True)
    s20 = r20.rank(pct=True)
    score = 0.5 * s5 + 0.5 * s20
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 8:
        return w

    n_valid = len(score)
    n_side = min(25, n_valid // 2)
    if n_side < 1:
        return w

    # Equal weight per name, capped at 2%; fixed gross target of up to 1.0.
    w_per = min(0.02, 1.0 / (2.0 * n_side))

    ranked = score.sort_values(kind='mergesort')
    longs = list(ranked.index[-n_side:])   # strongest relative strength
    shorts = list(ranked.index[:n_side])   # weakest relative strength

    for t in longs:
        w.loc[t] = w_per
    for t in shorts:
        w.loc[t] = -w_per

    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    w = w.clip(-0.02, 0.02)

    gross_actual = w.abs().sum()
    if gross_actual > 1.0 + 1e-12:
        w = w * (1.0 / gross_actual)

    return w
