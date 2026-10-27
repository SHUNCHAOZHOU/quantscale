import numpy as np
import pandas as pd


def _safe_field(history, name):
    try:
        df = history.get(name)
    except Exception:
        return None
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return None


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    adj = _safe_field(history, "adjusted_close")
    if adj is None:
        adj = _safe_field(history, "close")
    if adj is None:
        return zero

    cols = [c for c in universe if c in adj.columns]
    if len(cols) < 4:
        return zero

    idx = adj.index
    n = len(idx)
    if n < 80:
        return zero

    rebalance = 21
    min_obs = 80
    apos = ((n - 1) // rebalance) * rebalance
    if apos < min_obs:
        return zero

    start = max(0, apos - 259)
    px = adj.iloc[start:apos + 1].reindex(columns=cols).ffill()
    if len(px) < min_obs:
        return zero

    anchor_date = px.index[-1]
    last = px.iloc[-1]
    valid = last.notna() & (last > 0.0) & np.isfinite(last)

    elig = _safe_field(history, "eligible")
    if elig is not None:
        epos = int(elig.index.searchsorted(anchor_date, side="right"))
        if epos > 0:
            ev = elig.iloc[epos - 1].reindex(cols).fillna(True).astype(bool)
            valid = valid & ev

    volume = _safe_field(history, "volume")
    close = _safe_field(history, "close")
    if volume is not None:
        vv = volume.reindex(index=px.index, columns=cols)
        cp = close.reindex(index=px.index, columns=cols) if close is not None else px
        dollar = vv * cp
        window = min(63, len(dollar))
        med = dollar.tail(window).median(axis=0, skipna=True)
        liq_ok = pd.Series(True, index=cols, dtype=bool)
        for c in cols:
            if c in volume.columns:
                mval = med.get(c, np.nan)
                liq_ok.loc[c] = bool(np.isfinite(mval) and mval >= 1000000.0)
            else:
                liq_ok.loc[c] = True
        valid = valid & liq_ok

    ret = px / px.shift(1) - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)
    vol = ret.tail(63).std(axis=0, skipna=True)
    if len(px) < 22:
        return zero
    rev = -(px.iloc[-1] / px.iloc[-22] - 1.0)
    rev = rev.replace([np.inf, -np.inf], np.nan)
    vol = vol.replace([np.inf, -np.inf], np.nan)

    valid = valid & rev.notna() & vol.notna() & (vol > 0.0) & np.isfinite(rev) & np.isfinite(vol)
    cand = [c for c in cols if bool(valid.get(c, False))]
    if len(cand) < 4:
        return zero

    sr = rev.loc[cand]
    sv = vol.loc[cand]
    rev_rank = sr.rank(method="average", pct=True)
    lowvol_rank = (-sv).rank(method="average", pct=True)
    score = 0.55 * rev_rank + 0.45 * lowvol_rank
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 4:
        return zero

    order = score.sort_values(ascending=False)
    n_side = int(min(50, len(order) // 2))
    if n_side < 1:
        return zero

    w = min(0.02, 0.5 / float(n_side))
    weights = zero.copy()
    longs = list(order.index[:n_side])
    shorts = list(order.index[-n_side:])

    for t in longs:
        weights.loc[t] = weights.loc[t] + w
    for t in shorts:
        weights.loc[t] = weights.loc[t] - w

    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.02, 0.02)

    gross = float(weights.abs().sum())
    if gross > 1.0 and gross > 0.0:
        weights = weights * (1.0 / gross)

    net = float(weights.sum())
    if abs(net) > 0.01:
        weights = weights - net / float(len(universe))
        weights = weights.clip(-0.02, 0.02)
        gross = float(weights.abs().sum())
        if gross > 1.0 and gross > 0.0:
            weights = weights * (1.0 / gross)

    return weights.astype(float)