import numpy as np
import pandas as pd


def _field(history, name):
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

    adj = _field(history, 'adjusted_close')
    if adj is None:
        adj = _field(history, 'close')
    if adj is None:
        return zero

    cols = [c for c in universe if c in adj.columns]
    if len(cols) < 4:
        return zero

    adj = adj.loc[:, cols].astype(float)
    n = len(adj)
    if n < 90:
        return zero

    rebal = 63
    apos = ((n - 1) // rebal) * rebal
    if apos < 22:
        return zero

    px_raw = adj.iloc[:apos + 1]
    px = px_raw.ffill()
    if len(px) < 22:
        return zero

    anchor_date = px.index[-1]
    last_raw = px_raw.iloc[-1]
    last = px.iloc[-1]
    valid = last_raw.notna() & np.isfinite(last_raw) & (last_raw > 0.0) & last.notna() & np.isfinite(last) & (last > 0.0)

    elig = _field(history, 'eligible')
    if elig is not None:
        epos = int(elig.index.searchsorted(anchor_date, side='right'))
        if epos > 0:
            ev = elig.iloc[epos - 1].reindex(cols).fillna(False).astype(bool)
            valid = valid & ev

    vol = _field(history, 'volume')
    close = _field(history, 'close')
    if vol is not None and close is not None:
        vv = vol.reindex(index=px.index, columns=cols)
        cp = close.reindex(index=px.index, columns=cols)
        dollar = vv * cp
        med = dollar.tail(63).median(axis=0, skipna=True)
        liq = med.fillna(0.0) >= 1000000.0
        valid = valid & liq.reindex(cols).fillna(False)

    ret = px.iloc[-1] / px.iloc[-22] - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)
    score = -ret
    valid = valid & ret.notna() & score.notna() & np.isfinite(score)

    if len(px) >= 100:
        wlen = min(200, len(px))
        ma = px.tail(wlen).mean(axis=0, skipna=True)
        trend = px.iloc[-1] / ma - 1.0
        trend = trend.replace([np.inf, -np.inf], np.nan)
        breadth = trend[valid].median()
        if pd.notna(breadth) and breadth <= 0.0:
            return zero

    cand = score[valid].dropna()
    if len(cand) < 1:
        return zero

    order = cand.sort_values(ascending=False, kind='mergesort')
    n_side = min(50, len(order))
    if n_side < 1:
        return zero

    w = min(0.02, 1.0 / float(n_side))
    weights = zero.copy()
    weights.loc[order.index[:n_side]] = w
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.02, 0.02)

    gross = float(weights.abs().sum())
    if gross > 1.0 and gross > 0.0:
        weights = weights * (1.0 / gross)

    return weights.astype(float)