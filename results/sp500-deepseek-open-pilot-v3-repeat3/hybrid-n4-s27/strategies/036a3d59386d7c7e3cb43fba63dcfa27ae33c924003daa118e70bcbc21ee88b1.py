import numpy as np
import pandas as pd


def generate_weights(history, universe):
    universe = pd.Index(list(universe))
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    close = history.get('adjusted_close')
    if close is None or not isinstance(close, pd.DataFrame):
        close = history.get('close')
    vol = history.get('volume')
    if close is None or vol is None or not isinstance(close, pd.DataFrame) or not isinstance(vol, pd.DataFrame):
        return zero

    px = close.reindex(columns=universe).astype(float)
    vl = vol.reindex(columns=universe).astype(float)
    if px.shape[0] < 30:
        return zero

    last_pos = px.shape[0] - 1
    base_pos = last_pos - (last_pos % 21)
    if base_pos < 20:
        base_pos = last_pos

    start = max(0, base_pos - 200)
    sub = px.iloc[start:base_pos + 1]
    vsub = vl.iloc[start:base_pos + 1]
    if len(sub) < 22:
        return zero

    last_px = sub.iloc[-1]
    dollar = sub * vsub
    liq = dollar.tail(60).median(axis=0, skipna=True)
    liq_cnt = dollar.tail(60).count(axis=0)
    liq = liq.where(liq_cnt >= 10, np.nan)

    r5 = sub.iloc[-1] / sub.iloc[-6] - 1.0
    trend_lb = min(126, len(sub) - 1)
    if trend_lb < 20:
        return zero
    rT = sub.iloc[-1] / sub.iloc[-1 - trend_lb] - 1.0
    r5 = r5.replace([np.inf, -np.inf], np.nan)
    rT = rT.replace([np.inf, -np.inf], np.nan)

    elig_series = pd.Series(True, index=universe, dtype=bool)
    elig_df = history.get('eligible')
    if isinstance(elig_df, pd.DataFrame):
        if elig_df.shape[0] > base_pos:
            e = elig_df.reindex(columns=universe).iloc[base_pos]
            elig_series = e.fillna(False).astype(bool)

    valid = (
        last_px.notna() & np.isfinite(last_px) & (last_px > 0.0) &
        r5.notna() & np.isfinite(r5) &
        rT.notna() & np.isfinite(rT) &
        liq.notna() & np.isfinite(liq) & (liq > 0.0) &
        elig_series
    )
    cand = universe[valid]
    if len(cand) < 20:
        return zero

    liq_c = liq.reindex(cand).dropna()
    if len(liq_c) < 20:
        return zero
    q40 = liq_c.quantile(0.4)
    cand = liq_c[liq_c >= q40].index
    if len(cand) < 20:
        return zero

    r5_c = r5.reindex(cand)
    rT_c = rT.reindex(cand)
    r5_rank = r5_c.rank(method='average', pct=True)
    rT_rank = rT_c.rank(method='average', pct=True)

    score = rT_rank - r5_rank
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 20:
        return zero

    n_side = min(40, len(score) // 2)
    if n_side < 5:
        return zero

    ordered = score.sort_values(ascending=False, kind='mergesort')
    longs = ordered.index[:n_side]
    shorts = ordered.index[-n_side:]

    long_gross = 0.5025
    short_gross = 0.4975
    wl = min(0.02, long_gross / float(n_side))
    ws = min(0.02, short_gross / float(n_side))

    weights = zero.copy()
    weights.loc[longs] = wl
    weights.loc[shorts] = -ws
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    weights = weights.clip(-0.02, 0.02)

    gross = float(weights.abs().sum())
    if gross > 1.0:
        weights = weights * (1.0 / gross)

    net = float(weights.sum())
    if abs(net) > 0.01:
        if net > 0.0:
            adj = min(net / max(len(longs), 1), 0.001)
            weights.loc[longs] = weights.loc[longs] - adj
        else:
            adj = min((-net) / max(len(shorts), 1), 0.001)
            weights.loc[shorts] = weights.loc[shorts] + adj
        weights = weights.clip(-0.02, 0.02)
        gross2 = float(weights.abs().sum())
        if gross2 > 1.0:
            weights = weights / gross2

    return weights.astype(float)