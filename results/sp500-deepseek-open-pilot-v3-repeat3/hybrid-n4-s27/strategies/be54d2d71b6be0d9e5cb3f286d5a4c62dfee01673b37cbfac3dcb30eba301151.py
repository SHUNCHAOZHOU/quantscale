import numpy as np
import pandas as pd


def generate_weights(history, universe):
    universe = pd.Index(list(universe))
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    px = history.get('adjusted_close')
    if not isinstance(px, pd.DataFrame):
        px = history.get('close')
    if not isinstance(px, pd.DataFrame) or px.shape[0] < 70:
        return zero

    px = px.reindex(columns=universe).astype(float)
    if px.shape[1] == 0:
        return zero
    px = px.tail(160)
    if px.shape[0] < 70:
        return zero

    ret = px.pct_change(fill_method=None)
    ret = ret.replace([np.inf, -np.inf], np.nan)

    vol = ret.tail(60).std(ddof=0, skipna=True)
    cnt = ret.tail(60).count()
    last = px.iloc[-1]

    elig_df = history.get('eligible')
    if isinstance(elig_df, pd.DataFrame):
        elig = elig_df.reindex(index=px.index, columns=universe).ffill().iloc[-1]
        elig = elig.fillna(False).astype(bool)
    else:
        elig = pd.Series(True, index=universe, dtype=bool)

    valid = elig & last.notna() & np.isfinite(last) & (last > 0.0)
    valid = valid & vol.notna() & np.isfinite(vol) & (vol > 0.0) & (cnt >= 40)

    liq = None
    volume = history.get('volume')
    if isinstance(volume, pd.DataFrame):
        vol_df = volume.reindex(index=px.index, columns=universe).astype(float)
        dollar = px * vol_df
        liq = dollar.tail(20).median(skipna=True)
        liq_cnt = dollar.tail(20).count()
        liq = liq.where(liq_cnt >= 5)

    if liq is not None:
        valid = valid & liq.notna() & np.isfinite(liq) & (liq > 0.0)
        if valid.any():
            q20 = liq[valid].quantile(0.2)
            if np.isfinite(q20):
                valid = valid & (liq >= q20)

    if int(valid.sum()) < 10:
        return zero

    vol_valid = vol[valid].dropna()
    vol_valid = vol_valid.replace([np.inf, -np.inf], np.nan).dropna()
    if len(vol_valid) < 6:
        return zero

    ordered = vol_valid.sort_values(ascending=False, kind='mergesort')
    n_side = min(30, len(ordered) // 2)
    if n_side < 3:
        return zero

    high = ordered.iloc[:n_side].index
    low = ordered.iloc[-n_side:].index

    long_high = True
    look = min(63, len(ret) - 1)
    if look >= 20:
        r = ret.iloc[-look:]
        min_obs = max(10, look // 2)
        high_cum = ((1.0 + r[high]).prod(min_count=min_obs) - 1.0).mean(skipna=True)
        low_cum = ((1.0 + r[low]).prod(min_count=min_obs) - 1.0).mean(skipna=True)
        if np.isfinite(high_cum) and np.isfinite(low_cum):
            long_high = (high_cum - low_cum) >= 0.0

    if long_high:
        longs = high
        shorts = low
    else:
        longs = low
        shorts = high

    per_name = min(0.02, 0.4 / float(n_side))
    out = zero.copy()
    out.loc[longs] = per_name
    out.loc[shorts] = -per_name
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)

    gross = float(out.abs().sum())
    if gross > 1.0 + 1e-12:
        out = out * (1.0 / gross)
    if abs(float(out.sum())) > 0.01:
        return zero
    return out.astype(float)
