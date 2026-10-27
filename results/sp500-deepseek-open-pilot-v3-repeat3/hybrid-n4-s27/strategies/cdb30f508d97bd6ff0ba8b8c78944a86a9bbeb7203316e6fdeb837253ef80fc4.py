import pandas as pd
import numpy as np

def generate_weights(history, universe):
    universe = pd.Index(universe)
    weights = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return weights

    close_df = history.get('adjusted_close')
    vol_df = history.get('volume')
    if close_df is None or vol_df is None:
        return weights

    idx = close_df.index
    if len(idx) < 6:
        return weights

    close = close_df.reindex(index=idx, columns=universe)
    vol = vol_df.reindex(index=idx, columns=universe)

    elig_df = history.get('eligible')
    if elig_df is not None:
        elig = elig_df.reindex(index=idx, columns=universe)
    else:
        elig = None

    lookback = 3
    liq_window = 20
    max_side = 40
    max_abs = 0.02

    last_pos = len(idx) - 1
    base_pos = last_pos - (last_pos % 5)
    min_pos = max(lookback, liq_window - 1)
    if base_pos < min_pos:
        base_pos = last_pos
    if base_pos < lookback:
        return weights

    base_close = close.iloc[base_pos]
    prev_close = close.iloc[base_pos - lookback]
    sig = base_close / prev_close - 1.0
    sig = sig.replace([np.inf, -np.inf], np.nan)

    start = max(0, base_pos - liq_window + 1)
    close_slice = close.iloc[start:base_pos + 1]
    vol_slice = vol.iloc[start:base_pos + 1]
    dollar_vol = close_slice * vol_slice
    liq = dollar_vol.median(axis=0, skipna=True)
    liq_count = dollar_vol.count(axis=0)
    liq = liq.where(liq_count >= 3, np.nan)

    if elig is not None:
        base_elig = elig.iloc[base_pos]
        elig_mask = base_elig.fillna(False).astype(bool)
        if int(elig_mask.sum()) == 0:
            elig_mask = pd.Series(True, index=universe, dtype=bool)
    else:
        elig_mask = pd.Series(True, index=universe, dtype=bool)

    valid = sig.notna() & np.isfinite(sig) & liq.notna() & (liq > 0.0) & elig_mask
    valid = valid & (base_close > 0.0)
    candidates = universe[valid]
    if len(candidates) < 2:
        return weights

    liq_c = liq.reindex(candidates).replace([np.inf, -np.inf], np.nan).dropna()
    if len(liq_c) < 2:
        return weights
    q20 = liq_c.quantile(0.2)
    candidates = liq_c[liq_c >= q20].index
    if len(candidates) < 2:
        return weights

    sig_c = sig.reindex(candidates)
    n = min(max_side, len(candidates) // 2)
    if n < 1:
        return weights

    ordered_low = sig_c.sort_values(ascending=True, kind='mergesort')
    ordered_high = sig_c.sort_values(ascending=False, kind='mergesort')
    longs = ordered_low.index[:n]
    shorts = ordered_high.index[:n]

    w = min(max_abs, 0.5 / float(n))
    weights.loc[longs] = w
    weights.loc[shorts] = -w

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    gross = weights.abs().sum()
    if gross > 1.0:
        weights = weights * (1.0 / gross)
    return weights