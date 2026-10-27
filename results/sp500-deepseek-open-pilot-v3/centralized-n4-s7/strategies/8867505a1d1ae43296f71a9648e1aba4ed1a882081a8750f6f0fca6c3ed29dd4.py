import numpy as np
import pandas as pd

def _get(history, names):
    if not isinstance(history, dict):
        return None
    for name in names:
        df = history.get(name)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return None

def generate_weights(history, universe):
    u = pd.Index(universe)
    if len(u) == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=u)
    close = _get(history, ['adjusted_close', 'close'])
    if close is None:
        return zero
    cols = [c for c in u if c in close.columns]
    if len(cols) == 0:
        return zero
    close = close.loc[:, cols]
    if not close.index.is_monotonic_increasing:
        close = close.sort_index()
    idx = close.index
    n = len(idx)
    if n < 60:
        return zero
    months = idx.month.values
    years = idx.year.values
    is_first = np.ones(n, dtype=bool)
    is_first[1:] = (months[1:] != months[:-1]) | (years[1:] != years[:-1])
    first_pos = np.where(is_first)[0]
    if len(first_pos) == 0:
        return zero
    reb_pos = int(first_pos[-1])
    skip = 21
    lookback = 252
    if reb_pos - skip - lookback < 0:
        lookback = 126
        if reb_pos - skip - lookback < 0:
            return zero
    end_pos = reb_pos - skip
    start_pos = end_pos - lookback
    if start_pos < 0:
        return zero
    c_end = close.iloc[end_pos]
    c_start = close.iloc[start_pos]
    mom = c_end / c_start - 1.0
    valid = mom.replace([np.inf, -np.inf], np.nan).notna() & (c_end > 0) & (c_start > 0)
    volume = _get(history, ['volume'])
    if isinstance(volume, pd.DataFrame) and not volume.empty:
        v = volume.reindex(index=idx, columns=cols)
        liq = v.iloc[max(0, reb_pos - 20):reb_pos + 1].mean(axis=0, skipna=True)
        valid = valid & liq.notna() & (liq > 0)
        if valid.sum() >= 40:
            thresh = liq[valid].quantile(0.2)
            if np.isfinite(thresh):
                valid = valid & (liq >= thresh)
    elig = _get(history, ['eligible'])
    if isinstance(elig, pd.DataFrame) and not elig.empty:
        e = elig.reindex(index=idx, columns=cols).iloc[reb_pos].fillna(False).astype(bool)
        valid = valid & e
    candidates = mom[valid]
    if len(candidates) < 40:
        return zero
    disp = candidates.std(skipna=True)
    if not np.isfinite(disp) or disp < 0.08:
        return zero
    cdf = pd.DataFrame({'signal': candidates.values, 'ticker': [str(x) for x in candidates.index]}, index=candidates.index)
    cdf = cdf.sort_values(['signal', 'ticker'], ascending=[False, True])
    n_side = min(20, len(cdf) // 2)
    if n_side <= 0:
        return zero
    longs = cdf.index[:n_side]
    shorts = cdf.index[-n_side:]
    w = min(0.02, 0.5 / n_side)
    out = pd.Series(0.0, index=u)
    for t in longs:
        if t in out.index:
            out.loc[t] = w
    for t in shorts:
        if t in out.index:
            out.loc[t] = -w
    return out.reindex(u).fillna(0.0)