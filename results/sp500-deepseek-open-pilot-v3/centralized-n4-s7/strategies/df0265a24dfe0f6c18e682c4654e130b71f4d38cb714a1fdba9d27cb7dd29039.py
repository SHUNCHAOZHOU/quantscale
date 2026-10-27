import numpy as np
import pandas as pd

def _field(history, name):
    if isinstance(history, dict):
        return history.get(name)
    return None

def generate_weights(history, universe):
    u = pd.Index(universe)
    if len(u) == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=u)
    op = _field(history, 'adjusted_open')
    if op is None or not isinstance(op, pd.DataFrame) or op.dropna(how='all').empty:
        op = _field(history, 'open')
    if op is None or not isinstance(op, pd.DataFrame) or op.empty:
        return zero
    cols = [c for c in u if c in op.columns]
    if len(cols) == 0:
        return zero
    op = op.loc[:, cols]
    if not op.index.is_monotonic_increasing:
        op = op.sort_index()
    idx = op.index
    if len(idx) == 0:
        return zero
    wd = idx.dayofweek
    fri = np.where(wd == 4)[0]
    if len(fri) > 0:
        reb_pos = int(fri[-1])
    else:
        reb_pos = len(idx) - 1
    lookback = 5
    skip = 1
    need = lookback + skip + 1
    if reb_pos + 1 < need:
        return zero
    start_pos = max(0, reb_pos - (need + 30) + 1)
    op_win = op.iloc[start_pos:reb_pos+1].copy()
    if len(op_win) < need:
        return zero
    ret_all = op_win.shift(skip) / op_win.shift(skip + lookback) - 1.0
    ret_last = ret_all.iloc[-1]
    med = ret_last.median()
    sig = -(ret_last - med)
    last_open = op_win.iloc[-1]
    valid = sig.replace([np.inf, -np.inf], np.nan).notna() & (last_open > 0)
    vol = _field(history, 'volume')
    liq = None
    if isinstance(vol, pd.DataFrame) and not vol.empty:
        vol_sub = vol.reindex(index=idx).reindex(columns=cols)
        vol_win = vol_sub.iloc[start_pos:reb_pos+1]
        if len(vol_win) > 0:
            liq = vol_win.tail(20).mean(skipna=True)
            valid = valid & liq.notna() & (liq > 0)
            if valid.sum() >= 20:
                thresh = liq[valid].quantile(0.2)
                if np.isfinite(thresh):
                    valid = valid & (liq >= thresh)
    elig = _field(history, 'eligible')
    if isinstance(elig, pd.DataFrame) and not elig.empty:
        elig_sub = elig.reindex(index=idx).reindex(columns=cols)
        e = elig_sub.iloc[reb_pos]
        e = e.fillna(False).astype(bool)
        valid = valid & e
    candidates = sig[valid]
    if len(candidates) < 2:
        return zero
    n_side = min(25, len(candidates) // 2)
    if n_side <= 0:
        return zero
    cdf = pd.DataFrame({'signal': candidates.values, 'ticker': [str(x) for x in candidates.index]}, index=candidates.index)
    cdf = cdf.sort_values(['signal', 'ticker'], ascending=[False, True])
    longs = cdf.index[:n_side]
    shorts = cdf.index[-n_side:]
    w = 0.02
    out = pd.Series(0.0, index=u)
    for t in longs:
        if t in out.index:
            out.loc[t] = w
    for t in shorts:
        if t in out.index:
            out.loc[t] = -w
    return out.reindex(u).fillna(0.0)