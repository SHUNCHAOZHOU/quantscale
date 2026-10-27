import numpy as np
import pandas as pd

def _rank_pct_valid(x, valid):
    s = pd.Series(x[valid])
    return s.rank(pct=True, method='average').values

def generate_weights(history, universe):
    cols = pd.Index(universe)
    out = pd.Series(0.0, index=cols)
    if not isinstance(history, dict) or 'adjusted_close' not in history:
        return out
    ac_full = history['adjusted_close']
    if not isinstance(ac_full, pd.DataFrame) or ac_full.empty:
        return out
    ac = ac_full.reindex(columns=cols)
    n = len(ac)
    if n < 65:
        return out
    freq = 10
    pos = ((n - 1) // freq) * freq
    if pos < 60:
        return out
    elig_full = history.get('eligible', None)
    if isinstance(elig_full, pd.DataFrame):
        e = elig_full.reindex(index=ac.index, columns=cols)
        elig_pos = e.iloc[pos].fillna(False).astype(bool).values
        elig_now = e.iloc[-1].fillna(False).astype(bool).values
        elig = elig_pos & elig_now
    else:
        elig = np.ones(len(cols), dtype=bool)
    p0 = ac.iloc[pos].to_numpy(dtype=float)
    p5 = ac.iloc[pos-5].to_numpy(dtype=float)
    p60 = ac.iloc[pos-60].to_numpy(dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        r5 = p0 / p5 - 1.0
        mom = p5 / p60 - 1.0
    window = ac.iloc[pos-20:pos+1].to_numpy(dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        rets = window[1:] / window[:-1] - 1.0
    vol = np.nanstd(rets, axis=0, ddof=0)
    valid = (np.isfinite(r5) & np.isfinite(mom) & np.isfinite(vol) &
             np.isfinite(p0) & (p0 > 0) & np.isfinite(p5) & (p5 > 0) &
             np.isfinite(p60) & (p60 > 0) & elig)
    m = int(valid.sum())
    if m < 2:
        return out
    r5_rank = _rank_pct_valid(r5, valid)
    mom_rank = _rank_pct_valid(mom, valid)
    vol_rank = _rank_pct_valid(vol, valid)
    comp_valid = 0.35 * (1.0 - r5_rank) + 0.35 * mom_rank + 0.30 * (1.0 - vol_rank)
    valid_idx = np.flatnonzero(valid)
    comp = pd.Series(np.nan, index=cols, dtype=float)
    comp.iloc[valid_idx] = comp_valid
    comp = comp.dropna()
    if len(comp) < 2:
        return out
    K = min(60, len(comp) // 2)
    if K <= 0:
        return out
    w = min(0.5 / K, 0.02)
    sorted_comp = comp.sort_values(ascending=False, kind='mergesort')
    longs = sorted_comp.index[:K]
    shorts = sorted_comp.index[-K:]
    out.loc[longs] = w
    out.loc[shorts] = -w
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out