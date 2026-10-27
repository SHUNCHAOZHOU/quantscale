import numpy as np
import pandas as pd


def _rank_pct_valid(x, valid):
    s = pd.Series(x[valid])
    return s.rank(pct=True, method='average').to_numpy()


def generate_weights(history, universe):
    cols = pd.Index(universe)
    out = pd.Series(0.0, index=cols, dtype=float)
    if not isinstance(history, dict):
        return out
    ac_full = history.get('adjusted_close')
    if not isinstance(ac_full, pd.DataFrame) or ac_full.empty:
        return out
    ac = ac_full.reindex(columns=cols)
    n = len(ac)
    freq = 21
    if n < 130:
        return out
    pos = ((n - 1) // freq) * freq
    if pos < 126:
        return out

    elig_full = history.get('eligible')
    if isinstance(elig_full, pd.DataFrame):
        e = elig_full.reindex(index=ac.index, columns=cols)
        elig_pos = e.iloc[pos].fillna(False).astype(bool).to_numpy()
        elig_now = e.iloc[-1].fillna(False).astype(bool).to_numpy()
    else:
        elig_pos = np.ones(len(cols), dtype=bool)
        elig_now = elig_pos.copy()

    p0 = ac.iloc[pos].to_numpy(dtype=float)
    p21 = ac.iloc[pos - 21].to_numpy(dtype=float)
    p126 = ac.iloc[pos - 126].to_numpy(dtype=float)

    ma_window = ac.iloc[pos - 99:pos + 1].to_numpy(dtype=float)
    ma100 = np.nanmean(ma_window, axis=0)

    vol_window = ac.iloc[pos - 60:pos + 1].to_numpy(dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        rets = vol_window[1:] / vol_window[:-1] - 1.0
    vol = np.nanstd(rets, axis=0, ddof=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        mom = p21 / p126 - 1.0

    valid = (
        np.isfinite(p0) & (p0 > 0)
        & np.isfinite(p21) & (p21 > 0)
        & np.isfinite(p126) & (p126 > 0)
        & np.isfinite(ma100) & (p0 > ma100)
        & np.isfinite(mom) & np.isfinite(vol) & (vol > 0)
        & elig_pos
    )
    m = int(valid.sum())
    if m < 20:
        return out

    mom_rank = _rank_pct_valid(mom, valid)
    vol_rank = _rank_pct_valid(vol, valid)
    comp_valid = 0.65 * mom_rank + 0.35 * (1.0 - vol_rank)

    valid_idx = np.flatnonzero(valid)
    comp = pd.Series(np.nan, index=cols, dtype=float)
    comp.iloc[valid_idx] = comp_valid
    comp = comp.dropna()
    if len(comp) < 20:
        return out

    K = min(50, len(comp))
    sorted_comp = comp.sort_values(ascending=False, kind='mergesort')
    longs = sorted_comp.index[:K]

    trend_ok = np.isfinite(p0) & (p0 > 0) & np.isfinite(ma100) & elig_pos
    if int(trend_ok.sum()) > 0:
        breadth = float(np.mean(p0[trend_ok] > ma100[trend_ok]))
    else:
        breadth = 0.0
    scale = 1.0
    if breadth < 0.35:
        scale = 0.5
    elif breadth < 0.50:
        scale = 0.75

    w = min(0.02, (0.90 * scale) / float(K))
    out.loc[longs] = w

    current_mask = pd.Series(elig_now, index=cols)
    out = out.where(current_mask, 0.0)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out = out.clip(lower=-0.02, upper=0.02)
    return out
