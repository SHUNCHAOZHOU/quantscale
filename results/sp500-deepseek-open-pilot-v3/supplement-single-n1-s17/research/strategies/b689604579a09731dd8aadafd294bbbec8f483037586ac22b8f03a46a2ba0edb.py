import numpy as np
import pandas as pd

def _get_df(history, name):
    df = history.get(name)
    if isinstance(df, pd.DataFrame):
        return df
    return None

def generate_weights(history, universe):
    cols = pd.Index(pd.unique(pd.Index(universe).astype(str)))
    if len(cols) == 0:
        return pd.Series(dtype=float)
    zeros = pd.Series(0.0, index=cols, dtype=float)
    ac = _get_df(history, 'adjusted_close')
    if ac is None:
        ac = _get_df(history, 'close')
    if ac is None or ac.empty:
        return zeros
    available = [c for c in cols if c in ac.columns]
    if not available:
        return zeros
    ac = ac.loc[:, available].sort_index().tail(320)
    if ac.empty:
        return zeros
    dates = ac.index
    last_date = dates[-1]
    wd = dates.weekday
    fridays = dates[wd == 4]
    sig_date = fridays[-1] if len(fridays) else last_date
    hist_ac = ac.loc[:sig_date]
    if len(hist_ac.index) < 6:
        hist_ac = ac
        sig_date = last_date
    if len(hist_ac.index) < 6:
        return zeros
    px = hist_ac.iloc[-1]
    ret5 = hist_ac.iloc[-1] / hist_ac.iloc[-6] - 1.0
    rets = hist_ac.pct_change().tail(20)
    vol = rets.std(ddof=0)
    elig = pd.Series(True, index=available)
    edf = _get_df(history, 'eligible')
    if edf is not None:
        ee = edf.reindex(columns=available).loc[:sig_date].tail(1)
        if len(ee.index) > 0:
            elig = ee.iloc[0].fillna(False).astype(bool)
    liq = pd.Series(np.nan, index=available)
    vdf = _get_df(history, 'volume')
    if vdf is not None:
        vv = vdf.reindex(columns=available).loc[:sig_date].tail(20)
        pp = hist_ac.tail(20)
        common = vv.index.intersection(pp.index)
        if len(common) > 0:
            dv = vv.loc[common].mul(pp.loc[common], axis=0)
            liq = dv.mean()
    base = (px.notna() & ret5.notna() & vol.notna() & (vol > 0) & (px > 1.0) & elig)
    base = base & np.isfinite(px) & np.isfinite(ret5) & np.isfinite(vol)
    if liq.notna().any():
        base = base & (liq.fillna(0.0) > 1e6)
    if int(base.sum()) < 10:
        base = (px.notna() & ret5.notna() & vol.notna() & (vol > 0) & (px > 0.5) & elig)
        base = base & np.isfinite(px) & np.isfinite(ret5) & np.isfinite(vol)
    candidates = [c for c in available if bool(base.get(c, False))]
    if len(candidates) < 2:
        return zeros
    r = ret5.loc[candidates].astype(float)
    long_order = sorted(candidates, key=lambda c: (float(r.get(c, np.nan)), c))
    short_order = sorted(candidates, key=lambda c: (-float(r.get(c, np.nan)), c))
    max_n = min(40, len(candidates) // 2)
    if max_n < 1:
        return zeros
    longs = long_order[:max_n]
    used = set(longs)
    shorts = [c for c in short_order if c not in used][:max_n]
    N = min(len(longs), len(shorts))
    if N < 1:
        return zeros
    longs = longs[:N]
    shorts = shorts[:N]
    w = float(min(0.5 / N, 0.02))
    weights = zeros.copy()
    for c in longs:
        if c in weights.index:
            weights.loc[c] = w
    for c in shorts:
        if c in weights.index:
            weights.loc[c] = -w
    net = float(weights.sum())
    if abs(net) > 1e-12:
        nz = weights.index[weights != 0.0]
        if len(nz) > 0:
            weights.loc[nz] -= net / len(nz)
    weights = weights.clip(-0.02, 0.02)
    gross = float(weights.abs().sum())
    if gross > 1.0:
        weights = weights * (1.0 / gross)
    return weights.fillna(0.0)