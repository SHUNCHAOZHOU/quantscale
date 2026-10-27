import numpy as np
import pandas as pd


def generate_weights(history, universe):
    tickers = pd.Index(universe)
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    zeros = pd.Series(0.0, index=tickers, dtype=float)

    ac = history.get('adjusted_close')
    if ac is None or not isinstance(ac, pd.DataFrame) or ac.shape[0] == 0:
        ac = history.get('close')
    if ac is None or not isinstance(ac, pd.DataFrame) or ac.shape[0] == 0:
        return zeros

    ac = ac.reindex(columns=tickers)
    dates = ac.index
    if len(dates) == 0:
        return zeros

    years = dates.year.values
    months = dates.month.values
    cur_y = years[-1]
    cur_m = months[-1]
    mask = (years < cur_y) | ((years == cur_y) & (months < cur_m))
    if mask.any():
        ref_pos = int(np.where(mask)[0][-1])
    else:
        ref_pos = len(dates) - 1
    if ref_pos < 25:
        return zeros

    start = max(0, ref_pos - 300)
    c = ac.iloc[start:ref_pos + 1]
    if c.shape[0] < 22:
        return zeros

    elig = pd.Series(True, index=tickers)
    edf = history.get('eligible')
    if isinstance(edf, pd.DataFrame) and edf.shape[0] > 0:
        edf = edf.reindex(columns=tickers)
        ref_date = dates[ref_pos]
        if ref_date in edf.index:
            erow = edf.loc[ref_date]
        else:
            sub = edf.loc[:ref_date]
            erow = sub.iloc[-1] if len(sub) > 0 else pd.Series(True, index=tickers)
        elig = erow.reindex(tickers).fillna(False).astype(bool)

    last = pd.to_numeric(c.iloc[-1], errors='coerce')
    valid = last.notna() & (last > 1.0) & elig

    vdf = history.get('volume')
    if isinstance(vdf, pd.DataFrame) and vdf.shape[0] > 0:
        vdf = vdf.reindex(columns=tickers).reindex(index=c.index)
        cdf = history.get('close')
        if isinstance(cdf, pd.DataFrame) and cdf.shape[0] > 0:
            cdf = cdf.reindex(columns=tickers).reindex(index=c.index)
        else:
            cdf = c
        dv = (vdf.iloc[-20:] * cdf.iloc[-20:]).median()
        valid = valid & dv.notna() & (dv > 0)

    ret = c.pct_change().replace([np.inf, -np.inf], np.nan)
    n = c.shape[0]
    p_recent = c.iloc[-22]
    if n >= 253:
        p_base = c.iloc[-253]
    else:
        p_base = c.iloc[0]
    mom = p_recent / p_base - 1.0
    if n >= 60:
        vol = ret.iloc[-60:].std()
    else:
        vol = ret.std()
    score = mom / vol
    score = score.replace([np.inf, -np.inf], np.nan)
    valid = valid & score.notna() & vol.notna() & (vol > 1e-8)
    cnt = c.iloc[-60:].notna().sum()
    valid = valid & (cnt >= 20)

    if int(valid.sum()) < 6:
        return zeros

    mkt = ret.mean(axis=1)
    scale = 1.0
    if len(mkt) >= 60:
        mv20 = mkt.iloc[-20:].std()
        if len(mkt) >= 252:
            mv_long = mkt.iloc[-252:].std()
        else:
            mv_long = mkt.std()
        if np.isfinite(mv20) and np.isfinite(mv_long) and mv_long > 1e-8:
            ratio = mv20 / mv_long
            if ratio > 1.6:
                scale = 0.5
            elif ratio > 1.3:
                scale = 0.75

    scores = score[valid]
    n_side = min(35, len(scores) // 2)
    if n_side < 3:
        return zeros

    ranked = scores.sort_values(ascending=False)
    longs = list(ranked.index[:n_side])
    shorts = list(ranked.index[-n_side:])
    long_set = set(longs)
    shorts = [t for t in shorts if t not in long_set]
    n_side = min(len(longs), len(shorts))
    if n_side < 1:
        return zeros

    side_gross = 0.5 * scale
    w = side_gross / float(n_side)
    if w > 0.02:
        w = 0.02

    weights = zeros.copy()
    for t in longs[:n_side]:
        weights.loc[t] = w
    for t in shorts[:n_side]:
        weights.loc[t] = -w
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights