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
    if len(cols) < 8:
        return zero

    idx = adj.index
    n = len(idx)
    rebal = 21
    min_obs = 140
    apos = ((n - 1) // rebal) * rebal
    if apos < min_obs:
        return zero

    anchor_date = idx[apos]
    px = adj.loc[:anchor_date, cols].ffill()
    na = len(px)
    if na < min_obs:
        return zero

    last = px.iloc[-1].to_numpy(dtype=float)
    valid = np.isfinite(last) & (last > 0.0)

    elig = _field(history, 'eligible')
    if elig is not None:
        epos = int(elig.index.searchsorted(anchor_date, side='right'))
        if epos > 0:
            ev = elig.iloc[epos - 1].reindex(cols).fillna(False).astype(bool)
            valid = valid & ev.to_numpy(dtype=bool)

    vol_df = _field(history, 'volume')
    close_df = _field(history, 'close')
    if vol_df is not None:
        v = vol_df.reindex(index=px.index, columns=cols).to_numpy(dtype=float)
        if close_df is not None:
            c = close_df.reindex(index=px.index, columns=cols).to_numpy(dtype=float)
        else:
            c = px.to_numpy(dtype=float)
        w = min(60, na)
        with np.errstate(all='ignore'):
            dv = v[-w:] * c[-w:]
            med = np.nanmedian(dv, axis=0)
        valid = valid & np.isfinite(med) & (med >= 1e6)

    arr = px.to_numpy(dtype=float)

    def ret_over(lb):
        out = np.full(len(cols), np.nan, dtype=float)
        if na - 1 - lb < 0:
            return out
        with np.errstate(all='ignore'):
            r = arr[-1] / arr[-1 - lb] - 1.0
        r[~np.isfinite(r)] = np.nan
        return r

    rev5 = -ret_over(5)
    rev21 = -ret_over(21)
    rev63 = -ret_over(63)

    def rank_pct(x):
        out = np.full(len(x), np.nan, dtype=float)
        m = np.isfinite(x)
        cnt = int(m.sum())
        if cnt == 0:
            return out
        r = pd.Series(x[m]).rank(method='average').to_numpy(dtype=float)
        out[m] = (r - 0.5) / float(cnt)
        return out

    score = np.zeros(len(cols), dtype=float)
    wsum = np.zeros(len(cols), dtype=float)
    for x, wt in ((rev21, 0.5), (rev63, 0.3), (rev5, 0.2)):
        r = rank_pct(x)
        has = np.isfinite(r)
        score[has] += wt * r[has]
        wsum[has] += wt
    with np.errstate(all='ignore'):
        score = np.where(wsum > 0, score / wsum, np.nan)

    cand = np.where(valid & np.isfinite(score))[0]
    if cand.size < 8:
        return zero

    n_side = int(min(50, cand.size // 2))
    if n_side < 4:
        return zero

    order = cand[np.argsort(score[cand])]
    shorts = order[:n_side]
    longs = order[-n_side:]

    w = float(min(0.02, 0.5 / float(n_side)))
    res = pd.Series(0.0, index=cols, dtype=float)
    res.iloc[longs] = w
    res.iloc[shorts] = -w

    out = res.reindex(universe).fillna(0.0)
    out = out.clip(-0.02, 0.02)
    gross = float(out.abs().sum())
    if gross > 1.0 and gross > 0.0:
        out = out * (1.0 / gross)

    net = float(out.sum())
    if abs(net) > 1e-9:
        active = out[out != 0.0]
        if len(active) > 0:
            out.loc[active.index] = out.loc[active.index] - net / float(len(active))
            out = out.clip(-0.02, 0.02)
            gross = float(out.abs().sum())
            if gross > 1.0 and gross > 0.0:
                out = out * (1.0 / gross)

    return out.fillna(0.0).astype(float)
