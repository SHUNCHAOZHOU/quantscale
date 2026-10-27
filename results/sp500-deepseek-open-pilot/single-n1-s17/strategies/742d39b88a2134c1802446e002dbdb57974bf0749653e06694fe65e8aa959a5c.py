import numpy as np
import pandas as pd


def generate_weights(history, universe):
    uni = pd.Index(universe)
    if len(uni) == 0:
        return pd.Series(dtype=float)

    adj = history.get('adjusted_close')
    if adj is None:
        adj = history.get('close')
    if adj is None or not isinstance(adj, pd.DataFrame) or len(adj.index) == 0:
        return pd.Series(0.0, index=uni, dtype=float)

    dates = adj.index
    adj = adj.reindex(columns=uni)

    close = history.get('close')
    if close is not None:
        close = close.reindex(index=dates, columns=uni)
    volume = history.get('volume')
    if volume is not None:
        volume = volume.reindex(index=dates, columns=uni)
    eligible = history.get('eligible')
    if eligible is not None:
        eligible = eligible.reindex(index=dates, columns=uni)

    years = dates.year.values
    months = dates.month.values
    is_start = np.zeros(len(dates), dtype=bool)
    is_start[0] = True
    if len(dates) > 1:
        is_start[1:] = (years[1:] != years[:-1]) | (months[1:] != months[:-1])
    reb_pos = int(np.where(is_start)[0][-1])

    if reb_pos + 1 < 253:
        return pd.Series(0.0, index=uni, dtype=float)

    p = adj.iloc[:reb_pos + 1].to_numpy(dtype=float, copy=False)
    if p.shape[0] < 253:
        return pd.Series(0.0, index=uni, dtype=float)

    p0 = p[-1]
    p21 = p[-22]
    p126 = p[-127]
    p252 = p[-253]

    valid = np.isfinite(p0) & (p0 > 0)
    mom12 = p21 / p252 - 1.0
    mom6 = p21 / p126 - 1.0
    rev = -(p0 / p21 - 1.0)

    rets = p[1:] / p[:-1] - 1.0
    ret_df = pd.DataFrame(rets, columns=uni)
    vol = ret_df.tail(60).std(ddof=1, skipna=True).to_numpy(dtype=float)

    valid &= np.isfinite(mom12) & np.isfinite(mom6) & np.isfinite(rev) & np.isfinite(vol)

    if eligible is not None:
        try:
            e_row = eligible.iloc[reb_pos].fillna(False).astype(bool).to_numpy()
            valid &= e_row
        except Exception:
            pass

    liq = None
    if close is not None and volume is not None:
        c = close.iloc[:reb_pos + 1].to_numpy(dtype=float, copy=False)
        v = volume.iloc[:reb_pos + 1].to_numpy(dtype=float, copy=False)
        dv = c * v
        if dv.shape[0] > 0:
            liq = np.full(dv.shape[1], np.nan)
            tail = dv[-60:]
            for j in range(dv.shape[1]):
                col = tail[:, j]
                col = col[np.isfinite(col)]
                if col.size:
                    liq[j] = np.median(col)

    if liq is not None:
        finite_liq = np.isfinite(liq) & (liq > 0)
        base = liq[valid & finite_liq]
        if base.size >= 10:
            cutoff = np.quantile(base, 0.20)
            valid &= finite_liq & (liq >= cutoff)
        else:
            valid &= finite_liq

    idxs = np.where(valid)[0]
    if idxs.size < 20:
        return pd.Series(0.0, index=uni, dtype=float)

    cand = uni[idxs]

    def rank_pct(arr):
        s = pd.Series(arr, index=cand)
        return s.rank(method='average', pct=True).to_numpy(dtype=float)

    r_mom12 = rank_pct(mom12[idxs])
    r_mom6 = rank_pct(mom6[idxs])
    r_rev = rank_pct(rev[idxs])
    r_vol = rank_pct(-vol[idxs])

    score_arr = 0.40 * r_mom12 + 0.25 * r_mom6 + 0.25 * r_rev + 0.10 * r_vol
    score = pd.Series(score_arr, index=cand)
    ordered = score.sort_values(ascending=False, kind='mergesort')

    n = int(min(25, len(ordered) // 2))
    if n <= 0:
        return pd.Series(0.0, index=uni, dtype=float)

    longs = list(ordered.index[:n])
    shorts = list(ordered.index[-n:])
    w = pd.Series(0.0, index=uni, dtype=float)
    pos = min(0.0195, 0.5 / float(n))
    w.loc[longs] = pos
    w.loc[shorts] = -pos
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return w