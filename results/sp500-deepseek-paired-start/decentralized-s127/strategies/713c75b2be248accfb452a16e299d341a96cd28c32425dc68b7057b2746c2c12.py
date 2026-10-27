import numpy as np
import pandas as pd


def _rank_pct(values, valid):
    out = np.full(values.shape[0], np.nan, dtype=float)
    idx = np.where(valid & np.isfinite(values))[0]
    m = idx.size
    if m == 0:
        return out
    r = pd.Series(values[idx]).rank(method='average').to_numpy(dtype=float)
    out[idx] = (r - 0.5) / float(m)
    return out


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    ac_all = history.get('adjusted_close')
    if not isinstance(ac_all, pd.DataFrame) or ac_all.empty:
        ac_all = history.get('close')
    if not isinstance(ac_all, pd.DataFrame) or ac_all.empty:
        return zero

    cols = [c for c in universe if c in ac_all.columns]
    if len(cols) < 2:
        return zero

    ac = ac_all.loc[:, cols].sort_index()
    n_rows = ac.shape[0]
    if n_rows < 80:
        return zero

    rebalance = 10
    last_i = n_rows - 1
    anchor_i = last_i - (last_i % rebalance)
    if anchor_i < 60:
        return zero

    ac_slice = ac.iloc[:anchor_i + 1].ffill()
    prices = ac_slice.to_numpy(dtype=float, copy=True)
    n_p = prices.shape[0]
    n_cols = prices.shape[1]
    last_p = prices[-1]

    elig_df = history.get('eligible')
    if isinstance(elig_df, pd.DataFrame):
        e = elig_df.reindex(index=ac.index, columns=cols).iloc[anchor_i]
        e_last = e.fillna(False).astype(bool).to_numpy()
    else:
        e_last = np.ones(n_cols, dtype=bool)

    valid = np.isfinite(last_p) & (last_p > 0.0) & e_last

    vol_df = history.get('volume')
    close_df = history.get('close')
    if isinstance(vol_df, pd.DataFrame):
        v = vol_df.reindex(index=ac.index, columns=cols).iloc[:anchor_i + 1].to_numpy(dtype=float, copy=True)
        if isinstance(close_df, pd.DataFrame):
            px = close_df.reindex(index=ac.index, columns=cols).iloc[:anchor_i + 1].to_numpy(dtype=float, copy=True)
        else:
            px = prices
        w = min(60, n_p)
        dollar = v[-w:] * px[-w:]
        with np.errstate(all='ignore'):
            med_dv = np.nanmedian(dollar, axis=0)
        valid = valid & np.isfinite(med_dv) & (med_dv >= 1000000.0)

    rets = np.full((n_p, n_cols), np.nan, dtype=float)
    rets[1:] = prices[1:] / prices[:-1] - 1.0
    vol_w = min(60, n_p - 1)
    if vol_w < 10:
        return zero
    with np.errstate(all='ignore'):
        vol = np.nanstd(rets[-vol_w:], axis=0)
    valid = valid & np.isfinite(vol) & (vol > 0.0)
    if int(valid.sum()) < 4:
        return zero

    def past_return(lb):
        if n_p <= lb:
            return np.full(n_cols, np.nan, dtype=float)
        out = prices[-1] / prices[-1 - lb] - 1.0
        out[~np.isfinite(out)] = np.nan
        return out

    ret5 = past_return(5)
    ret21 = past_return(21)
    with np.errstate(all='ignore'):
        rev5 = -ret5 / vol
        rev21 = -ret21 / vol
        tw = min(200, n_p)
        ma = np.nanmean(prices[-tw:], axis=0)
        trend = last_p / ma - 1.0
    trend[~np.isfinite(trend)] = np.nan
    lowvol = -vol

    score_sum = np.zeros(n_cols, dtype=float)
    score_w = np.zeros(n_cols, dtype=float)
    for vals, wgt in ((rev5, 0.45), (rev21, 0.25), (trend, 0.20), (lowvol, 0.10)):
        rk = _rank_pct(vals, valid)
        have = np.isfinite(rk)
        score_sum[have] += wgt * rk[have]
        score_w[have] += wgt
    score = np.full(n_cols, np.nan, dtype=float)
    good = score_w > 0.0
    score[good] = score_sum[good] / score_w[good]

    cand = np.where(valid & np.isfinite(score))[0]
    if cand.size < 10:
        return zero
    n_side = int(min(50, cand.size // 2))
    if n_side < 5:
        return zero

    order = cand[np.argsort(score[cand])]
    shorts = order[:n_side]
    longs = order[-n_side:]
    side_w = min(0.02, 0.5 / float(n_side))

    weights = np.zeros(n_cols, dtype=float)
    weights[longs] = side_w
    weights[shorts] = -side_w

    out = pd.Series(0.0, index=universe, dtype=float)
    out.loc[cols] = weights
    out = out.clip(-0.02, 0.02).fillna(0.0)
    gross = float(out.abs().sum())
    if gross > 1.0 and gross > 0.0:
        out = out * (1.0 / gross)
    net = float(out.sum())
    if abs(net) > 0.01:
        out = out - net / float(len(universe))
        out = out.clip(-0.02, 0.02).fillna(0.0)
    return out.astype(float)