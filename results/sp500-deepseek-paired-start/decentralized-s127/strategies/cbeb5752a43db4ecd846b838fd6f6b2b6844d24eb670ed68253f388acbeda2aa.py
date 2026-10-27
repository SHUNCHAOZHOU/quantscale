import numpy as np
import pandas as pd


def _get_df(history, key):
    x = history.get(key)
    if isinstance(x, pd.DataFrame) and x.shape[0] > 0:
        return x
    return None


def _rank_pct_arr(vals, mask):
    out = np.full(vals.shape[0], np.nan, dtype=float)
    m = mask & np.isfinite(vals)
    if int(m.sum()) == 0:
        return out
    out[m] = pd.Series(vals[m]).rank(method='average', pct=True).to_numpy(dtype=float)
    return out


def _signals_at(q, prices, ret, excess):
    n_c = prices.shape[1]
    last = prices[q]

    start5 = max(0, q - 4)
    ex5 = np.nansum(excess[start5:q + 1], axis=0)
    cnt5 = np.sum(np.isfinite(excess[start5:q + 1]), axis=0)

    start21 = max(0, q - 20)
    ex21 = np.nansum(excess[start21:q + 1], axis=0)
    cnt21 = np.sum(np.isfinite(excess[start21:q + 1]), axis=0)

    vstart = max(0, q - 59)
    vol60 = np.nanstd(ret[vstart:q + 1], axis=0)

    tstart = max(0, q - 199)
    ma = np.nanmean(prices[tstart:q + 1], axis=0)

    with np.errstate(all='ignore'):
        rev5 = -ex5 / (vol60 * np.sqrt(5.0))
        rev21 = -ex21 / (vol60 * np.sqrt(21.0))
        lowvol = -vol60
        trend = last / ma - 1.0

    base = np.isfinite(last) & (last > 0.0) & np.isfinite(vol60) & (vol60 > 0.0)
    rev5 = np.where((cnt5 >= 3) & base, rev5, np.nan)
    rev21 = np.where((cnt21 >= 8) & base, rev21, np.nan)
    lowvol = np.where(base, lowvol, np.nan)
    trend = np.where(np.isfinite(ma) & (ma > 0.0) & base, trend, np.nan)

    return {'rev5': rev5, 'rev21': rev21, 'lowvol': lowvol, 'trend': trend}


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    ac_all = _get_df(history, 'adjusted_close')
    if ac_all is None:
        ac_all = _get_df(history, 'close')
    if ac_all is None:
        return zero

    cols = [c for c in universe if c in ac_all.columns]
    if len(cols) < 30:
        return zero

    ac = ac_all.loc[:, cols].sort_index()
    n_rows = ac.shape[0]
    if n_rows < 180:
        return zero

    rebalance = 21
    anchor = ((n_rows - 1) // rebalance) * rebalance
    if anchor < 140:
        return zero

    px_df = ac.iloc[:anchor + 1].ffill()
    prices = px_df.to_numpy(dtype=float, copy=True)
    n_p, n_c = prices.shape
    if n_p < 140:
        return zero

    last = prices[-1]

    elig_df = _get_df(history, 'eligible')
    if elig_df is not None:
        e = elig_df.reindex(index=ac.index, columns=cols)
        if anchor < len(e):
            e_last = e.iloc[anchor].fillna(False).astype(bool).to_numpy(dtype=bool)
        else:
            e_last = np.zeros(n_c, dtype=bool)
    else:
        e_last = np.ones(n_c, dtype=bool)

    valid = np.isfinite(last) & (last > 1.0) & e_last

    vol_df = _get_df(history, 'volume')
    close_df = _get_df(history, 'close')
    if vol_df is not None:
        v = vol_df.reindex(index=ac.index, columns=cols).iloc[:anchor + 1].ffill().to_numpy(dtype=float, copy=True)
        if close_df is not None:
            c = close_df.reindex(index=ac.index, columns=cols).iloc[:anchor + 1].ffill().to_numpy(dtype=float, copy=True)
        else:
            c = prices
        w_liq = min(60, n_p)
        with np.errstate(all='ignore'):
            dollar = v[-w_liq:] * c[-w_liq:]
            med_dv = np.nanmedian(dollar, axis=0)
        valid = valid & np.isfinite(med_dv) & (med_dv >= 1000000.0)

    ret = np.full_like(prices, np.nan, dtype=float)
    ret[1:] = prices[1:] / prices[:-1] - 1.0
    ret[~np.isfinite(ret)] = np.nan

    mkt = np.nanmean(ret, axis=1)
    excess = ret - mkt[:, None]

    vstart = max(0, n_p - 60)
    vol60 = np.nanstd(ret[vstart:], axis=0)
    valid = valid & np.isfinite(vol60) & (vol60 > 0.0) & (vol60 < 0.20)

    if int(valid.sum()) < 30:
        return zero

    signal_names = ['rev5', 'rev21', 'lowvol', 'trend']
    ic_lists = {k: [] for k in signal_names}

    for k in range(1, 7):
        q = anchor - rebalance * k
        if q < 140 or q + rebalance > anchor:
            continue
        sigs_q = _signals_at(q, prices, ret, excess)
        with np.errstate(all='ignore'):
            fwd = prices[q + rebalance] / prices[q] - 1.0
        fwd[~np.isfinite(fwd)] = np.nan
        for name in signal_names:
            s = sigs_q[name]
            mask = np.isfinite(s) & np.isfinite(fwd) & (prices[q] > 0.0)
            if int(mask.sum()) < 30:
                continue
            rs = pd.Series(s[mask]).rank(method='average').to_numpy(dtype=float)
            rf = pd.Series(fwd[mask]).rank(method='average').to_numpy(dtype=float)
            if np.std(rs) <= 1e-12 or np.std(rf) <= 1e-12:
                continue
            ic = np.corrcoef(rs, rf)[0, 1]
            if np.isfinite(ic):
                ic_lists[name].append(float(ic))

    weights_sig = {}
    for name in signal_names:
        ics = ic_lists[name]
        if len(ics) >= 4:
            mean_ic = float(np.mean(ics))
            if abs(mean_ic) >= 0.010:
                weights_sig[name] = float(np.clip(mean_ic, -0.08, 0.08))

    if len(weights_sig) == 0:
        return zero

    sig_anchor = _signals_at(anchor, prices, ret, excess)
    ranks = {}
    for name in signal_names:
        ranks[name] = _rank_pct_arr(sig_anchor[name], valid)

    score_sum = np.zeros(n_c, dtype=float)
    abs_sum = np.zeros(n_c, dtype=float)
    for name, wt in weights_sig.items():
        r = ranks[name]
        has = np.isfinite(r)
        score_sum[has] += wt * r[has]
        abs_sum[has] += abs(wt)

    score = np.full(n_c, np.nan, dtype=float)
    good = abs_sum > 0.0
    score[good] = score_sum[good] / abs_sum[good]

    cand = np.where(valid & np.isfinite(score))[0]
    if cand.size < 30:
        return zero

    n_side = int(min(40, cand.size // 2))
    if n_side < 10:
        return zero

    order = cand[np.argsort(score[cand])]
    shorts = order[:n_side]
    longs = order[-n_side:]

    side_w = float(min(0.02, 0.35 / float(n_side)))
    w_arr = np.zeros(n_c, dtype=float)
    w_arr[longs] = side_w
    w_arr[shorts] = -side_w

    out = zero.copy()
    out.loc[cols] = w_arr
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02).astype(float)

    net = float(out.sum())
    if abs(net) > 1e-12:
        out = out - net / float(len(universe))
        out = out.clip(-0.02, 0.02).fillna(0.0)

    gross = float(out.abs().sum())
    if gross > 1.0 and gross > 0.0:
        out = out * (1.0 / gross)

    return out.astype(float)