import numpy as np
import pandas as pd


def generate_weights(history, universe):
    universe = pd.Index(universe)
    n_univ = len(universe)
    if n_univ == 0:
        return pd.Series(dtype=float, index=universe)

    zero = pd.Series(0.0, index=universe, dtype=float)
    try:
        ac_all = history['adjusted_close']
    except Exception:
        return zero
    if not isinstance(ac_all, pd.DataFrame):
        return zero

    cols = [c for c in universe if c in ac_all.columns]
    if len(cols) == 0:
        return zero

    ac = ac_all.loc[:, cols]
    n_rows = ac.shape[0]
    if n_rows < 130:
        return zero

    prices = ac.to_numpy(dtype=float, copy=True)
    n_cols = prices.shape[1]
    if n_cols == 0:
        return zero

    last_p = prices[-1]

    elig_df = history.get('eligible')
    if isinstance(elig_df, pd.DataFrame):
        e = elig_df.reindex(index=ac.index, columns=cols)
        last_e = e.iloc[-1].fillna(False).astype(bool).to_numpy()
    else:
        last_e = np.ones(n_cols, dtype=bool)

    valid = np.isfinite(last_p) & (last_p > 0.0) & last_e

    vol_df = history.get('volume')
    if isinstance(vol_df, pd.DataFrame):
        v = vol_df.reindex(index=ac.index, columns=cols).to_numpy(dtype=float, copy=True)
        close_df = history.get('close')
        if isinstance(close_df, pd.DataFrame):
            px = close_df.reindex(index=ac.index, columns=cols).to_numpy(dtype=float, copy=True)
        else:
            px = prices
        w_liq = min(60, n_rows)
        with np.errstate(all='ignore'):
            dollar = v[-w_liq:] * px[-w_liq:]
            med_dv = np.nanmedian(dollar, axis=0)
        valid &= np.isfinite(med_dv) & (med_dv >= 1000000.0)

    rets = np.full((n_rows, n_cols), np.nan, dtype=float)
    with np.errstate(all='ignore'):
        rets[1:] = prices[1:] / prices[:-1] - 1.0

    cnt = np.sum(np.isfinite(rets), axis=1)
    sums = np.nansum(rets, axis=1)
    mkt = np.full(n_rows, np.nan, dtype=float)
    good_rows = cnt >= 20
    mkt[good_rows] = sums[good_rows] / cnt[good_rows]

    L = min(252, n_rows - 1)
    shortL = min(126, L)
    if L < 60:
        return zero

    R = rets[-L:]
    M = mkt[-L:]
    beta = np.full(n_cols, np.nan, dtype=float)
    rvol = np.full(n_cols, np.nan, dtype=float)

    for i in range(n_cols):
        y = R[:, i]
        mask = np.isfinite(y) & np.isfinite(M)
        n_obs = int(mask.sum())
        if n_obs < max(40, L // 2):
            continue
        x = M[mask]
        yy = y[mask]
        xm = x.mean()
        ym = yy.mean()
        dx = x - xm
        varx = float(np.sum(dx * dx))
        if varx <= 0.0 or not np.isfinite(varx):
            continue
        b = float(np.sum(dx * (yy - ym)) / varx)
        a = ym - b * xm
        resid = yy - a - b * x
        sl = min(shortL, n_obs)
        if sl < 20:
            continue
        rv = float(np.std(resid[-sl:], ddof=0))
        if np.isfinite(b) and np.isfinite(rv) and rv > 0.0:
            beta[i] = b
            rvol[i] = rv

    valid &= np.isfinite(beta) & np.isfinite(rvol) & (rvol > 0.0)
    cand = np.where(valid)[0]
    if cand.size < 80:
        return zero

    n_buckets = int(min(10, max(2, cand.size // 20)))
    beta_order = cand[np.argsort(beta[cand], kind='mergesort')]
    buckets = np.array_split(beta_order, n_buckets)

    long_parts = []
    short_parts = []
    for buck in buckets:
        if buck.size < 2:
            continue
        order_rv = buck[np.argsort(rvol[buck], kind='mergesort')]
        side = int(min(5, buck.size // 2))
        if side < 1:
            continue
        long_parts.append(order_rv[:side])
        short_parts.append(order_rv[-side:])

    if len(long_parts) == 0:
        return zero
    longs = np.concatenate(long_parts)
    shorts = np.concatenate(short_parts)
    total_side = int(longs.size)
    if total_side != int(shorts.size) or total_side < 20:
        return zero

    target_gross = 0.8
    w_per = target_gross / (2.0 * float(total_side))
    if w_per > 0.02:
        w_per = 0.02

    out = zero.copy()
    for idx in longs:
        ticker = cols[int(idx)]
        out.loc[ticker] = float(w_per)
    for idx in shorts:
        ticker = cols[int(idx)]
        out.loc[ticker] = -float(w_per)

    gross = float(out.abs().sum())
    if gross > 1.0 and gross > 0.0:
        out = out * (1.0 / gross)
    return out.astype(float)