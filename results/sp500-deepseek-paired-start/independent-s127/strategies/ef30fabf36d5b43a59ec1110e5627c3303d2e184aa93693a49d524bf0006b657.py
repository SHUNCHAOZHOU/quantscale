import numpy as np
import pandas as pd


def _rank_pct(values, valid):
    out = np.full(len(values), np.nan, dtype=float)
    idx = np.where(valid & np.isfinite(values))[0]
    if idx.size == 0:
        return out
    ranks = pd.Series(values[idx]).rank(method='average').to_numpy(dtype=float)
    out[idx] = (ranks - 0.5) / float(idx.size)
    return out


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
    if n_rows < 70:
        return zero
    prices = ac.to_numpy(dtype=float, copy=True)
    n_cols = prices.shape[1]

    elig_df = history.get('eligible')
    if isinstance(elig_df, pd.DataFrame):
        e = elig_df.reindex(index=ac.index, columns=cols)
        last_e = e.iloc[-1].fillna(False).astype(bool).to_numpy()
    else:
        last_e = np.ones(n_cols, dtype=bool)

    last_p = prices[-1]
    valid = np.isfinite(last_p) & (last_p > 0.0) & last_e

    vol_df = history.get('volume')
    if isinstance(vol_df, pd.DataFrame):
        v = vol_df.reindex(index=ac.index, columns=cols)
        vv = v.to_numpy(dtype=float, copy=True)
        close_df = history.get('close')
        if isinstance(close_df, pd.DataFrame):
            px = close_df.reindex(index=ac.index, columns=cols).to_numpy(dtype=float, copy=True)
        else:
            px = prices
        w_liq = min(60, n_rows)
        dollar = vv[-w_liq:] * px[-w_liq:]
        with np.errstate(all='ignore'):
            med_dv = np.nanmedian(dollar, axis=0)
        valid &= np.isfinite(med_dv) & (med_dv >= 1000000.0)

    rets = np.full((n_rows, n_cols), np.nan, dtype=float)
    with np.errstate(all='ignore'):
        rets[1:] = prices[1:] / prices[:-1] - 1.0

    rw = min(60, n_rows - 1)
    if rw >= 5:
        with np.errstate(all='ignore'):
            vola = np.nanstd(rets[-rw:], axis=0)
    else:
        vola = np.full(n_cols, np.nan)
    valid &= np.isfinite(vola) & (vola > 0.0)

    def trailing_ret(lookback):
        end_i = n_rows - 1
        start_i = end_i - lookback
        if start_i < 0:
            return np.full(n_cols, np.nan, dtype=float)
        p_end = prices[end_i]
        p_start = prices[start_i]
        with np.errstate(all='ignore'):
            out = p_end / p_start - 1.0
        out[~np.isfinite(out)] = np.nan
        return out

    r21 = trailing_ret(21)
    r63 = trailing_ret(63)

    # MAX effect: maximum single-day return over last 21 sessions.
    mw = min(21, n_rows - 1)
    if mw >= 5:
        with np.errstate(all='ignore'):
            maxret = np.nanmax(rets[-mw:], axis=0)
    else:
        maxret = np.full(n_cols, np.nan, dtype=float)

    rev21 = -r21
    rev63 = -r63
    lowvol = -vola
    maxeff = -maxret

    f21 = _rank_pct(rev21, valid)
    f63 = _rank_pct(rev63, valid)
    fv = _rank_pct(lowvol, valid)
    fm = _rank_pct(maxeff, valid)

    signal = np.full(n_cols, np.nan, dtype=float)
    wsum = np.zeros(n_cols, dtype=float)
    ssum = np.zeros(n_cols, dtype=float)
    for vals, wgt in ((f21, 0.50), (f63, 0.20), (fv, 0.15), (fm, 0.15)):
        has = np.isfinite(vals)
        ssum[has] += wgt * vals[has]
        wsum[has] += wgt
    good = wsum > 0.0
    signal[good] = ssum[good] / wsum[good]

    cand = np.where(valid & np.isfinite(signal))[0]
    if cand.size < 4:
        return zero
    n_side = int(min(40, cand.size // 2))
    if n_side < 2:
        return zero
    order = cand[np.argsort(signal[cand], kind='mergesort')]
    shorts = order[:n_side]
    longs = order[-n_side:]
    w = min(0.02, 0.5 / float(n_side))
    weights = np.zeros(n_cols, dtype=float)
    weights[longs] = w
    weights[shorts] = -w

    out = zero.copy()
    for i, ticker in enumerate(cols):
        val = weights[i]
        if val != 0.0:
            out.loc[ticker] = float(val)

    out = out.clip(-0.02, 0.02).fillna(0.0)
    gross = float(out.abs().sum())
    if gross > 1.0 and gross > 0.0:
        out = out * (1.0 / gross)
    net = float(out.sum())
    if abs(net) > 0.01 and n_univ > 0:
        adj = net / float(n_univ)
        out = out - adj
        out = out.clip(-0.02, 0.02)
        gross = float(out.abs().sum())
        if gross > 1.0 and gross > 0.0:
            out = out * (1.0 / gross)
    return out.astype(float)
