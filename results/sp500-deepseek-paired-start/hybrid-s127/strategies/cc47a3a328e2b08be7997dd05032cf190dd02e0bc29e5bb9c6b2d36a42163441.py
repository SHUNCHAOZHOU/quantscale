import numpy as np
import pandas as pd


def _rank_pct(values, mask):
    out = np.full(len(values), np.nan, dtype=float)
    idx = np.where(mask & np.isfinite(values))[0]
    if idx.size == 0:
        return out
    ranks = pd.Series(values[idx]).rank(method='average').to_numpy(dtype=float)
    out[idx] = (ranks - 0.5) / float(idx.size)
    return out


def generate_weights(history, universe):
    universe = pd.Index(universe)
    n_univ = len(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if n_univ == 0:
        return zero

    ac_all = history.get('adjusted_close')
    if not isinstance(ac_all, pd.DataFrame):
        ac_all = history.get('close')
    if not isinstance(ac_all, pd.DataFrame) or ac_all.empty:
        return zero

    cols = [c for c in universe if c in ac_all.columns]
    if len(cols) < 4:
        return zero

    ac = ac_all.loc[:, cols]
    n_rows = len(ac)
    if n_rows < 140:
        return zero

    rebal = 21
    apos = ((n_rows - 1) // rebal) * rebal
    if apos < 130:
        return zero

    sub_index = ac.index[:apos + 1]
    prices = ac.iloc[:apos + 1].to_numpy(dtype=float, copy=True)
    n_cols = prices.shape[1]
    last_p = prices[-1]
    valid = np.isfinite(last_p) & (last_p > 0.0)

    elig_df = history.get('eligible')
    if isinstance(elig_df, pd.DataFrame):
        e = elig_df.reindex(index=sub_index, columns=cols)
        last_e = e.iloc[-1].fillna(False).astype(bool).to_numpy()
        valid &= last_e

    vol_df = history.get('volume')
    if isinstance(vol_df, pd.DataFrame):
        v = vol_df.reindex(index=sub_index, columns=cols).to_numpy(dtype=float, copy=True)
        close_df = history.get('close')
        if isinstance(close_df, pd.DataFrame):
            px = close_df.reindex(index=sub_index, columns=cols).to_numpy(dtype=float, copy=True)
        else:
            px = prices
        w = min(60, len(prices))
        with np.errstate(all='ignore'):
            dollar = v[-w:] * px[-w:]
            med_dv = np.nanmedian(dollar, axis=0)
        valid &= np.isfinite(med_dv) & (med_dv >= 1000000.0)

    rets = np.full_like(prices, np.nan)
    rets[1:] = prices[1:] / prices[:-1] - 1.0
    rw = min(60, len(prices) - 1)
    if rw >= 2:
        with np.errstate(all='ignore'):
            vola = np.nanstd(rets[-rw:], axis=0)
    else:
        vola = np.full(n_cols, np.nan)
    valid &= np.isfinite(vola) & (vola > 0.0)

    def mom(lookback, skip):
        end = len(prices) - 1 - skip
        start = end - lookback
        if start < 0:
            return np.full(n_cols, np.nan, dtype=float)
        out = prices[end] / prices[start] - 1.0
        out[~np.isfinite(out)] = np.nan
        return out

    mom12 = mom(252, 21)
    mom6 = mom(126, 21)
    mom_main = np.where(np.isfinite(mom12), mom12, mom6)

    if len(prices) >= 200:
        with np.errstate(all='ignore'):
            ma200 = np.nanmean(prices[-200:], axis=0)
        trend = prices[-1] / ma200 - 1.0
        trend[~np.isfinite(trend)] = np.nan
    else:
        trend = np.full(n_cols, np.nan, dtype=float)

    r12 = _rank_pct(mom_main, valid)
    r6 = _rank_pct(mom6, valid)
    rv = _rank_pct(-vola, valid)
    rt = _rank_pct(trend, valid)

    ssum = np.zeros(n_cols, dtype=float)
    wsum = np.zeros(n_cols, dtype=float)
    for vals, wgt in ((r12, 0.55), (r6, 0.25), (rv, 0.15), (rt, 0.05)):
        has = np.isfinite(vals)
        ssum[has] += wgt * vals[has]
        wsum[has] += wgt

    score = np.full(n_cols, np.nan, dtype=float)
    good = wsum > 0.0
    score[good] = ssum[good] / wsum[good]

    cand = np.where(valid & np.isfinite(score))[0]
    if cand.size < 4:
        return zero

    n_side = int(min(50, cand.size // 2))
    if n_side < 1:
        return zero

    order = cand[np.argsort(score[cand])]
    shorts = order[:n_side]
    longs = order[-n_side:]

    w = float(min(0.02, 0.5 / float(n_side)))
    weights = np.zeros(n_cols, dtype=float)
    weights[longs] = w
    weights[shorts] = -w

    out = zero.copy()
    for i, ticker in enumerate(cols):
        if weights[i] != 0.0:
            out.loc[ticker] = weights[i]

    out = out.clip(-0.02, 0.02).fillna(0.0)
    gross = float(out.abs().sum())
    if gross > 1.0 and gross > 0.0:
        out = out * (1.0 / gross)

    net = float(out.sum())
    if abs(net) > 1e-12:
        out = out - net / float(n_univ)
        out = out.clip(-0.02, 0.02)
        gross = float(out.abs().sum())
        if gross > 1.0 and gross > 0.0:
            out = out * (1.0 / gross)

    return out.astype(float)