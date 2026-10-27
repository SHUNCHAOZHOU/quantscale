import numpy as np
import pandas as pd
import math


def _get_df(history, names):
    for name in names:
        obj = history.get(name)
        if isinstance(obj, pd.DataFrame):
            return obj
    return None


def _zscore(s):
    s = s.replace([np.inf, -np.inf], np.nan)
    x = s.dropna()
    if len(x) < 2:
        return pd.Series(np.nan, index=s.index)
    mu = x.mean()
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)

    zeros = pd.Series(0.0, index=universe)

    price = _get_df(history, ['adjusted_close', 'close'])
    if price is None:
        return zeros

    price = price.reindex(columns=universe)
    if price.shape[0] < 300:
        return zeros

    idx = price.index
    periods = idx.to_period('M')
    cur_period = periods[-1]

    asof_pos = -1
    for i in range(len(idx) - 1, -1, -1):
        if periods[i] < cur_period:
            asof_pos = i
            break

    if asof_pos < 260:
        return zeros

    p = price.iloc[:asof_pos + 1].tail(300)
    if p.shape[0] < 260:
        return zeros

    last = p.iloc[-1]
    cnt = p.notna().sum()
    valid = last.notna() & (last > 2.0) & (cnt >= 180)

    vol_df = _get_df(history, ['volume'])
    if vol_df is not None:
        vol_df = vol_df.reindex(index=p.index, columns=universe)
        vmed = vol_df.tail(60).median()
        valid = valid & (vmed > 100000)

    elig_df = _get_df(history, ['eligible'])
    if elig_df is not None:
        e = elig_df.reindex(index=p.index, columns=universe).iloc[-1]
        valid = valid & e.fillna(True).astype(bool)

    rets = p.pct_change()
    r = rets.tail(180)
    rcnt = r.notna().sum()
    valid = valid & (rcnt >= 120)

    if valid.sum() < 4:
        return zeros

    m = r.mean(axis=1)
    m_var = m.var()
    if not np.isfinite(m_var) or m_var <= 0:
        return zeros

    cov = r.apply(lambda col: col.cov(m))
    beta = cov / m_var
    vol = r.std()

    maxret = r.tail(120).max()

    R = r.values.astype(float)
    M = m.values.astype(float)
    B = beta.values.astype(float)
    B = np.where(np.isfinite(B), B, 1.0)
    resid = R - M.reshape(-1, 1) * B.reshape(1, -1)
    idio_arr = np.nanstd(resid, axis=0, ddof=1)
    idio = pd.Series(idio_arr, index=universe)

    valid = valid & idio.notna() & beta.notna()

    divy = pd.Series(0.0, index=universe)
    div_df = _get_df(history, ['dividends'])
    if div_df is not None:
        d = div_df.reindex(index=p.index, columns=universe).fillna(0.0)
        ttm = d.tail(252).sum()
        divy = ttm / last.replace(0.0, np.nan)
        divy = divy.where(np.isfinite(divy), 0.0).fillna(0.0)

    ivol_z = _zscore(idio.where(valid)).fillna(0.0)
    beta_z = _zscore(beta.where(valid)).fillna(0.0)
    max_z = _zscore(maxret.where(valid)).fillna(0.0)
    div_z = _zscore(divy.where(valid)).fillna(0.0)

    score = -0.55 * ivol_z - 0.25 * beta_z - 0.20 * max_z + 0.15 * div_z
    score = score.where(valid).replace([np.inf, -np.inf], np.nan).dropna()

    if len(score) < 10:
        score = (-idio).where(valid).replace([np.inf, -np.inf], np.nan).dropna()

    if len(score) < 4:
        return zeros

    n_side = min(40, len(score) // 2)
    if n_side < 2:
        return zeros

    srt = score.sort_values(ascending=False)
    long_names = srt.index[:n_side]
    short_names = srt.index[-n_side:]

    w = min(0.5 / n_side, 0.02)
    weights = pd.Series(0.0, index=universe)
    weights.loc[long_names] += w
    weights.loc[short_names] -= w

    net = weights.sum()
    if abs(net) > 1e-12:
        active = weights.index[weights != 0.0]
        if len(active) > 0:
            weights.loc[active] -= net / len(active)

    weights = weights.clip(-0.02, 0.02)
    gross = weights.abs().sum()
    if gross > 1.0:
        weights = weights * (1.0 / gross)

    return weights.fillna(0.0).astype(float)