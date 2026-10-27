import numpy as np
import pandas as pd
import math


def _get_df(history, names):
    for name in names:
        obj = history.get(name)
        if isinstance(obj, pd.DataFrame) and not obj.empty:
            return obj
    return None


def _zero(universe):
    return pd.Series(0.0, index=universe, dtype=float)


def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)

    price = _get_df(history, ['adjusted_close', 'close'])
    if price is None:
        return _zero(universe)

    price = price.reindex(columns=universe).iloc[-260:].astype(float)
    if price.shape[0] < 80:
        return _zero(universe)

    last = price.iloc[-1]
    cnt = price.notna().sum()
    valid = last.notna() & (last > 5.0) & (cnt >= 120)

    vol_df = _get_df(history, ['volume'])
    if vol_df is not None:
        vol_df = vol_df.reindex(columns=universe).iloc[-60:].astype(float)
        vmed = vol_df.tail(20).median()
        pmed = price.tail(20).median()
        dv = pmed * vmed
        valid = valid & vmed.notna() & (vmed > 200000.0) & dv.notna() & (dv > 500000.0)
        if valid.sum() > 20:
            dv_valid = dv[valid].dropna()
            if len(dv_valid) >= 20:
                cutoff = dv_valid.quantile(0.98)
                valid = valid & ~(dv > cutoff)

    elig = _get_df(history, ['eligible'])
    if elig is not None:
        e = elig.reindex(columns=universe).iloc[-1]
        valid = valid & e.fillna(True).astype(bool)

    if valid.sum() < 6:
        return _zero(universe)

    rets = price / price.shift(1) - 1.0
    mkt = rets.mean(axis=1, skipna=True)

    r120 = rets.tail(120)
    m120 = mkt.tail(120)
    m_mean = m120.mean()
    m_c = m120 - m_mean
    den = (m_c * m_c).sum()

    if np.isfinite(den) and den > 0:
        numer = r120.sub(r120.mean(), axis=1).mul(m_c, axis=0).sum()
        beta = numer / den
        beta = beta.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.2, 2.5)
    else:
        beta = pd.Series(1.0, index=universe)

    vol = rets.tail(60).std()
    vol = vol.replace(0.0, np.nan).clip(lower=0.005)

    rr = rets.tail(21)
    mm = mkt.tail(21)
    rvals = rr.values
    mvals = mm.values
    bvals = beta.values
    resid = rvals - mvals[:, None] * bvals[None, :]
    resid_sum = np.nansum(resid, axis=0)
    z_rev = pd.Series(-resid_sum / (vol.values * math.sqrt(21.0)), index=universe)

    med_vol = np.nanmedian(vol.values)
    if np.isfinite(med_vol) and med_vol > 0:
        z_vol = -(vol / med_vol)
    else:
        z_vol = pd.Series(0.0, index=universe)

    z_beta = -beta
    score = 0.55 * z_rev + 0.25 * z_vol + 0.20 * z_beta
    score = score.replace([np.inf, -np.inf], np.nan)
    score = score.where(valid).dropna()

    if len(score) < 6:
        return _zero(universe)

    lo = score.quantile(0.02)
    hi = score.quantile(0.98)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        score = score.clip(lo, hi)

    n_side = min(40, len(score) // 2)
    if n_side < 1:
        return _zero(universe)

    sorted_score = score.sort_values(ascending=False)
    long_names = sorted_score.index[:n_side]
    short_names = sorted_score.index[-n_side:]

    w = min(0.5 / float(n_side), 0.02)
    weights = pd.Series(0.0, index=universe)
    weights.loc[long_names] += w
    weights.loc[short_names] -= w

    net = weights.sum()
    if abs(net) > 1e-12:
        idx = weights.index[weights != 0.0]
        if len(idx) > 0:
            weights.loc[idx] -= net / len(idx)

    weights = weights.clip(-0.02, 0.02)
    gross = weights.abs().sum()
    if gross > 1.0:
        weights = weights * (1.0 / gross)

    net = weights.sum()
    if abs(net) > 1e-12:
        idx = weights.index[weights != 0.0]
        if len(idx) > 0:
            weights.loc[idx] -= net / len(idx)

    weights = weights.clip(-0.02, 0.02)
    gross = weights.abs().sum()
    if gross > 1.0:
        weights = weights * (1.0 / gross)

    return weights.fillna(0.0).astype(float)