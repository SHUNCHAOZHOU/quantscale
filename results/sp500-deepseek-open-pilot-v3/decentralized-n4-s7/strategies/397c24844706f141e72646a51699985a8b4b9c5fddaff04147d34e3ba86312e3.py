import numpy as np
import pandas as pd


def _zscore(s):
    s = s.astype(float).replace([np.inf, -np.inf], np.nan)
    med = s.median()
    mad = (s - med).abs().median()
    if pd.notna(mad) and mad > 1e-12:
        z = (s - med) / (1.4826 * mad)
    else:
        std = s.std()
        if pd.notna(std) and std > 1e-12:
            z = (s - s.mean()) / std
        else:
            z = pd.Series(0.0, index=s.index)
    return z.clip(-5.0, 5.0)


def generate_weights(history, universe):
    u = pd.Index(universe)
    zeros = pd.Series(0.0, index=u)
    if len(u) == 0:
        return zeros

    ac = history.get('adjusted_close')
    if ac is None or ac.empty:
        ac = history.get('close')
    if ac is None or ac.empty:
        return zeros

    cols = [t for t in u if t in ac.columns]
    if len(cols) < 20:
        return zeros

    px = ac.loc[:, cols].astype(float)
    if not px.index.is_monotonic_increasing:
        px = px.sort_index()

    has = px.notna().any(axis=1)
    if not has.any():
        return zeros
    t = px.index[has][-1]
    px = px.loc[:t]
    if len(px) < 130:
        return zeros

    # Monthly signal date: last month-end on or before t.
    try:
        nb = t + pd.offsets.BDay(1)
        at_month_end = (nb.year != t.year) or (nb.month != t.month)
    except Exception:
        at_month_end = False

    if at_month_end:
        sig = t
    else:
        mask = (px.index.year < t.year) | ((px.index.year == t.year) & (px.index.month < t.month))
        prior = px.index[mask]
        if len(prior) == 0:
            return zeros
        sig = prior[-1]

    sub = px.loc[:sig]
    n = len(sub)
    if n < 130:
        return zeros

    rets = sub.pct_change()
    look = min(126, n - 1)
    if look < 60:
        return zeros

    r = rets.iloc[-look:]
    # Equal-weighted market return each day.
    mkt = r.mean(axis=1, skipna=True)

    mean_r = r.mean(axis=0)
    mean_m = mkt.mean()
    var_m = mkt.var()

    if pd.isna(var_m) or var_m <= 1e-12:
        vol = r.std(axis=0)
        beta = pd.Series(1.0, index=cols)
        resid_vol = vol
    else:
        cov = r.sub(mean_r, axis=1).mul(mkt - mean_m, axis=0).mean(axis=0)
        beta = cov / var_m
        resid_arr = r.values - np.outer(mkt.values, beta.values)
        resid = pd.DataFrame(resid_arr, index=r.index, columns=r.columns)
        resid_vol = resid.std(axis=0)
        vol = r.std(axis=0)

    p0 = sub.iloc[-1]

    # Liquidity: median dollar volume over last 20 sessions.
    dvol = pd.Series(np.nan, index=cols)
    v = history.get('volume')
    c = history.get('close')
    if v is not None and c is not None:
        vv = v.reindex(columns=cols).reindex(sub.index).tail(20)
        cc = c.reindex(columns=cols).reindex(sub.index).tail(20)
        if len(vv) > 0 and len(cc) > 0:
            dvol = (vv * cc).median(axis=0, skipna=True)

    eligible = pd.Series(True, index=cols)
    e = history.get('eligible')
    if e is not None:
        e = e.reindex(columns=cols)
        if sig in e.index:
            eligible = e.loc[sig].fillna(False).astype(bool)
        elif t in e.index:
            eligible = e.loc[t].fillna(False).astype(bool)

    valid = (
        p0.notna()
        & (p0 > 1.0)
        & vol.notna()
        & (vol > 0.0)
        & resid_vol.notna()
        & (resid_vol > 0.0)
        & eligible.reindex(cols).fillna(False)
    )

    if dvol.notna().any():
        pos = dvol[dvol > 0.0]
        if len(pos) > 0:
            thresh = pos.quantile(0.20)
            valid = valid & dvol.notna() & (dvol >= thresh)
        else:
            valid = valid & dvol.notna()
    valid = valid.fillna(False)

    if int(valid.sum()) < 40:
        # Relax liquidity/price filter if too few names.
        valid = (
            p0.notna()
            & (p0 > 1.0)
            & vol.notna()
            & (vol > 0.0)
            & resid_vol.notna()
            & (resid_vol > 0.0)
            & eligible.reindex(cols).fillna(False)
        ).fillna(False)
        if int(valid.sum()) < 20:
            return zeros

    z_resid = _zscore(resid_vol.where(valid))
    z_beta = _zscore(beta.where(valid))

    # Long low idiosyncratic volatility, short high idiosyncratic volatility.
    # Mildly penalize high beta to reduce residual market exposure.
    score = -z_resid - 0.30 * z_beta
    score = score.replace([np.inf, -np.inf], np.nan)
    score[~valid] = np.nan
    score = score.dropna()

    if len(score) < 20:
        return zeros

    n_side = min(25, len(score) // 2)
    if n_side < 1:
        return zeros
    w_side = min(0.02, 0.5 / float(n_side))

    ranked = score.sort_values()
    shorts = ranked.index[:n_side]
    longs = ranked.index[-n_side:]

    w = zeros.copy()
    w.loc[longs] = w_side
    w.loc[shorts] = -w_side
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
    return w
