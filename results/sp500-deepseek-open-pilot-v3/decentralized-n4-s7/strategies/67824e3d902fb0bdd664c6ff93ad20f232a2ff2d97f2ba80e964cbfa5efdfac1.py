import numpy as np
import pandas as pd


def _rank_z(s, valid):
    s = s.astype(float).copy()
    s[~valid] = np.nan
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 2:
        return pd.Series(0.0, index=s.index)
    r = s.rank(pct=True)
    return (2.0 * r - 1.0).fillna(0.0)


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
    if len(cols) < 2:
        return zeros

    px = ac.loc[:, cols].astype(float)
    if not px.index.is_monotonic_increasing:
        px = px.sort_index()

    has = px.notna().any(axis=1)
    if not has.any():
        return zeros
    t = px.index[has][-1]
    px = px.loc[:t]
    if len(px) < 30:
        return zeros

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
    if n < 30:
        return zeros
    p0 = sub.iloc[-1]

    def ret(offset):
        if n - 1 - offset < 0:
            return pd.Series(np.nan, index=cols)
        return sub.iloc[-1] / sub.iloc[-1 - offset] - 1.0

    ret21 = ret(21)
    ret5 = ret(5)
    rev21 = -ret21
    rev5 = -ret5

    recent = sub.tail(127)
    rets = recent.pct_change()
    if len(rets) > 1:
        r126 = rets.tail(126)
        vol = r126.std()
    else:
        r126 = pd.DataFrame(index=rets.index, columns=cols, dtype=float)
        vol = pd.Series(np.nan, index=cols)

    beta = pd.Series(1.0, index=cols)
    mkt21 = np.nan
    if len(r126) >= 20:
        m = r126.mean(axis=1)
        mv = m.var()
        if pd.notna(mv) and mv > 1e-12:
            mean_r = r126.mean(axis=0)
            cov = r126.mul(m, axis=0).mean(axis=0) - mean_r * m.mean()
            beta = cov / mv
        if len(m) >= 21:
            mkt21 = (1.0 + m.tail(21)).prod() - 1.0

    if pd.notna(mkt21):
        resid21 = ret21 - beta * mkt21
    else:
        resid21 = ret21
    rev_resid = -resid21

    dvol = pd.Series(np.nan, index=cols)
    v = history.get('volume')
    c = history.get('close')
    if v is not None and c is not None:
        vv = v.reindex(columns=cols).loc[:sig]
        cc = c.reindex(columns=cols).loc[:sig]
        if len(vv) > 0 and len(cc) > 0:
            vv = vv.tail(20)
            cc = cc.tail(20)
            dvol = (vv * cc).median(axis=0, skipna=True)

    eligible = pd.Series(True, index=cols)
    e = history.get('eligible')
    if e is not None:
        e = e.reindex(columns=cols)
        if sig in e.index:
            eligible = e.loc[sig].fillna(False).astype(bool)
        elif t in e.index:
            eligible = e.loc[t].fillna(False).astype(bool)

    valid = p0.notna() & (p0 > 0.5) & rev21.notna() & rev5.notna() & vol.notna() & (vol > 0)
    valid = valid & eligible.reindex(cols).fillna(False)

    if dvol.notna().any():
        pos = dvol[dvol > 0]
        if len(pos) > 0:
            thresh = pos.quantile(0.20)
            valid = valid & dvol.notna() & (dvol >= thresh)
        else:
            valid = valid & dvol.notna()
    valid = valid.fillna(False)

    if int(valid.sum()) < 10:
        return zeros

    z_resid = _rank_z(rev_resid, valid)
    z_rev21 = _rank_z(rev21, valid)
    z_rev5 = _rank_z(rev5, valid)
    z_vol = _rank_z(vol, valid)

    score = 0.55 * z_resid + 0.20 * z_rev21 + 0.15 * z_rev5 - 0.15 * z_vol
    score = score.replace([np.inf, -np.inf], np.nan)
    score[~valid] = np.nan
    score = score.dropna()
    if len(score) < 10:
        return zeros

    n_side = min(50, len(score) // 2)
    if n_side < 1:
        return zeros

    ranked = score.sort_values()
    shorts = ranked.index[:n_side].tolist()
    longs = ranked.index[-n_side:].tolist()

    w_side = min(0.02, 0.5 / float(n_side))
    w = pd.Series(0.0, index=u)
    w.loc[longs] = w_side
    w.loc[shorts] = -w_side
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
    return w