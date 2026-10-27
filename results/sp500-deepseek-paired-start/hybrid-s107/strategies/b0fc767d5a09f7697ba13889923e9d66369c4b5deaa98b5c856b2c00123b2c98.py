import numpy as np
import pandas as pd


def _rank_center(s):
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return s
    return s.rank(method='average', pct=True) - 0.5


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=tickers, dtype=float)

    px = history.get('adjusted_close')
    if not isinstance(px, pd.DataFrame):
        px = history.get('close')
    if not isinstance(px, pd.DataFrame) or px.shape[0] < 300:
        return zero
    if not px.index.is_monotonic_increasing:
        px = px.sort_index()

    cols = [c for c in tickers if c in px.columns]
    if len(cols) < 30:
        return zero

    n = len(px)
    period = 21
    reb = ((n - 1) // period) * period
    if reb < 260:
        return zero

    lookback = 260
    start = max(0, reb - lookback + 1)
    sub = px.iloc[start:reb + 1][cols].astype(float)
    sub = sub.replace([np.inf, -np.inf], np.nan)
    if sub.shape[0] < 260:
        return zero

    last = sub.iloc[-1]
    valid = last.notna() & (last > 0)
    valid = valid & (sub.notna().sum(axis=0) >= 200)

    decision_date = px.index[reb]
    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame) and elig.shape[0] > 0:
        if not elig.index.is_monotonic_increasing:
            elig = elig.sort_index()
        if decision_date in elig.index:
            e = elig.loc[decision_date].reindex(cols).fillna(False).astype(bool)
            valid = valid & e

    vdf = history.get('volume')
    if isinstance(vdf, pd.DataFrame) and vdf.shape[0] > 0:
        if not vdf.index.is_monotonic_increasing:
            vdf = vdf.sort_index()
        vcols = [c for c in cols if c in vdf.columns]
        if len(vcols) > 0:
            vsub = vdf.reindex(index=sub.index, columns=vcols).astype(float)
            vsub = vsub.replace([np.inf, -np.inf], np.nan)
            dollar = vsub * sub[vcols]
            liq = dollar.tail(60).median(axis=0).reindex(cols)
            valid = valid & liq.notna() & (liq > 0)
            if liq.notna().sum() >= 50:
                thr = liq.quantile(0.20)
                if np.isfinite(thr):
                    valid = valid & (liq >= thr)

    idx = valid[valid].index
    if len(idx) < 30:
        return zero

    R = sub[idx].pct_change()
    R = R.replace([np.inf, -np.inf], np.nan)
    R_win = R.tail(180)
    if R_win.shape[0] < 120:
        return zero

    mkt = R_win.mean(axis=1, skipna=True)
    Rf = R_win.fillna(0.0)
    mf = mkt.fillna(0.0)

    Rb = Rf.tail(120)
    mb = mf.tail(120)
    mbd = mb - mb.mean()
    var = float((mbd.values ** 2).sum())
    if var <= 1e-18:
        beta = pd.Series(1.0, index=idx, dtype=float)
    else:
        beta = Rb.mul(mbd, axis=0).sum(axis=0) / var
    beta = beta.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    m_comp = pd.DataFrame(
        mf.values[:, None] * beta.values[None, :],
        index=Rf.index,
        columns=idx,
    )
    resid = Rf - m_comp

    resid_vol = resid.tail(60).std(axis=0, ddof=1)
    resid_vol = resid_vol.replace([np.inf, -np.inf], np.nan)
    res20 = resid.tail(20).sum(axis=0)
    rev = -res20 / (resid_vol * np.sqrt(20.0) + 1e-12)
    rev = rev.replace([np.inf, -np.inf], np.nan)

    try:
        corr_short = Rf.tail(60).corrwith(mf.tail(60))
        corr_long = Rf.tail(180).corrwith(mf.tail(180))
        corr_break = corr_long - corr_short
    except Exception:
        corr_break = pd.Series(np.nan, index=idx, dtype=float)
    corr_break = corr_break.replace([np.inf, -np.inf], np.nan)

    ivol = resid.tail(120).std(axis=0, ddof=1)
    ivol = ivol.replace([np.inf, -np.inf], np.nan)

    r_rev = _rank_center(rev.reindex(idx))
    r_break = _rank_center(corr_break.reindex(idx))
    r_ivol = _rank_center(-ivol.reindex(idx))
    if len(r_rev) < 20:
        return zero

    score = (
        0.60 * r_rev.reindex(idx).fillna(0.0)
        + 0.25 * r_break.reindex(idx).fillna(0.0)
        + 0.15 * r_ivol.reindex(idx).fillna(0.0)
    )
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 30:
        return zero

    n_side = min(50, len(score) // 3)
    if n_side < 10:
        return zero

    longs = score.nlargest(n_side).index
    shorts = score.nsmallest(n_side).index
    if len(set(longs).intersection(set(shorts))) > 0:
        return zero

    wmag = min(0.02, 0.5 / float(n_side))
    w = pd.Series(0.0, index=tickers, dtype=float)
    w.loc[longs] = wmag
    w.loc[shorts] = -wmag

    sel = list(longs) + list(shorts)
    b = beta.reindex(sel).fillna(1.0).astype(float)
    wsel = w.reindex(sel).astype(float)
    b_center = b - b.mean()
    denom = float((b_center * b_center).sum())
    if denom > 1e-18:
        c = float((wsel * b_center).sum()) / denom
        wsel = wsel - c * b_center

    w = pd.Series(0.0, index=tickers, dtype=float)
    w.loc[sel] = wsel.values
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    w = w.clip(-0.02, 0.02)

    gross = float(w.abs().sum())
    if gross > 1.0:
        w = w * (1.0 / gross)

    for _ in range(5):
        net = float(w.sum())
        if abs(net) <= 0.005:
            break
        active = w[w != 0]
        if len(active) == 0:
            break
        d = net / float(len(active))
        w.loc[active.index] = active - d
        w = w.clip(-0.02, 0.02)
        gross = float(w.abs().sum())
        if gross > 1.0:
            w = w * (1.0 / gross)

    w = w.clip(-0.02, 0.02)
    gross = float(w.abs().sum())
    if gross > 1.0 + 1e-12:
        w = w * (1.0 / gross)

    net = float(w.sum())
    if abs(net) > 0.01:
        active = w[w != 0]
        if len(active) > 0:
            d = net / float(len(active))
            w.loc[active.index] = active - d
            w = w.clip(-0.02, 0.02)
            gross = float(w.abs().sum())
            if gross > 1.0:
                w = w * (1.0 / gross)

    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return w.reindex(tickers).fillna(0.0).astype(float)