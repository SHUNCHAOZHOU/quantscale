import numpy as np
import pandas as pd


def _rank_pct(s):
    s = s.dropna()
    if len(s) == 0:
        return s
    return s.rank(method='average', pct=True) - 0.5


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    zero = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return zero

    px = history.get('adjusted_close')
    if not isinstance(px, pd.DataFrame):
        px = history.get('close')
    if not isinstance(px, pd.DataFrame) or px.shape[0] < 260:
        return zero
    if not px.index.is_monotonic_increasing:
        px = px.sort_index()

    cols = [c for c in tickers if c in px.columns]
    if len(cols) < 30:
        return zero

    n = px.shape[0]
    period = 21
    reb = ((n - 1) // period) * period
    if reb < 260:
        return zero

    start = max(0, reb - 340)
    sub = px.iloc[start:reb + 1][cols].astype(float)
    sub = sub.replace([np.inf, -np.inf], np.nan)
    m = sub.shape[0]
    if m < 260:
        return zero

    ret = sub / sub.shift(1) - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)

    # ---- market-state classification (point-in-time at reb) ----
    mkt_ret = ret.mean(axis=1)
    roll60 = mkt_ret.rolling(60, min_periods=40).std()
    tail60 = roll60.tail(252)
    vol60 = float(mkt_ret.tail(60).std())
    med60 = float(tail60.median()) if tail60.notna().sum() > 0 else vol60
    mkt_lvl = (1.0 + mkt_ret.fillna(0.0)).cumprod()
    trend200 = float(mkt_lvl.iloc[-1] / mkt_lvl.iloc[-201] - 1.0) if m >= 201 else 0.0

    calm = bool(np.isfinite(vol60) and np.isfinite(med60) and (vol60 <= med60))
    trending = bool(trend200 > 0.0)
    continuation = calm and trending

    # ---- validity / eligibility / liquidity ----
    last = sub.iloc[-1]
    valid = last.notna() & (last > 0)
    valid = valid & (sub.notna().sum(axis=0) >= 150)

    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame) and elig.shape[0] > 0:
        try:
            e = elig.reindex(index=sub.index, columns=cols)
            if not e.index.is_monotonic_increasing:
                e = e.sort_index()
            e_last = e.iloc[-1]
            if e_last.notna().sum() >= 10:
                valid = valid & e_last.fillna(False).astype(bool)
        except Exception:
            pass

    vdf = history.get('volume')
    if isinstance(vdf, pd.DataFrame) and vdf.shape[0] > 0:
        try:
            vsub = vdf.reindex(index=sub.index, columns=cols).astype(float)
            vsub = vsub.replace([np.inf, -np.inf], np.nan)
            dv = (vsub * sub).tail(60).median(axis=0)
            dv = dv.replace([np.inf, -np.inf], np.nan)
            if dv.notna().sum() >= 30:
                thr = dv.quantile(0.15)
                valid = valid & dv.notna() & (dv > 0) & (dv >= thr)
        except Exception:
            pass

    # ---- signals at reb ----
    p_now = sub.iloc[-1]
    p_6 = sub.iloc[-6]
    p_11 = sub.iloc[-11]
    p_126 = sub.iloc[-126]
    vol20 = ret.tail(20).std()

    r120 = (p_6 / p_126 - 1.0).replace([np.inf, -np.inf], np.nan)
    rev10 = (p_now / p_11 - 1.0).replace([np.inf, -np.inf], np.nan)
    vol20 = vol20.replace([np.inf, -np.inf], np.nan)

    valid = valid & r120.notna() & rev10.notna() & vol20.notna() & (vol20 > 0)
    valid = valid.fillna(False)
    vt = valid[valid].index
    if len(vt) < 20:
        return zero

    v20 = vol20.reindex(vt)
    med_v = v20.median()
    if not np.isfinite(med_v) or med_v <= 0:
        v20 = pd.Series(1.0, index=vt)
    else:
        v20 = v20.fillna(med_v)

    z_mom = _rank_pct(r120.reindex(vt))
    z_rev = _rank_pct(rev10.reindex(vt))
    z_lv = _rank_pct(-v20.reindex(vt))

    if continuation:
        score = 1.0 * z_mom + 0.3 * z_lv
    else:
        score = -1.0 * z_rev + 0.3 * z_lv
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 20:
        return zero

    n_side = min(50, len(score) // 2)
    if n_side < 5:
        return zero
    order = score.sort_values(ascending=False)
    longs = order.index[:n_side]
    shorts = order.index[-n_side:]

    wmag = min(0.02, 0.5 / float(n_side))
    selected = list(longs) + list(shorts)
    w = pd.Series(0.0, index=selected, dtype=float)
    w.loc[longs] = wmag
    w.loc[shorts] = -wmag

    # ---- beta neutralization vs equal-weight market ----
    ret_hist = ret[vt].tail(126)
    if ret_hist.shape[0] >= 40:
        mkt = ret_hist.mean(axis=1)
        mkt_d = mkt - mkt.mean()
        var = float((mkt_d * mkt_d).sum())
        if var > 1e-18:
            rh = ret_hist.fillna(0.0)
            beta = rh.mul(mkt_d, axis=0).sum() / var
            b = beta.reindex(selected).fillna(1.0)
            b = b - b.mean()
            denom = float((b * b).sum())
            if denom > 1e-18:
                c = float((w * b).sum()) / denom
                w = w - c * b

    # ---- map to full universe and enforce constraints ----
    weights = pd.Series(0.0, index=tickers, dtype=float)
    weights.loc[w.index] = w.values
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _enforce(wt):
        wt = wt.clip(-0.02, 0.02)
        g = float(wt.abs().sum())
        if g > 1.0:
            wt = wt * (1.0 / g)
        net = float(wt.sum())
        if abs(net) > 1e-9:
            nz = (wt != 0)
            cnt = int(nz.sum())
            if cnt > 0:
                wt.loc[nz] = wt.loc[nz] - net / cnt
        return wt

    weights = _enforce(weights)
    weights = _enforce(weights)

    net = float(weights.sum())
    if abs(net) > 0.01:
        nz = (weights != 0)
        cnt = int(nz.sum())
        if cnt > 0:
            weights.loc[nz] = weights.loc[nz] - net / cnt
        weights = weights.clip(-0.02, 0.02)
        g = float(weights.abs().sum())
        if g > 1.0:
            weights = weights * (1.0 / g)

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights.reindex(tickers).fillna(0.0)