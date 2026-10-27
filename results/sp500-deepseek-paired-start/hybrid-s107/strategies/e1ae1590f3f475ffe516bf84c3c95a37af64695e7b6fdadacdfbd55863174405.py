import numpy as np
import pandas as pd

def _rank_center(s):
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return s
    return s.rank(method='average', pct=True) - 0.5

def _allocate(raw, total, cap):
    if raw is None or len(raw) == 0:
        return None
    r = raw.replace([np.inf, -np.inf], np.nan).dropna()
    r = r[r > 0]
    if len(r) == 0:
        return None
    s = float(r.sum())
    if not np.isfinite(s) or s <= 0:
        return None
    w = r * (total / s)
    for _ in range(10):
        over = w > (cap + 1e-15)
        if not bool(over.any()):
            break
        w = w.clip(upper=cap)
        free = ~over
        rem = total - float(w[over].sum())
        if rem <= 1e-12 or not bool(free.any()):
            break
        sub = w[free]
        tot = float(sub.sum())
        if tot <= 1e-12:
            break
        w[free] = sub * (rem / tot)
    return w.clip(upper=cap)

def _beta_neutral(w, beta, cap):
    if w is None or len(w) == 0:
        return w
    b = beta.reindex(w.index).replace([np.inf, -np.inf], np.nan)
    if int(b.notna().sum()) >= max(3, len(w) // 2):
        b = b.fillna(float(b.median()))
    else:
        b = pd.Series(1.0, index=w.index)
    b = b - b.mean()
    den = float((b * b).sum())
    if den > 1e-18:
        c = float((w * b).sum()) / den
        w = w - c * b
    w = w.clip(-cap, cap)
    net = float(w.sum())
    if abs(net) > 1e-12:
        active = w[w != 0].index
        if len(active) > 0:
            w.loc[active] = w.loc[active] - net / float(len(active))
    return w.clip(-cap, cap)

def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=tickers, dtype=float)

    px = history.get('adjusted_close')
    if not isinstance(px, pd.DataFrame):
        px = history.get('close')
    if not isinstance(px, pd.DataFrame) or px.shape[0] < 220:
        return zero
    if not px.index.is_monotonic_increasing:
        px = px.sort_index()

    cols = [c for c in tickers if c in px.columns]
    if len(cols) < 30:
        return zero

    n = int(px.shape[0])
    period = 21
    reb = ((n - 1) // period) * period
    if reb < 180:
        return zero

    start = max(0, reb - 260 + 1)
    sub = px.iloc[start:reb + 1][cols].astype(float).replace([np.inf, -np.inf], np.nan)
    if sub.shape[0] < 160:
        return zero

    last = sub.iloc[-1]
    valid = last.notna() & (last > 0) & (sub.notna().sum(axis=0) >= 120)

    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame) and elig.shape[0] > 0:
        e = elig.reindex(columns=cols)
        if not e.index.is_monotonic_increasing:
            e = e.sort_index()
        erow = None
        if len(e.index) == n:
            erow = e.iloc[reb]
        else:
            dt = px.index[reb]
            if dt in e.index:
                erow = e.loc[dt]
        if erow is not None:
            er = erow.replace([np.inf, -np.inf], np.nan)
            if int(er.notna().sum()) >= 20:
                valid = valid & er.fillna(False).astype(bool)

    vdf = history.get('volume')
    volume_sub = None
    if isinstance(vdf, pd.DataFrame) and vdf.shape[0] > 0:
        if not vdf.index.is_monotonic_increasing:
            vdf = vdf.sort_index()
        vcols = [c for c in cols if c in vdf.columns]
        if len(vcols) > 0:
            volume_sub = vdf.reindex(index=sub.index, columns=vcols).astype(float).replace([np.inf, -np.inf], np.nan)
            dollar = volume_sub * sub[vcols]
            liq = dollar.tail(60).median(axis=0).reindex(cols)
            valid = valid & liq.notna() & (liq > 0)
            if int(liq.notna().sum()) >= 30:
                thr = liq.quantile(0.30)
                if np.isfinite(thr):
                    valid = valid & (liq >= thr)

    valid = valid.fillna(False)
    names0 = [c for c in cols if bool(valid.get(c, False))]
    if len(names0) < 16:
        return zero

    rets = sub.pct_change().replace([np.inf, -np.inf], np.nan)
    R = rets.tail(120)
    Rf = R.fillna(0.0)
    Rmat = Rf.values.astype(float)
    mkt = R.mean(axis=1).fillna(0.0)
    mvec = mkt.values.astype(float)
    m_d = mvec - mvec.mean()
    var = float(np.dot(m_d, m_d))
    if var <= 1e-18:
        return zero
    Rc = Rmat - Rmat.mean(axis=0, keepdims=True)
    beta_arr = (Rc * m_d[:, None]).sum(axis=0) / var
    beta_all = pd.Series(beta_arr, index=Rf.columns)
    resid = Rc - np.outer(m_d, beta_arr)
    resid_vol = pd.Series(resid.std(axis=0, ddof=1), index=Rf.columns)

    vol60 = rets.tail(60).std(axis=0, ddof=1)
    roll = rets.rolling(window=20, min_periods=15).std()
    vov = roll.tail(80).std(axis=0, ddof=1)
    max60 = rets.tail(60).max(axis=0)

    valid2 = valid & resid_vol.notna() & (resid_vol > 0) & vol60.notna() & (vol60 > 0) & vov.notna() & max60.notna()
    names = [c for c in cols if bool(valid2.get(c, False))]
    if len(names) < 16:
        return zero

    r_iv = _rank_center((-resid_vol).reindex(names))
    r_vov = _rank_center((-vov).reindex(names))
    r_max = _rank_center((-max60).reindex(names))

    if volume_sub is not None:
        v = volume_sub.reindex(columns=cols)
        v20 = v.tail(20).mean(axis=0)
        v120 = v.tail(120).mean(axis=0)
        volratio = np.log((v20.fillna(0.0) + 1.0) / (v120.fillna(0.0) + 1.0))
        volratio = volratio.replace([np.inf, -np.inf], np.nan)
        r_vt = _rank_center((-volratio).reindex(names))
        score = 0.45 * r_iv + 0.25 * r_vov + 0.20 * r_max + 0.10 * r_vt
    else:
        score = 0.50 * r_iv + 0.30 * r_vov + 0.20 * r_max

    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 16:
        return zero

    k = int(min(50, len(score) // 3))
    if k < 5:
        return zero
    order = score.sort_values(ascending=False)
    longs = order.index[:k]
    shorts = order.index[-k:]

    w = pd.Series(0.0, index=cols, dtype=float)
    for leg_names, sign in ((longs, 1.0), (shorts, -1.0)):
        vv = vol60.reindex(leg_names).replace([np.inf, -np.inf], np.nan).dropna()
        vv = vv[vv > 0]
        if len(vv) == 0:
            continue
        iv = 1.0 / vv.clip(lower=1e-4)
        wl = _allocate(iv, 0.45, 0.02)
        if wl is not None and len(wl) > 0:
            w.loc[wl.index] = w.loc[wl.index] + sign * wl.values

    beta_sel = beta_all.reindex(w.index)
    w = _beta_neutral(w, beta_sel, 0.02)

    gross = float(w.abs().sum())
    if gross > 0.90 and gross > 0:
        w = w * (0.90 / gross)

    net = float(w.sum())
    if abs(net) > 1e-12:
        active = w[w != 0].index
        if len(active) > 0:
            w.loc[active] = w.loc[active] - net / float(len(active))
    w = w.clip(-0.02, 0.02)
    gross = float(w.abs().sum())
    if gross > 1.0 and gross > 0:
        w = w * (1.0 / gross)

    out = pd.Series(0.0, index=tickers, dtype=float)
    w = w.reindex(cols).fillna(0.0).replace([np.inf, -np.inf], 0.0)
    out.loc[w.index] = w.values
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)