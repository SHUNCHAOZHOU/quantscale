import numpy as np
import pandas as pd

def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=tickers, dtype=float)

    ac = history.get('adjusted_close')
    if ac is None or not isinstance(ac, pd.DataFrame):
        return zero
    ac = ac.sort_index()
    n = len(ac.index)
    if n < 260:
        return zero

    reb = ((n - 1) // 21) * 21
    if reb < 220:
        return zero

    cols = [c for c in tickers if c in ac.columns]
    if len(cols) < 20:
        return zero

    px = ac.iloc[:reb + 1][cols].astype(float).copy()
    px = px.replace([np.inf, -np.inf], np.nan)
    if px.shape[0] < 180:
        return zero

    last = px.iloc[-1]
    ret = px.pct_change()
    ret = ret.replace([np.inf, -np.inf], np.nan)

    r = ret.tail(126).copy()
    cnt = r.notna().sum(axis=0)
    r_fill = r.fillna(0.0)

    mkt = ret.mean(axis=1, skipna=True).tail(126)
    m = mkt.reindex(r.index).fillna(0.0)
    r_dm = r_fill.sub(r_fill.mean(axis=0), axis=1)
    m_dm = m - m.mean()
    var_m = float((m_dm ** 2).sum())
    if var_m <= 0.0:
        return zero

    beta = r_dm.mul(m_dm, axis=0).sum(axis=0) / var_m
    vol = r.std(axis=0, ddof=1) * np.sqrt(252.0)
    vol = vol.replace([np.inf, -np.inf], np.nan)
    maxret = r.max(axis=0)

    valid = (
        last.notna()
        & (last > 0)
        & (cnt >= 90)
        & beta.notna()
        & vol.notna()
        & (vol > 0)
        & maxret.notna()
    )

    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame) and len(elig.index) > 0:
        edf = elig.sort_index()
        edf2 = edf.reindex(ac.index[:reb + 1])
        if len(edf2) > 0:
            e_last = edf2.iloc[-1]
        else:
            e_last = edf.iloc[-1]
        e = e_last.reindex(cols).fillna(False).astype(bool)
        valid = valid & e

    vdf = history.get('volume')
    if isinstance(vdf, pd.DataFrame) and len(vdf.index) > 0:
        v = vdf.sort_index()
        vsub = v.reindex(ac.index[:reb + 1])[cols].astype(float)
        dollar = vsub.mul(px, axis=1)
        liq = dollar.tail(60).median(axis=0)
        liq = liq.replace([np.inf, -np.inf], np.nan)
        valid = valid & liq.notna() & (liq > 0)
        if liq.notna().sum() >= 30:
            thresh = liq.quantile(0.20)
            valid = valid & (liq >= thresh)

    cand = [c for c in cols if bool(valid.get(c, False))]
    if len(cand) < 20:
        return zero

    b = beta.loc[cand]
    vv = vol.loc[cand]
    mr = maxret.loc[cand]

    def rp(s):
        return s.rank(method='average', pct=True)

    score = -(0.45 * rp(b) + 0.35 * rp(vv) + 0.20 * rp(mr))
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 10:
        return zero

    k = min(50, len(score) // 2)
    if k < 5:
        return zero

    order = score.sort_values(ascending=False)
    longs = order.index[:k]
    shorts = order.index[-k:]
    wmag = min(0.02, 0.5 / float(k))

    w = zero.copy()
    w.loc[longs] = wmag
    w.loc[shorts] = -wmag

    gross = float(w.abs().sum())
    if gross > 1.0:
        w = w * (1.0 / gross)
    w = w.clip(-0.02, 0.02)

    return w.replace([np.inf, -np.inf], 0.0).fillna(0.0).reindex(tickers).fillna(0.0)