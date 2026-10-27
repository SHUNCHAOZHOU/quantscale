import numpy as np
import pandas as pd


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=tickers, dtype=float)

    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None or not isinstance(ac, pd.DataFrame) or len(ac.index) == 0:
        return zero
    ac = ac.sort_index()

    n = len(ac.index)
    if n < 260:
        return zero

    reb = ((n - 1) // 21) * 21
    if reb < 252:
        return zero

    cols = [c for c in tickers if c in ac.columns]
    if len(cols) < 20:
        return zero

    start = max(0, reb - 300)
    px = ac.iloc[start:reb + 1][cols].astype(float).copy()
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
    mkt = mkt.reindex(r.index).fillna(0.0)
    r_dm = r_fill.sub(r_fill.mean(axis=0), axis=1)
    m_dm = mkt - mkt.mean()
    var_m = float((m_dm ** 2).sum())
    if var_m <= 0.0:
        return zero

    beta = r_dm.mul(m_dm, axis=0).sum(axis=0) / var_m
    vol = r.std(axis=0, ddof=1) * np.sqrt(252.0)
    vol = vol.replace([np.inf, -np.inf], np.nan)
    maxret = r.max(axis=0)

    if px.shape[0] > 22:
        ret21 = px.iloc[-1] / px.iloc[-22] - 1.0
    else:
        ret21 = pd.Series(np.nan, index=cols)
    ret21 = ret21.replace([np.inf, -np.inf], np.nan)

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
        esub = edf.reindex(index=px.index, columns=cols)
        if len(esub) > 0:
            e_last = esub.iloc[-1].fillna(False).astype(bool)
            valid = valid & e_last

    vdf = history.get('volume')
    if isinstance(vdf, pd.DataFrame) and len(vdf.index) > 0:
        v = vdf.sort_index()
        vsub = v.reindex(index=px.index, columns=cols).astype(float)
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
    r21 = ret21.reindex(cand).fillna(0.0)

    def rp(s):
        return s.replace([np.inf, -np.inf], np.nan).rank(method='average', pct=True)

    score = (
        0.40 * rp(b)
        + 0.30 * rp(vv)
        + 0.15 * rp(mr)
        + 0.15 * rp(r21)
    )
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