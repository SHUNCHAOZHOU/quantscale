import numpy as np
import pandas as pd

def generate_weights(history, universe):
    u = pd.Index(universe)
    z = pd.Series(0.0, index=u, dtype=float)
    if len(u) == 0:
        return z
    ac = history.get('adjusted_close')
    cl = history.get('close')
    vo = history.get('volume')
    el = history.get('eligible')
    if ac is None or cl is None or vo is None:
        return z
    ac = ac.reindex(columns=u)
    cl = cl.reindex(columns=u)
    vo = vo.reindex(columns=u)
    idx = ac.index
    if len(idx) < 130:
        return z
    last = idx[-1]
    per = idx.to_period('M')
    pm = per < last.to_period('M')
    if pm.any():
        asof = idx[pm][-1]
    else:
        asof = last
    ac = ac.loc[:asof]
    cl = cl.loc[:asof]
    vo = vo.loc[:asof]
    if len(ac) > 320:
        ac = ac.iloc[-320:]
        cl = cl.iloc[-320:]
        vo = vo.iloc[-320:]
    if len(ac) < 130:
        return z
    ac = ac.ffill()
    cl = cl.ffill()
    ret = ac.pct_change(fill_method=None)
    px = ac.iloc[-1]
    dv = (cl * vo).rolling(20, min_periods=5).median().iloc[-1]
    good = px.notna() & (px > 2.0) & dv.notna() & (dv > 5e5)
    if el is not None:
        e = el.reindex(columns=u)
        if len(e) > 0:
            e = e.loc[:asof]
            if len(e) > 0:
                good = good & e.iloc[-1].fillna(False).astype(bool)
    n = len(ac)
    p0 = ac.iloc[-1]
    p1 = ac.iloc[-22] if n >= 22 else ac.iloc[0]
    p6 = ac.iloc[-127] if n >= 127 else ac.iloc[0]
    if n >= 252:
        p12 = ac.iloc[-252]
    else:
        p12 = p6
    mom6 = p1 / p6 - 1.0
    mom12 = p1 / p12 - 1.0
    rev = p0 / p1 - 1.0
    vola = ret.iloc[-126:].std()
    good = good & np.isfinite(mom6) & np.isfinite(mom12) & np.isfinite(vola) & np.isfinite(rev)
    good = good & (vola > 0.0) & (vola < 0.15)
    cand = u[good.to_numpy()]
    if len(cand) < 20:
        return z
    m6 = mom6[cand].rank(pct=True)
    m12 = mom12[cand].rank(pct=True)
    rv = rev[cand].rank(pct=True)
    lv = vola[cand].rank(pct=True)
    score = 0.55 * m6 + 0.45 * m12 - 0.20 * rv - 0.25 * lv
    ns = int(min(50, max(10, len(cand) // 10)))
    ns = min(ns, len(cand) // 2)
    if ns < 1:
        return z
    order = score.sort_values(ascending=False)
    longs = order.index[:ns]
    shorts = order.index[-ns:]
    q = 0.49 / float(ns)
    if q > 0.02:
        q = 0.02
    w = z.copy()
    w.loc[longs] = q
    w.loc[shorts] = -q
    return w.fillna(0.0).astype(float)