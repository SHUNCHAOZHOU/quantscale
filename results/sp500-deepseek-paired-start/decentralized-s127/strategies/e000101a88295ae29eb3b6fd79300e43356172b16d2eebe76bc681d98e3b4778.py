import numpy as np
import pandas as pd


def _rank_pct(s):
    return s.rank(method="average", pct=True, na_option="keep")


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    try:
        adj = history.get("adjusted_close")
        if adj is None or adj.empty:
            adj = history.get("close")
    except Exception:
        adj = None
    if adj is None or adj.empty:
        return zero

    cols = [c for c in universe if c in adj.columns]
    if len(cols) == 0:
        return zero

    adj = adj[cols]
    idx = adj.index
    if len(idx) < 30:
        return zero

    last_date = idx[-1]
    try:
        nxt = last_date + pd.offsets.BDay(1)
        at_month_end = (nxt.year != last_date.year) or (nxt.month != last_date.month)
    except Exception:
        at_month_end = False

    if at_month_end:
        asof = last_date
    else:
        try:
            months = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max()
            asof = months.iloc[-2] if len(months) >= 2 else idx[0]
        except Exception:
            asof = idx[-1]

    pos = int(idx.searchsorted(asof, side="right"))
    if pos <= 0:
        return zero

    start = max(0, pos - 330)
    px = adj.iloc[start:pos]
    if len(px) < 30:
        return zero

    ret = px / px.shift(1) - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)

    mom = px.shift(21) / px.shift(252) - 1.0
    rev = -(px / px.shift(21) - 1.0)
    vol = ret.tail(63).std()

    last = px.iloc[-1].reindex(cols)
    m = mom.iloc[-1].reindex(cols)
    r = rev.iloc[-1].reindex(cols)
    v = vol.iloc[-1].reindex(cols)

    score = 0.40 * _rank_pct(m) + 0.40 * _rank_pct(r) - 0.20 * _rank_pct(v)

    valid = last.notna() & (last > 0) & score.notna() & m.notna() & r.notna()

    try:
        elig = history.get("eligible")
    except Exception:
        elig = None
    if elig is not None:
        ecols = [c for c in cols if c in elig.columns]
        if len(ecols) > 0:
            e = elig[ecols]
            epos = int(e.index.searchsorted(asof, side="right"))
            if epos > 0:
                ev = e.iloc[epos - 1].reindex(cols).fillna(True).astype(bool)
                valid = valid & ev

    try:
        volume = history.get("volume")
    except Exception:
        volume = None
    if volume is not None and not volume.empty:
        vcols = [c for c in cols if c in volume.columns]
        if len(vcols) > 0:
            vv = volume[vcols]
            vpos = int(vv.index.searchsorted(asof, side="right"))
            if vpos > 0:
                vs = vv.iloc[max(0, vpos - 21):vpos]
                pxs = px[vcols].tail(21)
                adv = (vs.tail(21).reindex(index=pxs.index) * pxs).mean()
                liq_ok = adv > 0.0
                liq_ok = liq_ok.reindex(cols).fillna(True).astype(bool)
                valid = valid & liq_ok

    cand = score[valid].dropna()

    if len(cand) < 2:
        alt = 0.60 * _rank_pct(r) - 0.40 * _rank_pct(v)
        valid_alt = last.notna() & (last > 0) & alt.notna() & r.notna()
        if elig is not None and len(ecols) > 0 and epos > 0:
            valid_alt = valid_alt & ev
        cand = alt[valid_alt].dropna()

    if len(cand) < 2:
        return zero

    n = min(50, len(cand) // 2)
    if n < 1:
        return zero

    order = cand.sort_values(ascending=False)
    longs = order.index[:n]
    shorts = order.index[-n:]

    w = min(0.02, 0.5 / float(n))
    weights = pd.Series(0.0, index=universe, dtype=float)

    for t in longs:
        weights.loc[t] = weights.loc[t] + w
    for t in shorts:
        weights.loc[t] = weights.loc[t] - w

    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.02, 0.02)
    return weights