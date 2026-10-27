import numpy as np
import pandas as pd


def _rank_pct(s):
    return s.rank(method='average', pct=True, na_option='keep')


def _row_asof(df, asof, cols):
    if df is None or df.empty:
        return pd.Series(np.nan, index=cols, dtype=float)
    try:
        pos = int(df.index.searchsorted(asof, side='right'))
    except Exception:
        return pd.Series(np.nan, index=cols, dtype=float)
    if pos <= 0:
        return pd.Series(np.nan, index=cols, dtype=float)
    row = df.iloc[pos - 1]
    if isinstance(row, pd.Series):
        return row.reindex(cols)
    try:
        return pd.Series(row, index=df.columns).reindex(cols)
    except Exception:
        return pd.Series(np.nan, index=cols, dtype=float)


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    adj = None
    try:
        adj = history.get('adjusted_close')
        if adj is None or adj.empty:
            adj = history.get('close')
    except Exception:
        adj = None
    if adj is None or adj.empty:
        return zero

    cols = [c for c in universe if c in adj.columns]
    if len(cols) == 0:
        return zero

    px = adj.reindex(columns=cols).astype(float)
    idx = px.index
    if len(idx) < 260:
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
            asof = months.iloc[-2] if len(months) >= 2 else idx[-1]
        except Exception:
            asof = idx[-1]

    pos = int(idx.searchsorted(asof, side='right'))
    if pos <= 0:
        return zero

    start = max(0, pos - 300)
    hist = px.iloc[start:pos]
    if len(hist) < 260:
        return zero

    ret = hist / hist.shift(1) - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)

    mom = hist.shift(21) / hist.shift(252) - 1.0
    rev = -(hist / hist.shift(21) - 1.0)
    vol = ret.tail(63).std()

    mom = mom.replace([np.inf, -np.inf], np.nan)
    rev = rev.replace([np.inf, -np.inf], np.nan)

    last = hist.iloc[-1]
    m = mom.iloc[-1]
    r = rev.iloc[-1]
    v = vol.iloc[-1]

    score = 0.40 * _rank_pct(m) + 0.40 * _rank_pct(r) - 0.20 * _rank_pct(v)
    valid = last.notna() & (last > 0.0) & score.notna() & m.notna() & r.notna() & v.notna()

    elig = None
    try:
        elig = history.get('eligible')
    except Exception:
        elig = None
    if elig is not None and not elig.empty:
        ecols = [c for c in cols if c in elig.columns]
        if len(ecols) > 0:
            ev = _row_asof(elig[ecols], asof, cols)
            ev = ev.fillna(True).astype(bool)
            valid = valid & ev

    volume = None
    try:
        volume = history.get('volume')
    except Exception:
        volume = None
    if volume is not None and not volume.empty:
        vcols = [c for c in cols if c in volume.columns]
        if len(vcols) > 0:
            vv = volume.reindex(index=hist.index, columns=vcols)
            pp = hist[vcols]
            adv = (vv.tail(21) * pp.tail(21)).mean()
            adv = adv.reindex(cols)
            liq_ok = adv.fillna(0.0) > 1000000.0
            valid = valid & liq_ok

    cand = score[valid].dropna()
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