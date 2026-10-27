import numpy as np
import pandas as pd


def _get_df(history, name):
    x = history.get(name)
    if isinstance(x, pd.DataFrame):
        return x
    return None


def generate_weights(history, universe):
    cols = pd.Index(pd.unique(pd.Index(universe).astype(str)))
    if len(cols) == 0:
        return pd.Series(dtype=float)
    zeros = pd.Series(0.0, index=cols, dtype=float)

    ac = _get_df(history, 'adjusted_close')
    if ac is None:
        ac = _get_df(history, 'close')
    if ac is None or ac.empty:
        return zeros

    avail = [c for c in cols if c in ac.columns]
    if len(avail) < 30:
        return zeros

    ac = ac.reindex(columns=avail).sort_index()
    if len(ac.index) < 40:
        return zeros
    ac = ac.tail(90)
    dates = ac.index
    n = len(dates)

    cdf = _get_df(history, 'close')
    if cdf is None:
        close = ac.copy()
    else:
        close = cdf.reindex(columns=avail).reindex(index=dates)
        close = close.where(close.notna(), ac)

    vdf = _get_df(history, 'volume')
    adv = None
    if vdf is not None:
        vv = vdf.reindex(columns=avail).reindex(index=dates)
        dv = vv.mul(close, axis=0)
        adv = dv.rolling(20, min_periods=10).median()

    edf = _get_df(history, 'eligible')
    if edf is not None:
        edf = edf.reindex(columns=avail).reindex(index=dates).ffill()
    else:
        edf = pd.DataFrame(True, index=dates, columns=avail)

    rets = ac.pct_change()

    def signal(i):
        """Risk-adjusted 5-day reversal score at row i; None if unusable."""
        if i is None or i < 22:
            return None
        px = ac.iloc[i]
        p5 = ac.iloc[i - 5]
        pcl = close.iloc[i]
        v = rets.iloc[i - 19:i + 1].std(ddof=0)
        r5 = px / p5 - 1.0
        try:
            e = edf.iloc[i]
        except Exception:
            e = pd.Series(True, index=avail)
        e = e.reindex(avail).fillna(False).astype(bool)
        if adv is not None:
            a = adv.iloc[i].reindex(avail)
        else:
            a = pd.Series(np.nan, index=avail, dtype=float)

        ok = px.notna() & p5.notna() & pcl.notna() & v.notna() & r5.notna() & e
        ok = ok & np.isfinite(px) & np.isfinite(r5) & np.isfinite(v) & np.isfinite(pcl)
        ok = ok & (pcl > 5.0) & (v > 0.004) & (v < 0.12) & (r5.abs() < 0.60)
        if a.notna().any():
            ok = ok & (a.fillna(0.0) > 3e6)

        if int(ok.sum()) < 60:
            ok2 = px.notna() & p5.notna() & pcl.notna() & v.notna() & r5.notna() & e
            ok2 = ok2 & np.isfinite(px) & np.isfinite(r5) & np.isfinite(v) & np.isfinite(pcl)
            ok2 = ok2 & (pcl > 3.0) & (v > 0.003) & (v < 0.15) & (r5.abs() < 0.80)
            if a.notna().any():
                ok2 = ok2 & (a.fillna(0.0) > 1e6)
            if int(ok2.sum()) > int(ok.sum()):
                ok = ok2

        cand = [c for c in avail if bool(ok.get(c, False))]
        if len(cand) < 8:
            return None
        sc = (-r5.loc[cand] / v.loc[cand]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(sc) < 8:
            return None
        return sc

    wd = np.array([d.weekday() for d in dates])
    fri = np.where(wd == 4)[0]
    if len(fri) > 0:
        i_cur = int(fri[-1])
    else:
        i_cur = n - 1

    sc_cur = signal(i_cur)
    if sc_cur is None:
        return zeros

    ordered = sc_cur.sort_values(ascending=False)
    n_cand = len(ordered)
    n_side = min(25, n_cand // 2)
    if n_side < 4:
        return zeros
    pct = sc_cur.rank(method='average', pct=True)

    prev_fri = fri[fri < i_cur]
    i_prev = int(prev_fri[-1]) if len(prev_fri) > 0 else None
    prev_longs = []
    prev_shorts = []
    if i_prev is not None:
        sc_prev = signal(i_prev)
        if sc_prev is not None and len(sc_prev) >= 8:
            so = sc_prev.sort_values(ascending=False)
            k = min(n_side, max(1, len(so) // 2))
            prev_longs = list(so.index[:k])
            prev_shorts = list(so.index[-k:])

    keep_hi = 0.65
    keep_lo = 0.35

    longs = []
    shorts = []
    for c in prev_longs:
        p = pct.get(c, None)
        if p is not None and np.isfinite(p) and float(p) >= keep_hi:
            if c not in longs:
                longs.append(c)
    for c in prev_shorts:
        p = pct.get(c, None)
        if p is not None and np.isfinite(p) and float(p) <= keep_lo:
            if c not in shorts:
                shorts.append(c)

    held = set(longs) | set(shorts)
    longs = longs[:n_side]
    shorts = shorts[:n_side]

    for c in ordered.index:
        if len(longs) >= n_side:
            break
        if c in held:
            continue
        p = float(pct.get(c, 0.0))
        if p < 0.5:
            break
        longs.append(c)
        held.add(c)

    for c in reversed(list(ordered.index)):
        if len(shorts) >= n_side:
            break
        if c in held:
            continue
        p = float(pct.get(c, 0.0))
        if p > 0.5:
            break
        shorts.append(c)
        held.add(c)

    longs = [c for c in longs if c in zeros.index][:n_side]
    shorts = [c for c in shorts if c in zeros.index and c not in set(longs)][:n_side]
    nL = len(longs)
    nS = len(shorts)
    if nL < 1 or nS < 1:
        return zeros

    side_cap = 0.5
    w = min(0.02, side_cap / float(max(nL, nS)))
    if w <= 0.0:
        return zeros

    weights = zeros.copy()
    for c in longs:
        weights.loc[c] = w
    for c in shorts:
        weights.loc[c] = -w

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
    gross = float(weights.abs().sum())
    if gross > 1.0:
        weights = weights * (1.0 / gross)
    return weights.fillna(0.0)