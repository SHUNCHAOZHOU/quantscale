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
    if len(avail) < 20:
        return zeros

    ac = ac.reindex(columns=avail).sort_index()
    if len(ac.index) < 70:
        return zeros

    ac_t = ac.tail(150)
    dates = ac_t.index
    n = len(dates)
    if n < 70:
        return zeros

    cdf = _get_df(history, 'close')
    if cdf is None:
        close_t = ac_t.copy()
    else:
        close_t = cdf.reindex(columns=avail).reindex(index=dates)
        close_t = close_t.where(close_t.notna(), ac_t)

    vdf = _get_df(history, 'volume')
    if vdf is None:
        vol_t = pd.DataFrame(np.nan, index=dates, columns=avail)
        has_vol = False
    else:
        vol_t = vdf.reindex(columns=avail).reindex(index=dates)
        has_vol = bool(vol_t.notna().any().any())

    if has_vol:
        dollar = close_t.mul(vol_t, axis=0)
        adv20 = dollar.rolling(20, min_periods=10).median()
        v5 = vol_t.rolling(5, min_periods=3).mean()
        v60 = vol_t.rolling(60, min_periods=30).mean()
        vratio = (v5 / v60).replace([np.inf, -np.inf], np.nan)
    else:
        adv20 = None
        vratio = None

    rets = ac_t.pct_change()
    vol20 = rets.rolling(20, min_periods=10).std(ddof=0)

    edf = _get_df(history, 'eligible')
    if edf is not None:
        elig_t = edf.reindex(columns=avail).reindex(index=dates).ffill()
    else:
        elig_t = pd.DataFrame(True, index=dates, columns=avail)

    wd = np.array([d.weekday() for d in dates])
    fri = np.where(wd == 4)[0]
    if len(fri) > 0:
        i_cur = int(fri[-1])
    else:
        i_cur = n - 1
    if i_cur < 25:
        return zeros

    def signal(i):
        if i is None or i < 25:
            return None
        px = ac_t.iloc[i]
        p1 = ac_t.iloc[i - 1]
        p5 = ac_t.iloc[i - 5]
        pcl = close_t.iloc[i]
        v = vol20.iloc[i]
        r1 = px / p1 - 1.0
        r5 = px / p5 - 1.0

        try:
            e = elig_t.iloc[i]
        except Exception:
            e = pd.Series(True, index=avail, dtype=bool)
        e = e.reindex(avail).fillna(False).astype(bool)
        if int(e.sum()) < 10:
            e = pd.Series(True, index=avail, dtype=bool)

        ok = px.notna() & p1.notna() & p5.notna() & pcl.notna() & v.notna() & r1.notna() & r5.notna() & e
        ok = ok & np.isfinite(px) & np.isfinite(pcl) & np.isfinite(v)
        ok = ok & np.isfinite(r1) & np.isfinite(r5)
        ok = ok & (pcl > 5.0) & (v > 0.004) & (v < 0.10)
        ok = ok & (r5.abs() < 0.60) & (r1.abs() < 0.25)
        if adv20 is not None:
            a = adv20.iloc[i].reindex(avail)
            ok = ok & a.notna() & (a > 3e6)

        if int(ok.sum()) < 45:
            ok = px.notna() & p1.notna() & p5.notna() & pcl.notna() & v.notna() & r1.notna() & r5.notna() & e
            ok = ok & np.isfinite(px) & np.isfinite(pcl) & np.isfinite(v)
            ok = ok & np.isfinite(r1) & np.isfinite(r5)
            ok = ok & (pcl > 3.0) & (v > 0.003) & (v < 0.14)
            ok = ok & (r5.abs() < 0.80) & (r1.abs() < 0.40)
            if adv20 is not None:
                a = adv20.iloc[i].reindex(avail)
                ok = ok & a.notna() & (a > 1e6)

        cand = list(ok[ok].index)
        if len(cand) < 12:
            return None

        r1c = r1.loc[cand].astype(float)
        r5c = r5.loc[cand].astype(float)
        vc = v.loc[cand].astype(float)
        rev = -(0.65 * r5c + 0.35 * r1c) / vc

        if vratio is not None:
            vr = vratio.iloc[i].reindex(cand).astype(float)
            vr = vr.clip(lower=0.25, upper=4.0).fillna(1.0)
            rev = rev * (0.75 + 0.25 * vr)

        rev = rev.replace([np.inf, -np.inf], np.nan).dropna()
        if len(rev) < 12:
            return None
        return rev.rank(method='average', pct=True)

    sc = signal(i_cur)
    if sc is None:
        return zeros

    n_side = min(25, len(sc) // 2)
    if n_side < 5:
        return zeros

    prev_fri = fri[fri < i_cur]
    prev_longs = []
    prev_shorts = []
    if len(prev_fri) > 0:
        i_prev = int(prev_fri[-1])
        sc_prev = signal(i_prev)
        if sc_prev is not None and len(sc_prev) >= 2 * n_side:
            sp = sc_prev.sort_values(ascending=False)
            prev_longs = list(sp.index[:n_side])
            prev_shorts = list(sp.index[-n_side:])

    pct = sc.rank(method='average', pct=True)
    longs = []
    shorts = []

    for c in prev_longs:
        p = pct.get(c, np.nan)
        if np.isfinite(p) and float(p) >= 0.42:
            if c not in longs:
                longs.append(c)

    for c in prev_shorts:
        p = pct.get(c, np.nan)
        if np.isfinite(p) and float(p) <= 0.58:
            if c not in shorts:
                shorts.append(c)

    held = set(longs) | set(shorts)

    desc = sc.sort_values(ascending=False).index
    for c in desc:
        if len(longs) >= n_side:
            break
        if c in held:
            continue
        p = pct.get(c, np.nan)
        if not np.isfinite(p):
            continue
        if float(p) < 0.5:
            break
        longs.append(c)
        held.add(c)

    asc = sc.sort_values(ascending=True).index
    for c in asc:
        if len(shorts) >= n_side:
            break
        if c in held:
            continue
        p = pct.get(c, np.nan)
        if not np.isfinite(p):
            continue
        if float(p) > 0.5:
            break
        shorts.append(c)
        held.add(c)

    longs = [c for c in longs if c in zeros.index][:n_side]
    shorts = [c for c in shorts if c in zeros.index and c not in set(longs)][:n_side]
    nL = len(longs)
    nS = len(shorts)
    if nL < 5 or nS < 5:
        return zeros

    side_gross = min(0.5, 0.02 * min(nL, nS))
    if side_gross <= 0.0:
        return zeros

    wl = side_gross / float(nL)
    ws = side_gross / float(nS)

    weights = zeros.copy()
    for c in longs:
        weights.loc[c] = wl
    for c in shorts:
        weights.loc[c] = -ws

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    weights = weights.clip(-0.02, 0.02)
    gross = float(weights.abs().sum())
    if gross > 1.0:
        weights = weights * (1.0 / gross)
    return weights.fillna(0.0)