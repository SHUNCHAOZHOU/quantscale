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
    if len(ac.index) < 100:
        return zeros

    ac_t = ac.tail(320)
    dates = ac_t.index
    n = len(dates)
    if n < 80:
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
        ok = ok & np.isfinite(px) & np.isfinite(pcl) & np.isfinite(v) & np.isfinite(r1) & np.isfinite(r5)
        ok = ok & (pcl > 5.0) & (v > 0.004) & (v < 0.10) & (r5.abs() < 0.60) & (r1.abs() < 0.25)
        if adv20 is not None:
            a = adv20.iloc[i].reindex(avail)
            ok = ok & a.notna() & (a > 3e6)

        if int(ok.sum()) < 45:
            ok = px.notna() & p1.notna() & p5.notna() & pcl.notna() & v.notna() & r1.notna() & r5.notna() & e
            ok = ok & np.isfinite(px) & np.isfinite(pcl) & np.isfinite(v) & np.isfinite(r1) & np.isfinite(r5)
            ok = ok & (pcl > 3.0) & (v > 0.003) & (v < 0.14) & (r5.abs() < 0.80) & (r1.abs() < 0.40)
            if adv20 is not None:
                a = adv20.iloc[i].reindex(avail)
                ok = ok & a.notna() & (a > 1e6)

        cand = list(ok[ok].index)
        if len(cand) < 12:
            return None

        r1c = r1.loc[cand].astype(float)
        r5c = r5.loc[cand].astype(float)
        vc = v.loc[cand].astype(float).replace(0.0, np.nan)
        rev = -(0.65 * r5c + 0.35 * r1c) / vc

        if vratio is not None:
            vr = vratio.iloc[i].reindex(cand).astype(float)
            vr = vr.clip(lower=0.5, upper=3.0).fillna(1.0)
            rev = rev * (0.8 + 0.2 * vr)

        rev = rev.replace([np.inf, -np.inf], np.nan).dropna()
        if len(rev) < 12:
            return None
        return rev.rank(method='average', pct=True)

    wd = np.array([d.weekday() for d in dates])
    fri = np.where(wd == 4)[0]
    if len(fri) == 0:
        fri = np.array([n - 1])

    n_side = 25
    keep_hi = 0.40
    keep_lo = 0.60
    entry_hi = 0.85
    entry_lo = 0.15
    max_new = 8

    def update_side(prev_list, pct, is_long):
        if pct is None:
            return list(prev_list)
        valid = pct.dropna()
        if len(valid) == 0:
            return list(prev_list)

        if len(prev_list) == 0:
            if is_long:
                ordered = valid.sort_values(ascending=False).index
                sel = [c for c in ordered if float(valid.get(c, 0.0)) >= 0.5][:n_side]
                if len(sel) < n_side:
                    sel = list(ordered[:n_side])
            else:
                ordered = valid.sort_values(ascending=True).index
                sel = [c for c in ordered if float(valid.get(c, 1.0)) <= 0.5][:n_side]
                if len(sel) < n_side:
                    sel = list(ordered[:n_side])
            return sel

        prev = list(prev_list)
        if is_long:
            retained = [c for c in prev if c in valid.index and float(valid.get(c, -1.0)) >= keep_hi]
            retained = sorted(retained, key=lambda c: -float(valid.get(c, 0.0)))
            new_cand = [c for c in valid.sort_values(ascending=False).index if c not in prev and float(valid.get(c, 0.0)) >= entry_hi]
            stale = [c for c in prev if c not in retained]
            stale = sorted(stale, key=lambda c: -float(valid.get(c, -1.0)) if c in valid.index else 1.0)

            selected = retained[:n_side]
            for c in new_cand:
                if len(selected) >= n_side:
                    break
                if len([x for x in selected if x not in retained]) >= max_new:
                    break
                if c not in selected:
                    selected.append(c)
            for c in stale:
                if len(selected) >= n_side:
                    break
                if c not in selected:
                    selected.append(c)
            if len(selected) < n_side:
                for c in valid.sort_values(ascending=False).index:
                    if len(selected) >= n_side:
                        break
                    if c in selected:
                        continue
                    if float(valid.get(c, 0.0)) >= 0.5:
                        selected.append(c)
            return selected[:n_side]

        retained = [c for c in prev if c in valid.index and float(valid.get(c, 2.0)) <= keep_lo]
        retained = sorted(retained, key=lambda c: float(valid.get(c, 1.0)))
        new_cand = [c for c in valid.sort_values(ascending=True).index if c not in prev and float(valid.get(c, 1.0)) <= entry_lo]
        stale = [c for c in prev if c not in retained]
        stale = sorted(stale, key=lambda c: float(valid.get(c, 2.0)) if c in valid.index else 2.0)

        selected = retained[:n_side]
        for c in new_cand:
            if len(selected) >= n_side:
                break
            if len([x for x in selected if x not in retained]) >= max_new:
                break
            if c not in selected:
                selected.append(c)
        for c in stale:
            if len(selected) >= n_side:
                break
            if c not in selected:
                selected.append(c)
        if len(selected) < n_side:
            for c in valid.sort_values(ascending=True).index:
                if len(selected) >= n_side:
                    break
                if c in selected:
                    continue
                if float(valid.get(c, 1.0)) <= 0.5:
                    selected.append(c)
        return selected[:n_side]

    prev_longs = []
    prev_shorts = []
    last_i = None

    for i in fri:
        i = int(i)
        if i < 25:
            continue
        pct = signal(i)
        if pct is None:
            continue

        new_longs = update_side(prev_longs, pct, True)
        new_shorts = update_side(prev_shorts, pct, False)

        overlap = set(new_longs) & set(new_shorts)
        if overlap:
            for c in list(overlap):
                if c in pct.index and float(pct.get(c, 0.5)) >= 0.5:
                    new_shorts.remove(c)
                else:
                    new_longs.remove(c)

        prev_longs = new_longs
        prev_shorts = new_shorts
        last_i = i

    if last_i is None:
        return zeros

    nL = len(prev_longs)
    nS = len(prev_shorts)
    if nL < 5 or nS < 5:
        return zeros

    side_gross = min(0.5, 0.02 * min(nL, nS))
    if side_gross <= 0.0:
        return zeros

    wl = side_gross / float(nL)
    ws = side_gross / float(nS)

    weights = zeros.copy()
    long_set = set(prev_longs)
    for c in prev_longs:
        if c in weights.index:
            weights.loc[c] = wl
    for c in prev_shorts:
        if c in weights.index and c not in long_set:
            weights.loc[c] = -ws

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
    gross = float(weights.abs().sum())
    if gross > 1.0:
        weights = weights * (1.0 / gross)
    return weights.fillna(0.0)