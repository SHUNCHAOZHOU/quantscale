import numpy as np
import pandas as pd


def _df(history, name):
    x = history.get(name)
    if isinstance(x, pd.DataFrame):
        return x
    return None


def generate_weights(history, universe):
    cols = pd.Index(pd.unique(pd.Index(universe).astype(str)))
    if len(cols) == 0:
        return pd.Series(dtype=float)
    zeros = pd.Series(0.0, index=cols, dtype=float)

    ac = _df(history, 'adjusted_close')
    if ac is None:
        ac = _df(history, 'close')
    if ac is None or ac.empty:
        return zeros

    available = [c for c in cols if c in ac.columns]
    if not available:
        return zeros

    ac = ac.reindex(columns=available).sort_index()
    dates = ac.index
    if len(dates) < 7:
        return zeros

    cdf = _df(history, 'close')
    if cdf is None:
        cdf = ac.copy()
    else:
        cdf = cdf.reindex(index=dates, columns=available)
        cdf = cdf.where(cdf.notna(), ac)

    vdf = _df(history, 'volume')
    adv = None
    if vdf is not None:
        vdf = vdf.reindex(index=dates, columns=available)
        dv = vdf.mul(cdf, axis=0)
        adv = dv.rolling(20, min_periods=10).median()

    edf = _df(history, 'eligible')
    if edf is not None:
        edf = edf.reindex(columns=available)
        edf = edf.reindex(index=dates).ffill()
    else:
        edf = pd.DataFrame(True, index=dates, columns=available)

    rets = ac / ac.shift(1) - 1.0
    vol = rets.rolling(60, min_periods=30).std(ddof=0)

    weekdays = dates.weekday
    weekly = dates[weekdays == 4]
    if len(weekly) == 0:
        weekly = dates[::5]

    pos = {d: i for i, d in enumerate(dates)}
    longs = []
    shorts = []
    target_n = 60

    for d in weekly:
        idx = pos.get(d, -1)
        if idx < 60:
            continue

        px = ac.iloc[idx]
        v = vol.iloc[idx]
        p_prev = ac.iloc[idx - 1]
        p5 = ac.iloc[idx - 5]
        pclose = cdf.iloc[idx]

        ret1 = px / p_prev - 1.0
        ret5 = px / p5 - 1.0
        raw = 0.8 * ret5 + 0.2 * ret1

        try:
            e = edf.loc[d]
        except Exception:
            e = pd.Series(False, index=available)
        e = e.reindex(available).fillna(False).astype(bool)

        base = px.notna() & p_prev.notna() & p5.notna() & v.notna() & raw.notna()
        base = base & np.isfinite(raw) & np.isfinite(v)
        base = base & (v > 0.003) & (v < 0.10)
        base = base & pclose.notna() & (pclose > 5.0) & e
        if adv is not None:
            a = adv.iloc[idx]
            base = base & a.notna() & (a > 3e6)

        candidates = [c for c in available if bool(base.get(c, False))]
        if len(candidates) < 12:
            base = px.notna() & p_prev.notna() & p5.notna() & v.notna() & raw.notna()
            base = base & np.isfinite(raw) & np.isfinite(v)
            base = base & (v > 0.002) & (v < 0.12)
            base = base & pclose.notna() & (pclose > 3.0) & e
            if adv is not None:
                a = adv.iloc[idx]
                base = base & a.notna() & (a > 1e6)
            candidates = [c for c in available if bool(base.get(c, False))]
            if len(candidates) < 4:
                continue

        score = (-raw.loc[candidates] / v.loc[candidates]).replace([np.inf, -np.inf], np.nan).dropna()
        candidates = list(score.index)
        if len(candidates) < 4:
            continue

        n_target = min(target_n, max(1, len(candidates) // 3))
        if n_target < 1:
            continue

        sorted_c = sorted(candidates, key=lambda c: (-float(score[c]), c))
        cand_set = set(candidates)
        keep_n = min(len(sorted_c), max(n_target * 2, n_target + 10))
        top_keep = set(sorted_c[:keep_n])
        bottom_keep = set(sorted_c[-keep_n:])

        new_longs = [c for c in longs if c in cand_set and c in top_keep]
        new_shorts = [c for c in shorts if c in cand_set and c in bottom_keep]

        common = set(new_longs) & set(new_shorts)
        if common:
            for c in list(common):
                if float(score.get(c, 0.0)) >= 0.0:
                    if c in new_shorts:
                        new_shorts.remove(c)
                else:
                    if c in new_longs:
                        new_longs.remove(c)

        if len(new_longs) > n_target:
            new_longs = sorted(new_longs, key=lambda c: (-float(score[c]), c))[:n_target]
        if len(new_shorts) > n_target:
            new_shorts = sorted(new_shorts, key=lambda c: (float(score[c]), c))[:n_target]

        held = set(new_longs) | set(new_shorts)

        for c in sorted_c:
            if len(new_longs) >= n_target:
                break
            if c in held:
                continue
            new_longs.append(c)
            held.add(c)

        for c in reversed(sorted_c):
            if len(new_shorts) >= n_target:
                break
            if c in held:
                continue
            new_shorts.append(c)
            held.add(c)

        longs, shorts = new_longs, new_shorts

    if len(longs) == 0 or len(shorts) == 0:
        idx = len(dates) - 1
        if idx >= 5:
            px = ac.iloc[idx]
            p_prev = ac.iloc[idx - 1]
            p5 = ac.iloc[idx - 5]
            pclose = cdf.iloc[idx]
            v = vol.iloc[idx]
            ret1 = px / p_prev - 1.0
            ret5 = px / p5 - 1.0
            raw = 0.8 * ret5 + 0.2 * ret1
            try:
                e = edf.loc[dates[idx]].reindex(available).fillna(False).astype(bool)
            except Exception:
                e = pd.Series(True, index=available)
            base = px.notna() & p_prev.notna() & p5.notna() & v.notna() & raw.notna()
            base = base & np.isfinite(raw) & np.isfinite(v)
            base = base & (v > 0.001) & (v < 0.15)
            base = base & pclose.notna() & (pclose > 1.0) & e
            candidates = [c for c in available if bool(base.get(c, False))]
            if len(candidates) >= 4:
                score = (-raw.loc[candidates] / v.loc[candidates]).replace([np.inf, -np.inf], np.nan).dropna()
                candidates = list(score.index)
                if len(candidates) >= 4:
                    sorted_c = sorted(candidates, key=lambda c: (-float(score[c]), c))
                    n = min(30, max(1, len(candidates) // 3))
                    longs = sorted_c[:n]
                    shorts = list(reversed(sorted_c))[:n]

    long_set = []
    seen = set()
    for c in longs:
        if c in zeros.index and c not in seen:
            long_set.append(c)
            seen.add(c)

    short_set = []
    for c in shorts:
        if c in zeros.index and c not in seen:
            short_set.append(c)
            seen.add(c)

    nL = len(long_set)
    nS = len(short_set)
    if nL == 0 or nS == 0:
        return zeros

    side_gross = min(0.475, 0.02 * min(nL, nS))
    if side_gross <= 0.0:
        return zeros

    weights = zeros.copy()
    wl = side_gross / nL
    ws = -side_gross / nS
    for c in long_set:
        weights.loc[c] = wl
    for c in short_set:
        weights.loc[c] = ws

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    weights = weights.clip(-0.02, 0.02)
    gross = float(weights.abs().sum())
    if gross > 1.0:
        weights = weights * (1.0 / gross)
    return weights