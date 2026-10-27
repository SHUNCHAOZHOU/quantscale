import numpy as np
import pandas as pd
import math


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    zero = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return zero

    def get_df(name):
        if not isinstance(history, dict):
            return None
        obj = history.get(name)
        if isinstance(obj, pd.DataFrame):
            try:
                return obj.reindex(columns=tickers)
            except Exception:
                return None
        return None

    ac = get_df('adjusted_close')
    if ac is None or ac.dropna(how='all').shape[1] == 0:
        ac = get_df('close')
    if ac is None or ac.empty:
        return zero
    ac = ac.astype(float)
    dates = ac.index
    n = len(dates)
    if n < 70:
        return zero

    raw = get_df('close')
    if raw is None:
        raw = ac.copy()
    else:
        raw = raw.reindex(index=dates).astype(float)
        raw = raw.fillna(ac)

    vol = get_df('volume')
    if vol is not None:
        vol = vol.reindex(index=dates).astype(float)

    elig = get_df('eligible')
    if elig is not None:
        elig = elig.reindex(index=dates)

    ac_arr = ac.values.astype(float)
    raw_arr = raw.values.astype(float)
    vol_arr = vol.values.astype(float) if vol is not None else None
    elig_arr = elig.values if elig is not None else None

    asof = None
    try:
        if n >= 2:
            last_date = dates[-1]
            nb = last_date + pd.offsets.BDay(1)
            if nb.month != last_date.month:
                asof = n - 1
            else:
                months = np.array([d.month for d in dates])
                changes = np.where(months[:-1] != months[1:])[0]
                if changes.size > 0:
                    asof = int(changes[-1])
    except Exception:
        asof = None
    if asof is None:
        asof = n - 1
    if asof < 70:
        return zero

    ncol = len(tickers)
    nan_vec = np.full(ncol, np.nan, dtype=float)

    def compute_scores(pos):
        close_full = ac_arr[:pos + 1]
        nrow = len(close_full)
        if nrow < 60:
            return None
        sub = close_full[-min(300, nrow):]
        nsub = len(sub)
        if nsub < 60:
            return None
        cur = sub[-1]
        valid = np.isfinite(cur) & (cur > 2.0)
        cnt = np.sum(np.isfinite(sub), axis=0)
        valid = valid & (cnt >= 50)

        if elig_arr is not None:
            try:
                e = pd.to_numeric(pd.Series(elig_arr[pos]), errors='coerce').values
                eligmask = (e == 1) | np.isnan(e)
                if np.sum(eligmask) >= 10:
                    valid = valid & eligmask
            except Exception:
                pass

        if vol_arr is not None:
            try:
                start = max(0, pos - 59)
                v = vol_arr[start:pos + 1]
                c = raw_arr[start:pos + 1]
                with np.errstate(invalid='ignore'):
                    dv = np.nanmedian(v * c, axis=0)
                dvv = dv[np.isfinite(dv)]
                if dvv.size >= 5:
                    thr = max(100000.0, float(np.nanpercentile(dvv, 20)))
                    valid = valid & np.isfinite(dv) & (dv >= thr)
            except Exception:
                pass

        def ret(k, skip):
            if nsub > k + skip:
                a = sub[-1 - skip]
                b = sub[-1 - skip - k]
                return a / b - 1.0
            return nan_vec.copy()

        r12_1 = ret(231, 21) if nsub >= 253 else nan_vec.copy()
        r6_1 = ret(105, 21) if nsub >= 127 else nan_vec.copy()
        r21 = ret(21, 0) if nsub >= 22 else nan_vec.copy()

        mom = 0.65 * r12_1 + 0.35 * r6_1
        mom = np.where(np.isfinite(mom), mom, r6_1)
        mom = np.where(np.isfinite(mom), mom, r12_1)
        mom = np.where(np.isfinite(mom), mom, r21)

        with np.errstate(invalid='ignore', divide='ignore'):
            rets = sub[1:] / sub[:-1] - 1.0
        rets = np.where(np.isfinite(rets), rets, np.nan)
        win60 = min(60, rets.shape[0])
        win21 = min(21, rets.shape[0])
        if win60 > 5:
            vol60 = np.nanstd(rets[-win60:], axis=0) * math.sqrt(252.0)
        else:
            vol60 = nan_vec.copy()
        if win21 > 3:
            vol21 = np.nanstd(rets[-win21:], axis=0) * math.sqrt(252.0)
        else:
            vol21 = nan_vec.copy()
        vol60 = np.where(np.isfinite(vol60) & (vol60 > 0.005), vol60, np.nan)
        vol21 = np.where(np.isfinite(vol21) & (vol21 > 0.005), vol21, np.nan)

        valid = valid & np.isfinite(mom) & np.isfinite(vol60)
        valid = valid & (mom > -0.95) & (mom < 10.0)
        valid_rev = valid & np.isfinite(r21) & np.isfinite(vol21)
        valid_rev = valid_rev & (r21 > -0.8) & (r21 < 3.0)

        def rank_pct(x, mask):
            arr = np.where(mask, x, np.nan)
            return pd.Series(arr).rank(pct=True).values

        mom_adj = mom / vol60
        score_mom = 0.65 * rank_pct(mom_adj, valid) + 0.35 * rank_pct(mom, valid)
        score_mom = np.where(np.isfinite(score_mom), score_mom, 0.5)

        rev_adj = -r21 / vol21
        score_rev = rank_pct(rev_adj, valid_rev)
        score_rev = np.where(np.isfinite(score_rev), score_rev, 0.5)

        score_lv = rank_pct(-vol60, valid)
        score_lv = np.where(np.isfinite(score_lv), score_lv, 0.5)

        return {
            'valid': valid,
            'valid_rev': valid_rev,
            'score_mom': score_mom,
            'score_rev': score_rev,
            'score_lv': score_lv,
        }

    base = compute_scores(asof)
    if base is None or int(np.sum(base['valid'])) < 8:
        return zero

    perf_m = []
    perf_r = []
    perf_lv = []
    for lag in (21, 42, 63, 84, 105, 126):
        pos = asof - lag
        if pos < 60:
            continue
        sc = compute_scores(pos)
        if sc is None:
            continue
        fwd_j = min(pos + 21, asof)
        if fwd_j <= pos:
            continue
        p0 = ac_arr[pos]
        p1 = ac_arr[fwd_j]
        with np.errstate(invalid='ignore', divide='ignore'):
            fwd = p1 / p0 - 1.0
        fwd = np.where(np.isfinite(fwd), fwd, np.nan)
        for factor, out in (('mom', perf_m), ('rev', perf_r), ('lv', perf_lv)):
            score = sc['score_' + factor]
            if factor == 'rev':
                mask = sc['valid_rev'] & np.isfinite(score) & np.isfinite(fwd)
            else:
                mask = sc['valid'] & np.isfinite(score) & np.isfinite(fwd)
            idx = np.where(mask)[0]
            if idx.size < 8:
                continue
            s = score[idx]
            order = np.argsort(s)
            nside = min(40, idx.size // 2)
            if nside < 2:
                continue
            longs = idx[order[-nside:]]
            shorts = idx[order[:nside]]
            lr = np.nanmean(fwd[longs])
            sr = np.nanmean(fwd[shorts])
            if np.isfinite(lr) and np.isfinite(sr):
                out.append(float(lr - sr))

    def avg(lst):
        if len(lst) == 0:
            return 0.0
        v = np.nanmean(np.array(lst, dtype=float))
        return float(v) if np.isfinite(v) else 0.0

    pm = avg(perf_m)
    pr = avg(perf_r)
    pv = avg(perf_lv)
    scale = 0.04
    raws = []
    for p in (pm, pr, pv):
        raws.append(max(0.2, 1.0 + float(np.clip(p / scale, -0.8, 0.8))))
    raws = np.array(raws, dtype=float)
    w = raws / raws.sum()
    w = np.clip(w, 0.10, 0.60)
    w = w / w.sum()
    w_mom, w_rev, w_lv = w[0], w[1], w[2]

    score = (w_mom * base['score_mom'] +
             w_rev * base['score_rev'] +
             w_lv * base['score_lv'])
    valid = base['valid'] & np.isfinite(score)
    if int(np.sum(valid)) < 8:
        return zero

    idx = np.where(valid)[0]
    s = score[idx]
    order = idx[np.argsort(s)]
    nside = min(40, idx.size // 2)
    if nside < 2:
        return zero
    longs = order[-nside:]
    shorts = order[:nside]
    wt = min(0.02, 0.5 / float(nside))

    wv = np.zeros(ncol, dtype=float)
    wv[longs] = wt
    wv[shorts] = -wt
    wv = np.clip(wv, -0.02, 0.02)
    wv = np.nan_to_num(wv, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(wv, index=tickers, dtype=float).fillna(0.0)