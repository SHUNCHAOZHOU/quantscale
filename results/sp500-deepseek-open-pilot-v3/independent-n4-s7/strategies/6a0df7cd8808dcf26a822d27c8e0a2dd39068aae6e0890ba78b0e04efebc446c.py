import numpy as np
import pandas as pd


def _field(history, names):
    for k in names:
        if k in history:
            df = history[k]
            if isinstance(df, pd.DataFrame) and df.shape[0] > 0:
                return df
    return None


def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)
    zeros = pd.Series(0.0, index=universe)

    px = _field(history, ('adjusted_close', 'adj_close', 'close'))
    if px is None:
        return zeros
    cols = [c for c in universe if c in px.columns]
    if len(cols) == 0:
        return zeros

    p = px[cols].copy()
    if len(p) < 60:
        return zeros

    tail_n = min(len(p), 320)
    pt = p.iloc[-tail_n:]
    arr = pt.to_numpy(dtype=float)
    n, k = arr.shape

    with np.errstate(all='ignore'):
        ret_all = arr[1:] / arr[:-1] - 1.0
    ret_all[~np.isfinite(ret_all)] = np.nan
    market = np.nanmean(ret_all, axis=1)
    market = np.where(np.isfinite(market), market, 0.0)

    edf = _field(history, ('eligible', 'tradable'))
    if edf is not None:
        try:
            last = edf.reindex(columns=cols).iloc[-1]
            elig = last.fillna(False).astype(bool).to_numpy()
        except Exception:
            elig = np.ones(k, dtype=bool)
    else:
        elig = np.ones(k, dtype=bool)

    liq = np.ones(k, dtype=bool)
    vdf = _field(history, ('volume', 'vol'))
    if vdf is not None:
        try:
            v = vdf.reindex(index=pt.index, columns=cols).to_numpy(dtype=float)
            cdf = _field(history, ('close', 'adjusted_close', 'adj_close'))
            if cdf is not None:
                c = cdf.reindex(index=pt.index, columns=cols).to_numpy(dtype=float)
            else:
                c = arr
            dv = v * c
            if len(dv) >= 20:
                adv = np.nanmean(dv[-20:], axis=0)
            else:
                adv = np.nanmean(dv, axis=0)
            finite = np.isfinite(adv)
            liq[finite] = adv[finite] > 1e6
        except Exception:
            liq = np.ones(k, dtype=bool)

    tnames = []
    scores = []
    trends = []
    vols = []
    jj = []

    for j, t in enumerate(cols):
        s = arr[:, j]
        valid = np.isfinite(s) & (s > 0.0)
        if not valid[-1]:
            continue

        with np.errstate(all='ignore'):
            r = s[1:] / s[:-1] - 1.0
        r[~np.isfinite(r)] = np.nan

        y = r[-252:]
        x = market[-252:]
        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() >= 60:
            yv = y[mask]
            xv = x[mask]
            xm = xv.mean()
            ym = yv.mean()
            var = np.mean((xv - xm) ** 2)
            if var > 0.0:
                beta = np.mean((xv - xm) * (yv - ym)) / var
            else:
                beta = 1.0
        else:
            beta = 1.0

        resid = r - beta * market
        resid[~np.isfinite(resid)] = np.nan

        look = 126
        skip = 21
        if len(resid) < look + skip:
            look = 63
            if len(resid) < look + skip:
                look = 21
                if len(resid) < look + skip:
                    continue

        window = resid[-(skip + look):-skip] if skip > 0 else resid[-look:]
        window = window[np.isfinite(window)]
        if len(window) < max(8, look // 3):
            continue
        cum = float(np.sum(window))

        rv = resid[-252:]
        rv = rv[np.isfinite(rv)]
        if len(rv) >= 20:
            vol = float(np.std(rv, ddof=1) * np.sqrt(252.0))
        else:
            vol = np.nan
        if not np.isfinite(vol) or vol <= 0.0:
            vol = 0.20
        vol = max(vol, 0.05)

        score = cum / (vol * np.sqrt(look / 252.0))

        sv = s[valid]
        ma_n = min(200, len(sv))
        if ma_n > 0:
            ma = float(np.mean(sv[-ma_n:]))
            trend = s[-1] / ma - 1.0 if ma > 0.0 else 0.0
        else:
            trend = 0.0

        tnames.append(t)
        scores.append(score)
        trends.append(trend)
        vols.append(vol)
        jj.append(j)

    if len(tnames) < 6:
        return zeros

    scores = np.asarray(scores, dtype=float)
    trends = np.asarray(trends, dtype=float)

    def _z(a):
        a = np.asarray(a, dtype=float)
        med = np.nanmedian(a)
        mad = np.nanmedian(np.abs(a - med))
        if mad > 0.0:
            z = (a - med) / (1.4826 * mad)
        else:
            sd = np.nanstd(a)
            if sd > 0.0:
                z = (a - med) / sd
            else:
                z = np.zeros_like(a)
        z[~np.isfinite(z)] = 0.0
        return z

    zs = _z(scores)
    zt = _z(trends)
    comp = zs + 0.4 * zt

    for i, j in enumerate(jj):
        if not bool(elig[j]) or not bool(liq[j]):
            comp[i] = np.nan

    keep = np.isfinite(comp)
    if keep.sum() < 6:
        return zeros

    kept_names = [tnames[i] for i in range(len(tnames)) if keep[i]]
    comp = comp[keep]

    order = np.argsort(comp)
    n_side = min(30, len(comp) // 2)
    if n_side < 1:
        return zeros

    shorts = [kept_names[i] for i in order[:n_side]]
    longs = [kept_names[i] for i in order[-n_side:]]

    short_set = set(shorts)
    longs = [x for x in longs if x not in short_set]
    long_set = set(longs)
    shorts = [x for x in shorts if x not in long_set]

    n_side = min(len(longs), len(shorts))
    if n_side < 1:
        return zeros
    longs = longs[:n_side]
    shorts = shorts[:n_side]

    w = min(0.02, 0.98 / (2.0 * n_side))
    weights = zeros.copy()
    for t in longs:
        weights.loc[t] = w
    for t in shorts:
        weights.loc[t] = -w

    return weights.fillna(0.0)
