import numpy as np
import pandas as pd


def _zero_weights(cols):
    return pd.Series(0.0, index=cols, dtype=float)


def _cap_weights(inv_raw, gross, cap):
    w = np.array(inv_raw, dtype=float, copy=True)
    n = w.shape[0]
    s = np.nansum(w)
    if not np.isfinite(s) or s <= 0:
        return np.full(n, gross / float(n))
    w = w / s * gross
    for _ in range(40):
        over = w > cap
        if not over.any():
            break
        excess = float(np.sum(w[over]) - cap * np.sum(over))
        w[over] = cap
        free = ~over
        if not free.any():
            break
        fs = float(np.sum(w[free]))
        if fs <= 0:
            break
        w[free] = w[free] * (1.0 + excess / fs)
    return w


def generate_weights(history, universe):
    cols = pd.Index(universe)
    if len(cols) == 0:
        return _zero_weights(cols)

    px = history.get('adjusted_close')
    if not isinstance(px, pd.DataFrame) or px.shape[0] == 0:
        px = history.get('close')
    if not isinstance(px, pd.DataFrame) or px.shape[0] == 0:
        return _zero_weights(cols)

    px = px.reindex(columns=cols)
    dates = px.index
    if dates.shape[0] < 140:
        return _zero_weights(cols)

    # Most recent month-end close available: the book is frozen inside a month.
    months = dates.year.to_numpy() * 12 + dates.month.to_numpy()
    last_of_month = np.zeros(len(dates), dtype=bool)
    last_of_month[:-1] = months[1:] != months[:-1]
    last_of_month[-1] = True
    pos = int(np.flatnonzero(last_of_month)[-1])

    start = max(0, pos - 259)
    seg = px.iloc[start:pos + 1]
    if seg.shape[0] < 130:
        return _zero_weights(cols)

    P = seg.to_numpy(dtype=float, copy=True)
    n = P.shape[0]
    m = P.shape[1]
    asof = dates[pos]

    with np.errstate(divide='ignore', invalid='ignore'):
        R = P[1:] / P[:-1] - 1.0
    R[~np.isfinite(R)] = np.nan

    vol_sub = R[-60:] if R.shape[0] >= 60 else R
    vol = np.full(m, np.nan)
    if vol_sub.shape[0] >= 5:
        cnt = np.sum(np.isfinite(vol_sub), axis=0)
        v = np.nanstd(vol_sub, axis=0)
        v[cnt < max(20, vol_sub.shape[0] // 2)] = np.nan
        vol = v

    navail = np.sum(np.isfinite(P), axis=0)

    m12 = np.full(m, np.nan)
    m6 = np.full(m, np.nan)
    rev = np.full(m, np.nan)
    if n >= 22:
        with np.errstate(divide='ignore', invalid='ignore'):
            rev = P[-1] / P[-22] - 1.0
            if n >= 253:
                m12 = P[-22] / P[-253] - 1.0
            else:
                m12 = P[-22] / P[0] - 1.0
            if n >= 127:
                m6 = P[-22] / P[-127] - 1.0
    m12[~np.isfinite(m12)] = np.nan
    m6[~np.isfinite(m6)] = np.nan
    rev[~np.isfinite(rev)] = np.nan

    # Eligibility flag as of the decision close.
    elig = np.ones(m, dtype=bool)
    e = history.get('eligible')
    if isinstance(e, pd.DataFrame) and e.shape[0] > 0:
        e = e.reindex(columns=cols)
        sub = e.loc[:asof]
        if sub.shape[0] > 0:
            arr = sub.iloc[-1].to_numpy()
            if np.any(pd.notna(arr)):
                elig = np.where(pd.isna(arr), False, arr.astype(bool))

    # Liquidity screen: median dollar volume over the last 60 sessions.
    liq = np.full(m, np.nan)
    vdf = history.get('volume')
    if isinstance(vdf, pd.DataFrame) and vdf.shape[0] > 0:
        vseg = vdf.reindex(index=seg.index, columns=cols).to_numpy(dtype=float)
        with np.errstate(invalid='ignore'):
            dv = vseg * P
        liq = np.nanmedian(dv[-60:], axis=0)

    last_px = P[-1]
    valid = (
        np.isfinite(last_px)
        & (last_px > 1.0)
        & np.isfinite(vol)
        & (vol > 1e-6)
        & np.isfinite(m12)
        & np.isfinite(m6)
        & np.isfinite(rev)
        & elig
        & np.isfinite(liq)
        & (liq > 0)
        & (navail >= int(0.9 * n))
    )
    idx = np.flatnonzero(valid)
    if idx.size < 12:
        return _zero_weights(cols)

    feat = pd.DataFrame({ 
        'm12': m12[idx] / vol[idx],
        'm6': m6[idx] / vol[idx],
        'rev': rev[idx],
        'vol': vol[idx],
    }, index=cols[idx])
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    if feat.shape[0] < 12:
        return _zero_weights(cols)

    score = (
        0.55 * feat['m12'].rank(pct=True)
        + 0.20 * feat['m6'].rank(pct=True)
        - 0.15 * feat['rev'].rank(pct=True)
        - 0.10 * feat['vol'].rank(pct=True)
    )

    n_side = min(40, feat.shape[0] // 2)
    if n_side < 5:
        return _zero_weights(cols)

    order = score.sort_values(ascending=False)
    longs = order.index[:n_side]
    shorts = order.index[-n_side:]

    side_gross = 0.49
    w = _zero_weights(cols)
    invl = 1.0 / np.maximum(feat.loc[longs, 'vol'].to_numpy(dtype=float), 1e-6)
    invs = 1.0 / np.maximum(feat.loc[shorts, 'vol'].to_numpy(dtype=float), 1e-6)
    w.loc[longs] = _cap_weights(invl, side_gross, 0.02)
    w.loc[shorts] = -_cap_weights(invs, side_gross, 0.02)

    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    tot = float(w.sum())
    if abs(tot) > 1e-12:
        nz = (w != 0.0).to_numpy()
        if nz.any():
            w.iloc[np.flatnonzero(nz)] = w.to_numpy()[nz] - tot / float(nz.sum())

    w = w.clip(-0.02, 0.02)
    gross = float(w.abs().sum())
    if gross > 1.0:
        w = w * (1.0 / gross)
    return w.astype(float)