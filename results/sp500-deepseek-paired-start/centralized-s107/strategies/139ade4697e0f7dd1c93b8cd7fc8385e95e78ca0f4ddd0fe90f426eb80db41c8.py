import numpy as np
import pandas as pd


def _neutralize(w, longs, shorts):
    s = float(w.sum())
    if abs(s) <= 1e-12:
        return w
    if s > 0.0:
        grp = [c for c in longs if float(w.get(c, 0.0)) > 0.0]
        if grp:
            tot = float(w.loc[grp].sum())
            if tot > 0.0:
                w.loc[grp] = w.loc[grp] - s * (w.loc[grp] / tot)
    else:
        grp = [c for c in shorts if float(w.get(c, 0.0)) < 0.0]
        if grp:
            tot = float(-w.loc[grp].sum())
            if tot > 0.0:
                w.loc[grp] = w.loc[grp] - (s * (-w.loc[grp] / tot))
    return w


def generate_weights(history, universe):
    idx = pd.Index(universe)
    if len(idx) == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=idx, dtype=float)

    ac = history.get('adjusted_close')
    if ac is None or ac.empty:
        ac = history.get('close')
    if ac is None or ac.empty:
        return zero

    n = len(ac.index)
    period = 15
    if n < 120:
        return zero
    reb = ((n - 1) // period) * period
    if reb < 110:
        return zero

    cols = [c for c in idx if c in ac.columns]
    if len(cols) < 40:
        return zero

    lookback = 130
    start = max(0, reb - lookback + 1)
    close = ac.iloc[start:reb + 1][cols].astype(float)
    close = close.replace([np.inf, -np.inf], np.nan)
    if close.shape[0] < 100:
        return zero

    last = close.iloc[-1]
    valid = last.notna() & (close.notna().sum(axis=0) >= 100)
    if int(valid.sum()) < 30:
        return zero

    ret = close.pct_change()
    ret = ret.replace([np.inf, -np.inf], np.nan)
    vol = ret.iloc[-60:].std(axis=0, ddof=1)
    vol = vol.where(vol > 1e-6, np.nan)

    med_vol = vol[valid].median()
    if pd.notna(med_vol) and float(med_vol) > 0.0:
        valid = valid & (vol <= 3.0 * float(med_vol))

    horizon_w = [(5, 0.25), (10, 0.30), (21, 0.45)]
    num = pd.Series(0.0, index=cols, dtype=float)
    den = pd.Series(0.0, index=cols, dtype=float)
    for h, wgt in horizon_w:
        if close.shape[0] <= h + 1:
            continue
        r = close.iloc[-1] / close.iloc[-1 - h] - 1.0
        r = r.replace([np.inf, -np.inf], np.nan)
        mask = valid & r.notna()
        if int(mask.sum()) < 20:
            continue
        mu = float(r[mask].mean())
        resid = r - mu
        s = -resid / (vol + 1e-8)
        s = s.where(mask)
        num = num.add((s * wgt).fillna(0.0))
        den = den.add(pd.Series(float(wgt), index=cols).where(s.notna(), 0.0))

    den = den.replace(0.0, np.nan)
    score = num / den
    score = score.replace([np.inf, -np.inf], np.nan)

    vol_df = history.get('volume')
    if vol_df is not None and not vol_df.empty:
        vcols = [c for c in cols if c in vol_df.columns]
        if len(vcols) > 0:
            vsub = vol_df.iloc[start:reb + 1][vcols].astype(float)
            vsub = vsub.replace([np.inf, -np.inf], np.nan)
            vsub = vsub.reindex(columns=cols)
            dollar = vsub * close
            liq = dollar.iloc[-60:].median(axis=0)
            liq = liq.replace([np.inf, -np.inf], np.nan)
            valid = valid & (liq > 0.0)
            if int((liq[valid].notna()).sum()) >= 30:
                thresh = float(liq[valid].quantile(0.30))
                valid = valid & (liq >= thresh)

    elig_df = history.get('eligible')
    if elig_df is not None and not elig_df.empty:
        try:
            dt = ac.index[reb]
            if dt in elig_df.index:
                row = elig_df.loc[dt]
            else:
                row = elig_df.iloc[min(reb, len(elig_df) - 1)]
            row = row.reindex(cols).fillna(False).astype(bool)
            valid = valid & row
        except Exception:
            pass

    valid = valid & score.notna() & np.isfinite(score)
    candidates = [c for c in cols if bool(valid.get(c, False))]
    if len(candidates) < 40:
        return zero

    sc = score.loc[candidates].sort_values(ascending=False)
    k = min(70, len(candidates) // 2)
    if k < 20:
        return zero

    longs = sc.index[:k].tolist()
    shorts = sc.index[-k:].tolist()
    wmag = min(0.02, 0.98 / (2.0 * k))

    w = zero.copy()
    w.loc[longs] = wmag
    w.loc[shorts] = -wmag

    w = _neutralize(w, longs, shorts)
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)

    gross = float(w.abs().sum())
    if gross > 0.98 and gross > 0.0:
        w = w * (0.98 / gross)
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
    return w
