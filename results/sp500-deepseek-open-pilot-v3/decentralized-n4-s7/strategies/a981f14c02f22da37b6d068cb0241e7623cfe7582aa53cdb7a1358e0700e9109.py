import numpy as np
import pandas as pd


def _zscore(s):
    s = s.astype(float).replace([np.inf, -np.inf], np.nan)
    med = s.median()
    mad = (s - med).abs().median()
    if pd.notna(mad) and mad > 1e-12:
        z = (s - med) / (1.4826 * mad)
    else:
        std = s.std()
        if pd.notna(std) and std > 1e-12:
            z = (s - s.mean()) / std
        else:
            z = pd.Series(0.0, index=s.index)
    return z.clip(-5.0, 5.0)


def generate_weights(history, universe):
    u = pd.Index(universe)
    zeros = pd.Series(0.0, index=u)
    if len(u) == 0:
        return zeros

    px = history.get('adjusted_close')
    if px is None or px.empty:
        px = history.get('close')
    if px is None or px.empty:
        return zeros

    cols = [t for t in u if t in px.columns]
    if len(cols) < 10:
        return zeros

    px = px.loc[:, cols].astype(float).dropna(how='all')
    if len(px) < 30:
        return zeros

    eligible = None
    e = history.get('eligible')
    if e is not None:
        e = e.reindex(columns=cols)
        e = e.reindex(px.index)
        if len(e) > 0:
            eligible = e.iloc[-1].fillna(False).astype(bool)

    p0 = px.iloc[-1]
    p5 = px.iloc[-6] if len(px) >= 6 else px.iloc[0]
    p21 = px.iloc[-22] if len(px) >= 22 else px.iloc[0]

    r5 = (p0 / p5 - 1.0).replace([np.inf, -np.inf], np.nan)
    r21 = (p0 / p21 - 1.0).replace([np.inf, -np.inf], np.nan)

    rets = px.pct_change()
    if len(rets) >= 60:
        vol = rets.iloc[-60:].std()
    else:
        vol = rets.std()

    valid = p0.notna() & (p0 > 1.0) & r5.notna() & r21.notna() & vol.notna() & (vol > 0.0)

    if eligible is not None:
        valid = valid & eligible.reindex(cols).fillna(False)

    v = history.get('volume')
    c = history.get('close')
    if v is not None and c is not None:
        v = v.reindex(columns=cols).reindex(px.index)
        c = c.reindex(columns=cols).reindex(px.index)
        if len(v) >= 20 and len(c) >= 20:
            dv = (v.iloc[-20:] * c.iloc[-20:]).median(axis=0, skipna=True)
            if dv.notna().any():
                positive = dv[dv > 0.0]
                if len(positive) > 0:
                    thresh = positive.quantile(0.30)
                    valid = valid & dv.notna() & (dv >= thresh)
                else:
                    valid = valid & dv.notna()

    valid = valid.fillna(False)
    if int(valid.sum()) < 20:
        return zeros

    score = (
        -_zscore(r21.where(valid))
        - 0.50 * _zscore(r5.where(valid))
        - 0.20 * _zscore(vol.where(valid))
    )
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(score)
    if n < 20:
        return zeros

    ordered = score.sort_values()
    k = min(24, n // 2)
    if k < 1:
        return zeros

    shorts = ordered.index[:k]
    longs = ordered.index[-k:]

    weight = min(0.02, 0.5 / float(k))
    w = pd.Series(0.0, index=u)
    w.loc[longs] = weight
    w.loc[shorts] = -weight
    return w
