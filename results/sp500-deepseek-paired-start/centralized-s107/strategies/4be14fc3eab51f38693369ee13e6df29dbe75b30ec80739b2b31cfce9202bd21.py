import numpy as np
import pandas as pd


def _finalize_weights(w):
    w = w.astype(float).copy()
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(w) == 0:
        return w
    w = w - w.mean()
    w = w.clip(-0.02, 0.02)
    w = w - w.mean()
    w = w.clip(-0.02, 0.02)
    gross = w.abs().sum()
    if gross <= 1e-12:
        return w
    maxabs = w.abs().max()
    scale = 1.0 / gross
    if maxabs > 0:
        scale = min(scale, 0.02 / maxabs)
    w = w * scale
    w = w - w.mean()
    w = w.clip(-0.02, 0.02)
    gross = w.abs().sum()
    if gross > 1.0 + 1e-12:
        w = w / gross
    if abs(w.sum()) > 1e-12:
        w = w - w.mean()
        w = w.clip(-0.02, 0.02)
    return w


def _target_weights(adj, eligible, cols, asof, universe):
    out = pd.Series(0.0, index=universe)
    if adj is None:
        return out
    px_all = adj[cols]
    if not px_all.index.is_monotonic_increasing:
        px_all = px_all.sort_index()
    px = px_all.loc[:asof].tail(330)
    n = len(px)
    if n < 255:
        return out

    last = px.iloc[-1]
    valid = last.notna() & (last > 0)
    valid = valid & (px.notna().sum() >= 200)

    if eligible is not None and isinstance(eligible, pd.DataFrame):
        e = eligible
        if not e.index.is_monotonic_increasing:
            e = e.sort_index()
        ecols = [c for c in cols if c in e.columns]
        if ecols:
            er = e[ecols].loc[:asof].tail(1)
            if len(er) > 0:
                ev = er.iloc[-1].reindex(cols).fillna(False).astype(bool)
                valid = valid & ev

    idx = valid[valid].index
    if len(idx) < 30:
        return out

    ret5 = px.iloc[-1] / px.iloc[-6] - 1.0
    ret120 = px.iloc[-1] / px.iloc[-121] - 1.0
    ret12_1 = px.iloc[-22] / px.iloc[-253] - 1.0
    ret6_1 = px.iloc[-22] / px.iloc[-127] - 1.0

    r5 = ret5.loc[idx]
    r120 = ret120.loc[idx]
    r12 = ret12_1.loc[idx]
    r6 = ret6_1.loc[idx]

    good = r5.notna() & r120.notna() & r12.notna() & r6.notna()
    good = good & np.isfinite(r5) & np.isfinite(r120) & np.isfinite(r12) & np.isfinite(r6)
    idx2 = idx[good]
    if len(idx2) < 30:
        return out

    r5 = r5.loc[idx2]
    r120 = r120.loc[idx2]
    r12 = r12.loc[idx2]
    r6 = r6.loc[idx2]

    def rp(s):
        return s.rank(method='average', pct=True)

    raw = -0.75 * (rp(r5) - 0.5) - 0.25 * (rp(r120) - 0.5)

    X = np.column_stack([np.ones(len(idx2)), rp(r12).values, rp(r6).values])
    y = raw.values
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
    except Exception:
        resid = y - y.mean()

    resid_s = pd.Series(resid, index=idx2)
    resid_s = resid_s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(resid_s) < 30:
        return out

    score = resid_s.rank(method='average', pct=True) - 0.5
    w = _finalize_weights(score)
    out.loc[w.index] = w
    return out


def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)

    adj = history.get('adjusted_close')
    if adj is None:
        adj = history.get('close')
    if adj is None or not isinstance(adj, pd.DataFrame):
        return pd.Series(0.0, index=universe)

    cols = [c for c in universe if c in adj.columns]
    if len(cols) == 0:
        return pd.Series(0.0, index=universe)

    if not adj.index.is_monotonic_increasing:
        adj = adj.sort_index()
    idx = adj.index
    if len(idx) == 0:
        return pd.Series(0.0, index=universe)

    pos_last = (len(idx) - 1) // 7 * 7
    current_reb = idx[pos_last]
    prev_reb = idx[pos_last - 7] if pos_last >= 7 else None
    eligible = history.get('eligible')

    target = _target_weights(adj, eligible, cols, current_reb, universe)

    if prev_reb is not None:
        prev = _target_weights(adj, eligible, cols, prev_reb, universe)
        nz = target[target.abs() > 0]
        if len(nz) > 0:
            band = max(0.0005, 0.25 * float(nz.abs().median()))
        else:
            band = 0.0005
        keep = (target - prev).abs() < band
        adjusted = target.copy()
        adjusted[keep] = prev[keep]
        target = _finalize_weights(adjusted)

    return target.reindex(universe).fillna(0.0)