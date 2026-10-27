import numpy as np
import pandas as pd

def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero
    adj = history.get('adjusted_close')
    if not isinstance(adj, pd.DataFrame) or adj.empty:
        adj = history.get('close')
    if not isinstance(adj, pd.DataFrame) or adj.empty:
        return zero
    cols = [c for c in universe if c in adj.columns]
    if len(cols) < 20:
        return zero
    adj = adj[cols].sort_index()
    n = len(adj)
    if n < 300:
        return zero
    reb = 63
    anchor = ((n - 1) // reb) * reb
    if anchor < 280:
        return zero
    start = max(0, anchor - 300)
    px = adj.iloc[start:anchor + 1].ffill()
    if len(px) < 280:
        return zero
    last = px.iloc[-1]
    valid = last.notna() & np.isfinite(last) & (last > 0.0)
    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame) and not elig.empty:
        e = elig.reindex(index=adj.index, columns=cols).iloc[anchor]
        valid = valid & e.fillna(False).astype(bool)
    vol = history.get('volume')
    close = history.get('close')
    if isinstance(vol, pd.DataFrame) and isinstance(close, pd.DataFrame):
        v = vol.reindex(index=px.index, columns=cols).ffill()
        c = close.reindex(index=px.index, columns=cols).ffill()
        dv = (v * c).tail(60).median()
        valid = valid & dv.notna() & (dv >= 1.0e6)
    rets = px.pct_change().replace([np.inf, -np.inf], np.nan)
    if len(rets) < 260:
        return zero
    mkt = rets.mean(axis=1)
    win = rets.tail(252)
    mwin = mkt.tail(252)
    if win.shape[0] < 252:
        return zero
    cov = win.cov(mwin)
    var = mwin.var()
    if not np.isfinite(var) or var <= 0.0:
        return zero
    beta = cov / var
    beta = beta.reindex(cols)
    mvals = mwin.to_numpy(dtype=float)
    bvals = beta.to_numpy(dtype=float)
    wvals = win.to_numpy(dtype=float)
    res = wvals - mvals[:, None] * bvals[None, :]
    res_df = pd.DataFrame(res, index=win.index, columns=win.columns)
    res_mom = res_df.iloc[:-21].sum()
    res_vol = res_df.std()
    valid = valid & res_mom.notna() & res_vol.notna() & (res_vol > 0.0)
    valid = valid & (win.notna().sum() >= 200)
    if int(valid.sum()) < 20:
        return zero
    score = res_mom / res_vol
    f = pd.DataFrame({'score': score, 'beta': beta}).loc[valid].replace([np.inf, -np.inf], np.nan).dropna()
    if len(f) < 20:
        return zero
    try:
        f['bucket'] = pd.qcut(f['beta'], 5, labels=False, duplicates='drop')
    except Exception:
        f['bucket'] = 0
    n_buckets = int(f['bucket'].nunique())
    if n_buckets < 1:
        return zero
    n_side_target = min(50, max(10, len(f) // 8))
    k_target = max(1, n_side_target // n_buckets)
    longs = []
    shorts = []
    for b, g in f.groupby('bucket'):
        g = g.sort_values('score', ascending=False)
        k = min(k_target, len(g) // 2)
        if k < 1:
            continue
        longs.extend(g.index[:k].tolist())
        shorts.extend(g.index[-k:].tolist())
    m = min(len(longs), len(shorts))
    if m < 5:
        return zero
    w = min(0.02, 0.5 / float(m))
    out = zero.copy()
    for t in longs:
        out.loc[t] = w
    for t in shorts:
        out.loc[t] = -w
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
    gross = float(out.abs().sum())
    if gross > 1.0 and gross > 0.0:
        out = out * (1.0 / gross)
    net = float(out.sum())
    if abs(net) > 1e-12:
        nz = out != 0.0
        if nz.any():
            out.loc[nz] = out.loc[nz] - net / float(nz.sum())
            out = out.clip(-0.02, 0.02)
            gross = float(out.abs().sum())
            if gross > 1.0 and gross > 0.0:
                out = out * (1.0 / gross)
    return out.astype(float)