import numpy as np
import pandas as pd

def _get_df(history, key):
    df = history.get(key)
    if df is None:
        return None
    return df

def _rebalance_date(index):
    if len(index) == 0:
        return None
    cur = index[-1]
    try:
        mask = (index.weekday == 4) & (index <= cur)
        if mask.any():
            return index[mask][-1]
    except Exception:
        pass
    return cur

def generate_weights(history, universe):
    idx = pd.Index(universe)
    w = pd.Series(0.0, index=idx, dtype=float)
    adj = _get_df(history, 'adjusted_close')
    if adj is None or adj.empty:
        adj = _get_df(history, 'close')
    if adj is None or adj.empty:
        return w
    adj = adj.reindex(columns=idx)
    if adj.empty:
        return w
    rd = _rebalance_date(adj.index)
    if rd is None:
        return w

    def sl(key):
        df = _get_df(history, key)
        if df is None:
            return None
        try:
            return df.loc[:rd].reindex(columns=idx)
        except Exception:
            return df.reindex(columns=idx)

    adj = sl('adjusted_close')
    if adj is None or adj.empty:
        adj = sl('close')
    if adj is None or adj.empty:
        return w
    close_df = sl('close')
    if close_df is None:
        close_df = adj
    vol_df = sl('volume')
    elig_df = sl('eligible')
    last_e = None
    if elig_df is not None and len(elig_df) > 0:
        last_e = elig_df.iloc[-1]

    rows = []
    for t in idx:
        if t not in adj.columns:
            continue
        s = adj[t].dropna()
        n = len(s)
        if n < 80:
            continue
        p = float(s.iloc[-1])
        if not np.isfinite(p) or p <= 1.0:
            continue
        if last_e is not None:
            ev = last_e.get(t, False)
            try:
                if pd.isna(ev) or not bool(ev):
                    continue
            except Exception:
                if not bool(ev):
                    continue
        if vol_df is not None and close_df is not None:
            v = vol_df[t].dropna()
            c = close_df[t].dropna()
            if len(v) >= 5 and len(c) >= 5:
                dv = (c * v).dropna()
                if len(dv) >= 5:
                    mdv = float(dv.iloc[-20:].median())
                    if np.isfinite(mdv) and mdv < 1e6:
                        continue
        ret = s.pct_change().dropna()
        if len(ret) < 30:
            continue
        vol = float(ret.iloc[-60:].std())
        if not np.isfinite(vol) or vol <= 1e-12:
            vol = float(ret.std())
        if not np.isfinite(vol) or vol <= 1e-12:
            continue
        if n >= 253:
            mom = float(s.iloc[-22] / s.iloc[-253] - 1.0)
        elif n > 22:
            mom = float(s.iloc[-22] / s.iloc[0] - 1.0)
        else:
            mom = float(s.iloc[-1] / s.iloc[0] - 1.0)
        if n >= 22:
            rev = float(-(s.iloc[-1] / s.iloc[-22] - 1.0))
        else:
            rev = 0.0
        if not np.isfinite(mom):
            mom = 0.0
        if not np.isfinite(rev):
            rev = 0.0
        rows.append((t, mom, rev, vol))

    if len(rows) < 10:
        return w
    tickers = [r[0] for r in rows]
    mom = np.array([r[1] for r in rows], dtype=float)
    rev = np.array([r[2] for r in rows], dtype=float)
    vol = np.array([r[3] for r in rows], dtype=float)

    def rz(x):
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        lo = np.nanpercentile(x, 5)
        hi = np.nanpercentile(x, 95)
        x = np.clip(x, lo, hi)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        if not np.isfinite(mad) or mad <= 1e-12:
            sd = np.nanstd(x)
            if not np.isfinite(sd) or sd <= 1e-12:
                return np.zeros_like(x)
            return (x - med) / sd
        return (x - med) / (1.4826 * mad)

    score = 1.0 * rz(mom) + 0.5 * rz(rev) - 0.4 * rz(np.log(np.maximum(vol, 1e-8)))
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(score)
    m = len(order)
    k = min(25, m // 2)
    if k < 1:
        return w
    short_i = order[:k]
    long_i = order[-k:]
    wt = min(0.02, 1.0 / (2.0 * k))
    for i in long_i:
        w.loc[tickers[i]] = wt
    for i in short_i:
        w.loc[tickers[i]] = -wt
    return w