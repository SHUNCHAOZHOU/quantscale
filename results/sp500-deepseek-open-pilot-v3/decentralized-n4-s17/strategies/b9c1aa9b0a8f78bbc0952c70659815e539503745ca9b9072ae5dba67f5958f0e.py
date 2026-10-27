import numpy as np
import pandas as pd


def generate_weights(history, universe):
    u = pd.Index(universe)
    zero = pd.Series(0.0, index=u, dtype=float)
    if len(u) == 0:
        return zero

    adj = history.get('adjusted_close')
    if not isinstance(adj, pd.DataFrame) or adj.empty:
        adj = history.get('close')
    if not isinstance(adj, pd.DataFrame) or adj.empty:
        return zero

    if not adj.index.is_monotonic_increasing:
        adj = adj.sort_index()
    adj = adj.reindex(columns=u)
    n = len(adj)
    if n < 260:
        return zero

    last_pos = n - 1
    reb_pos = last_pos - (last_pos % 21)
    if reb_pos < 252:
        return zero

    start = max(0, reb_pos - 259)
    p = adj.iloc[start:reb_pos + 1].ffill()
    if len(p) < 200:
        return zero

    last = p.iloc[-1]
    last_num = pd.to_numeric(last, errors='coerce')
    valid = last_num.notna() & np.isfinite(last_num.to_numpy()) & (last_num > 0.5)

    raw = adj.iloc[start:reb_pos + 1].reindex(columns=u)
    raw_counts = raw.notna().sum()
    valid = valid & (raw_counts >= min(120, max(1, len(p) // 2)))

    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame) and not elig.empty:
        try:
            e = elig.reindex(index=adj.index, columns=u).loc[p.index].ffill().iloc[-1]
            valid = valid & e.fillna(False).astype(bool)
        except Exception:
            pass

    vol_df = history.get('volume')
    if isinstance(vol_df, pd.DataFrame) and not vol_df.empty:
        try:
            v = vol_df.reindex(index=adj.index, columns=u).loc[p.index].ffill()
            close_df = history.get('close')
            if isinstance(close_df, pd.DataFrame) and not close_df.empty:
                px = close_df.reindex(index=adj.index, columns=u).loc[p.index].ffill()
            else:
                px = p
            if len(v) >= 20:
                dollar = (v.iloc[-20:] * px.iloc[-20:]).median()
                valid = valid & dollar.notna() & np.isfinite(dollar.to_numpy()) & (dollar > 1e6)
        except Exception:
            pass

    valid_cols = valid[valid].index
    if len(valid_cols) < 10:
        return zero

    pv = p.loc[:, valid_cols]
    logp = np.log(pv.replace([np.inf, -np.inf], np.nan))
    rets = logp.diff().iloc[1:]
    if len(rets) < 60:
        return zero

    counts = rets.notna().sum()
    good_cols = counts[counts >= min(120, max(1, len(rets) // 2))].index
    if len(good_cols) < 10:
        return zero

    rets = rets.loc[:, good_cols]
    mkt = rets.mean(axis=1)
    resid = rets.sub(mkt, axis=0)

    if len(resid) < 60:
        return zero

    mom = resid.iloc[:-21].sum()
    rev = resid.iloc[-21:].sum()
    volr = resid.std(ddof=0).replace([np.inf, -np.inf], np.nan)
    volr = volr.where(volr > 1e-12)

    if len(resid) >= 126:
        mom6 = resid.iloc[-126:-21].sum()
        mom6_z = mom6 / (volr * np.sqrt(105.0))
    else:
        mom6_z = pd.Series(0.0, index=resid.columns, dtype=float)

    mom_z = mom / (volr * np.sqrt(max(1.0, float(len(resid) - 21))))
    rev_z = rev / (volr * np.sqrt(21.0))

    r_mom = mom_z.replace([np.inf, -np.inf], np.nan).rank(method='average', pct=True)
    r_mom6 = mom6_z.replace([np.inf, -np.inf], np.nan).rank(method='average', pct=True)
    r_rev = (-rev_z).replace([np.inf, -np.inf], np.nan).rank(method='average', pct=True)

    score = 0.70 * r_mom + 0.15 * r_mom6 + 0.15 * r_rev
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 10:
        return zero

    ordered = score.sort_values(ascending=False, kind='mergesort')
    n_each = min(25, len(ordered) // 2)
    if n_each < 1:
        return zero

    longs = ordered.index[:n_each]
    shorts = ordered.index[-n_each:]
    weight = min(0.02, 0.5 / float(n_each))

    w = pd.Series(0.0, index=u, dtype=float)
    for t in longs:
        if t in w.index:
            w.loc[t] = weight
    for t in shorts:
        if t in w.index:
            w.loc[t] = -weight

    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return w.astype(float)