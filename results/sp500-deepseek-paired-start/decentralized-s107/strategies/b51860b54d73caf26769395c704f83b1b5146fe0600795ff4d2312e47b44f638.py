import numpy as np
import pandas as pd

def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=tickers, dtype=float)

    adj = history.get('adjusted_close')
    if adj is None:
        adj = history.get('close')
    if adj is None or not isinstance(adj, pd.DataFrame) or len(adj.index) == 0:
        return zero
    if not adj.index.is_monotonic_increasing:
        adj = adj.sort_index()
    n = len(adj)
    if n < 260:
        return zero
    anchor = ((n - 1) // 21) * 21
    if anchor < 252:
        return zero

    cols = [c for c in tickers if c in adj.columns]
    if len(cols) < 20:
        return zero

    px = adj.iloc[:anchor + 1][cols].astype(float).copy()
    px = px.replace([np.inf, -np.inf], np.nan)
    if px.shape[0] < 252:
        return zero

    last = px.iloc[-1]
    valid = last.notna() & (last > 0) & (px.notna().sum(axis=0) >= 200)

    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame) and len(elig.index) > 0:
        if not elig.index.is_monotonic_increasing:
            elig = elig.sort_index()
        e = elig.reindex(index=px.index, columns=cols)
        if len(e) > 0:
            e_last = e.iloc[-1]
            valid = valid & e_last.fillna(False).astype(bool)

    vol = history.get('volume')
    if isinstance(vol, pd.DataFrame) and len(vol.index) > 0:
        if not vol.index.is_monotonic_increasing:
            vol = vol.sort_index()
        v = vol.reindex(index=px.index, columns=cols).astype(float)
        dollar = v * px
        liq = dollar.tail(60).median(axis=0)
        liq = liq.replace([np.inf, -np.inf], np.nan)
        valid = valid & liq.notna() & (liq > 0)
        if valid.sum() >= 30:
            thresh = liq[valid].quantile(0.2)
            valid = valid & (liq >= thresh)

    valid_idx = valid[valid].index
    if len(valid_idx) < 30:
        return zero

    rets = px.pct_change()
    rets = rets.replace([np.inf, -np.inf], np.nan)
    window = rets.tail(126)

    mkt = window.mean(axis=1)
    mkt_c = mkt - mkt.mean()
    x_c = window.sub(window.mean(axis=0), axis=1)
    denom = float((mkt_c ** 2).sum())
    if denom > 0:
        beta = x_c.mul(mkt_c, axis=0).sum(axis=0) / denom
    else:
        beta = pd.Series(0.0, index=cols)
    beta = beta.replace([np.inf, -np.inf], np.nan)

    vol126 = window.std(axis=0) * np.sqrt(252.0)
    vol126 = vol126.replace([np.inf, -np.inf], np.nan)

    def _ret_lag(lag):
        if len(px) > lag:
            return px.iloc[-1] / px.iloc[-1 - lag] - 1.0
        return pd.Series(np.nan, index=cols)

    ret21 = _ret_lag(21).replace([np.inf, -np.inf], np.nan)

    if len(px) > 252:
        mom12 = px.iloc[-22] / px.iloc[-253] - 1.0
    elif len(px) > 126:
        mom12 = px.iloc[-22] / px.iloc[-127] - 1.0
    else:
        mom12 = _ret_lag(126)
    mom12 = mom12.replace([np.inf, -np.inf], np.nan)

    high252 = px.tail(252).max(axis=0)
    prox = last / high252.replace(0.0, np.nan) - 1.0
    prox = prox.replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({
        'beta': beta,
        'vol': vol126,
        'ret21': ret21,
        'mom12': mom12,
        'prox': prox
    }).loc[valid_idx].copy()

    for c in df.columns:
        med = df[c].median()
        if pd.isna(med):
            med = 0.0
        df[c] = df[c].fillna(med)

    def rp(s):
        return s.rank(method='average', pct=True)

    score = (0.40 * rp(df['vol'])
             + 0.30 * rp(df['ret21'])
             - 0.20 * rp(df['mom12'])
             - 0.10 * rp(df['prox']))
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 30:
        return zero

    beta_s = df.loc[score.index, 'beta']
    num_buckets = min(10, max(2, len(beta_s) // 10))
    try:
        buckets = pd.qcut(beta_s, q=num_buckets, labels=False, duplicates='drop')
    except Exception:
        return zero
    if buckets.isna().all():
        return zero

    longs = []
    shorts = []
    for b in sorted(buckets.dropna().unique()):
        idx_b = buckets[buckets == b].index
        if len(idx_b) < 4:
            continue
        s_b = score.reindex(idx_b).dropna().sort_values(ascending=False)
        if len(s_b) < 4:
            continue
        k = max(1, min(5, len(s_b) // 2))
        longs.extend(s_b.index[:k].tolist())
        shorts.extend(s_b.index[-k:].tolist())

    longs = list(dict.fromkeys(longs))
    shorts = list(dict.fromkeys(shorts))
    if len(longs) == 0 or len(shorts) == 0:
        return zero
    m = min(len(longs), len(shorts))
    if m < 1:
        return zero
    longs = longs[:m]
    shorts = shorts[:m]

    w = min(0.02, 0.5 / float(m))
    weights = zero.copy()
    weights.loc[longs] = w
    weights.loc[shorts] = -w

    gross = float(weights.abs().sum())
    if gross > 1.0 + 1e-12:
        weights = weights * (1.0 / gross)
    weights = weights.clip(-0.02, 0.02)
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights.reindex(tickers).fillna(0.0)