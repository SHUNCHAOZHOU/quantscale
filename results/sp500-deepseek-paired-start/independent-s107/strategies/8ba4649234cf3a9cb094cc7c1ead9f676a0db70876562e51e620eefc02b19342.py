import numpy as np
import pandas as pd


def _zrank(s):
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return s * 0.0
    return (s.rank(method='average', pct=True) - 0.5) * 2.0


def _week_key(ts):
    try:
        iso = ts.isocalendar()
        return (int(iso[0]), int(iso[1]))
    except Exception:
        return (int(ts.year), int(ts.month) * 5 + int(ts.day) // 7)


def _last_rebalance_pos(idx):
    i = len(idx) - 1
    while i >= 1:
        if _week_key(idx[i]) != _week_key(idx[i - 1]):
            break
        i -= 1
    return i


def _inverse_vol_weights(names, vol, cap, total):
    n = len(names)
    if n == 0:
        return np.zeros(0)
    v = vol.reindex(names).values.astype(float)
    v = np.where(np.isfinite(v) & (v > 1e-4), v, np.nan)
    med = np.nanmedian(v) if np.isfinite(np.nanmedian(v)) else 0.02
    v = np.where(np.isfinite(v), v, med)
    r = 1.0 / np.maximum(v, 1e-4)
    s = r.sum()
    if not np.isfinite(s) or s <= 0:
        w = np.full(n, total / float(n))
    else:
        w = r / s * total
    for _ in range(8):
        over = w > cap
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = ~over
        if free.any():
            rs = r[free].sum()
            if np.isfinite(rs) and rs > 0:
                w[free] = w[free] + excess * (r[free] / rs)
            else:
                w[free] = w[free] + excess / float(free.sum())
        else:
            break
    return w


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    zeros = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return zeros

    px_all = history.get('adjusted_close')
    if px_all is None:
        return zeros
    try:
        if px_all.shape[0] < 260 or px_all.shape[1] == 0:
            return zeros
        idx = px_all.index
    except Exception:
        return zeros

    ref_pos = _last_rebalance_pos(idx)
    if ref_pos < 252:
        return zeros
    ref_date = idx[ref_pos]

    start = ref_pos - 252
    px = px_all.iloc[start:ref_pos + 1].reindex(columns=tickers).astype(float)
    px = px.replace([np.inf, -np.inf], np.nan)
    if px.shape[0] < 253 or px.shape[1] == 0:
        return zeros

    p_now = px.iloc[-1]
    p_21 = px.iloc[-22]
    p_126 = px.iloc[-127]
    p_252 = px.iloc[0]

    rev = p_now / p_21 - 1.0
    mom6 = p_21 / p_126 - 1.0
    mom12 = p_21 / p_252 - 1.0

    rets = px.pct_change()
    rets = rets.replace([np.inf, -np.inf], np.nan)
    vol60 = rets.iloc[-60:].std()

    sub = rets.iloc[-120:]
    beta = pd.Series(np.nan, index=tickers, dtype=float)
    if sub.shape[0] >= 40 and sub.shape[1] > 0:
        m = sub.mean(axis=1)
        dm = m - m.mean()
        var_m = float((dm ** 2).sum())
        if np.isfinite(var_m) and var_m > 0:
            dem = sub.sub(m, axis=0)
            beta = dem.mul(dm, axis=0).sum() / var_m

    mdv = pd.Series(np.nan, index=tickers, dtype=float)
    vol_all = history.get('volume')
    if vol_all is not None:
        try:
            vwin = vol_all.iloc[start:ref_pos + 1].reindex(columns=tickers).astype(float)
            vwin = vwin.replace([np.inf, -np.inf], np.nan)
            if vwin.shape[0] > 0:
                mdv = (vwin * px).iloc[-60:].median()
        except Exception:
            mdv = pd.Series(np.nan, index=tickers, dtype=float)

    elig_flag = pd.Series(True, index=tickers, dtype=bool)
    elig_all = history.get('eligible')
    if elig_all is not None:
        try:
            if elig_all.shape[0] > 0 and elig_all.shape[1] > 0:
                eidx = elig_all.index
                pos = int(eidx.searchsorted(ref_date, side='right')) - 1
                if pos >= 0:
                    erow = elig_all.iloc[pos].reindex(tickers)
                    elig_flag = erow.fillna(False).astype(bool)
                else:
                    elig_flag = pd.Series(False, index=tickers, dtype=bool)
        except Exception:
            pass

    df = pd.DataFrame({
        'rev': rev,
        'mom6': mom6,
        'mom12': mom12,
        'vol': vol60,
        'beta': beta,
        'mdv': mdv,
        'px': p_now,
    }).reindex(tickers)

    valid = (
        df['px'].notna() & (df['px'] > 3.0)
        & np.isfinite(df['rev']) & np.isfinite(df['mom6']) & np.isfinite(df['mom12'])
        & df['vol'].notna() & (df['vol'] > 1e-4)
        & df['beta'].notna() & np.isfinite(df['beta'])
        & df['mdv'].notna() & (df['mdv'] > 2e6)
        & elig_flag.reindex(tickers).fillna(False).astype(bool)
    )
    df = df[valid]
    if len(df) < 40:
        return zeros

    z_rev = _zrank(df['rev'])
    z_m6 = _zrank(df['mom6'])
    z_m12 = _zrank(df['mom12'])
    z_vol = _zrank(df['vol'])

    common = z_rev.index.intersection(z_m6.index).intersection(z_m12.index).intersection(z_vol.index)
    if len(common) < 30:
        return zeros

    score = (-0.9 * z_rev + 0.45 * z_m6 + 0.35 * z_m12 - 0.2 * z_vol).reindex(common)
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 30:
        return zeros

    d = df.loc[score.index].copy()
    d['score'] = score.reindex(d.index)
    d = d.dropna(subset=['score', 'beta', 'vol'])
    if len(d) < 30:
        return zeros

    try:
        d['bq'] = pd.qcut(d['beta'].rank(method='first'), 3, labels=False)
    except Exception:
        d['bq'] = 0
    d = d.dropna(subset=['bq'])
    if len(d) < 30:
        return zeros

    longs = []
    shorts = []
    for q in sorted(d['bq'].unique()):
        sub_q = d[d['bq'] == q].sort_values('score', ascending=False)
        k = min(12, len(sub_q) // 4)
        if k < 1:
            continue
        longs.extend(list(sub_q.index[:k]))
        shorts.extend(list(sub_q.index[-k:]))

    if len(longs) < 6 or len(shorts) < 6:
        return zeros

    cap = 0.02
    side_total = 0.48

    wl = _inverse_vol_weights(pd.Index(longs), d['vol'], cap, side_total)
    ws = _inverse_vol_weights(pd.Index(shorts), d['vol'], cap, side_total)

    weights = zeros.copy()
    weights.loc[longs] = wl
    weights.loc[shorts] = -ws

    weights = weights.clip(-cap, cap)
    gross = float(weights.abs().sum())
    if not np.isfinite(gross) or gross > 1.0:
        if gross > 0:
            weights = weights * (1.0 / gross)
        else:
            return zeros

    net = float(weights.sum())
    if not np.isfinite(net) or abs(net) > 0.01:
        return zeros

    return weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
