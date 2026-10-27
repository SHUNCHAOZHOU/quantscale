import numpy as np
import pandas as pd

def _reindex(df, tickers):
    try:
        return df.reindex(columns=tickers)
    except Exception:
        return None

def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=tickers, dtype=float)
    if not isinstance(history, dict):
        return zero
    ac = history.get('adjusted_close')
    if not isinstance(ac, pd.DataFrame) or ac.empty:
        ac = history.get('close')
    if not isinstance(ac, pd.DataFrame) or ac.empty:
        return zero
    ac = _reindex(ac, tickers)
    if ac is None or ac.shape[1] != n:
        return zero
    dates = ac.index
    if len(dates) < 260:
        return zero
    last_pos = len(dates) - 1
    cur = dates[last_pos]
    sig_pos = last_pos
    if isinstance(cur, pd.Timestamp):
        mask = (dates.year < cur.year) | ((dates.year == cur.year) & (dates.month < cur.month))
        pos = np.where(mask)[0]
        if len(pos) > 0:
            sig_pos = int(pos[-1])
    if sig_pos < 130:
        return zero
    p = ac.iloc[:sig_pos + 1]
    last = p.iloc[-1].values.astype(float)
    valid = np.isfinite(last) & (last > 1.0)
    cnt = p.notna().sum(axis=0).values
    valid &= cnt >= 130
    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame):
        try:
            e = elig.reindex(index=dates, columns=tickers).iloc[:sig_pos + 1]
            if len(e) > 0:
                le = e.iloc[-1].values
                ev = np.array([(x is True) or (x == 1) for x in le], dtype=bool)
                valid &= ev
        except Exception:
            pass
    vol_df = history.get('volume')
    if isinstance(vol_df, pd.DataFrame):
        try:
            v = vol_df.reindex(index=dates, columns=tickers).iloc[:sig_pos + 1]
            raw = history.get('close')
            if isinstance(raw, pd.DataFrame):
                c = raw.reindex(index=dates, columns=tickers).iloc[:sig_pos + 1]
            else:
                c = p
            m = min(60, len(p))
            dv = (c.iloc[-m:] * v.iloc[-m:]).median(axis=0).values
            dvv = np.where(np.isfinite(dv) & (dv > 0), dv, np.nan)
            if np.isfinite(dvv[valid]).sum() >= 10:
                thresh = max(250000.0, float(np.nanpercentile(dvv[valid], 20)))
                valid &= np.isfinite(dvv) & (dvv >= thresh)
        except Exception:
            pass
    rets = p.pct_change(fill_method=None)
    r = rets.iloc[-min(252, len(rets)):]
    R = r.values.astype(float)
    mkt = np.nanmean(R, axis=1)
    mask = np.isfinite(R) & np.isfinite(mkt)[:, None]
    cntb = mask.sum(axis=0)
    R0 = np.where(mask, R, 0.0)
    M0 = np.where(mask, mkt[:, None], 0.0)
    meanR = R0.sum(axis=0) / np.maximum(cntb, 1)
    meanM = M0.sum(axis=0) / np.maximum(cntb, 1)
    cov = ((R0 - meanR) * (M0 - meanM)).sum(axis=0) / np.maximum(cntb, 1)
    varM = ((M0 - meanM) ** 2).sum(axis=0) / np.maximum(cntb, 1)
    beta = np.where((cntb >= 60) & (varM > 1e-12), cov / varM, np.nan)
    vret = rets.iloc[-min(60, len(rets)):].values.astype(float)
    vol = np.nanstd(vret, axis=0) * np.sqrt(252.0)
    vol = np.where(np.isfinite(vol) & (vol > 1e-6), vol, np.nan)
    def skip_ret(look):
        off = 22
        if len(p) > look + off:
            end = p.iloc[-off].values.astype(float)
            start = p.iloc[-off - look].values.astype(float)
            out = end / start - 1.0
            return np.where(np.isfinite(out), out, np.nan)
        return np.full(n, np.nan)
    lt1 = skip_ret(252)
    lt2 = skip_ret(504)
    lt3 = skip_ret(756)
    vals = np.vstack([lt1, lt2, lt3])
    cntv = np.isfinite(vals).sum(axis=0)
    lt = np.where(cntv > 0, np.nansum(vals, axis=0) / np.maximum(cntv, 1), np.nan)
    valid = valid & np.isfinite(lt) & np.isfinite(beta) & np.isfinite(vol)
    if valid.sum() < 8:
        return zero
    def zscore(x, maskv):
        y = x[maskv]
        mu = np.nanmean(y)
        sd = np.nanstd(y)
        if not np.isfinite(sd) or sd < 1e-12:
            return np.zeros(n, dtype=float)
        return (x - mu) / sd
    lt_z = zscore(lt, valid)
    vol_z = zscore(vol, valid)
    score = -lt_z - 0.20 * vol_z
    score = np.where(valid & np.isfinite(score), score, -np.inf)
    qs = np.nanpercentile(beta[valid], [20, 40, 60, 80])
    bucket = np.digitize(beta, qs)
    long_idx = []
    short_idx = []
    for b in range(5):
        idx_b = np.where(valid & (bucket == b))[0]
        if len(idx_b) < 2:
            continue
        sc = score[idx_b]
        order = np.argsort(sc)
        k = min(8, len(idx_b) // 2)
        if k < 1:
            continue
        long_idx.extend(idx_b[order[-k:]].tolist())
        short_idx.extend(idx_b[order[:k]].tolist())
    total = len(long_idx)
    if total < 4 or len(short_idx) != total:
        return zero
    w = min(0.02, 0.5 / float(total))
    weights = np.zeros(n, dtype=float)
    weights[long_idx] = w
    weights[short_idx] = -w
    weights = np.clip(weights, -0.02, 0.02)
    return pd.Series(weights, index=tickers).fillna(0.0)