import numpy as np
import pandas as pd

def _market_beta(ret_window):
    if ret_window is None or ret_window.empty:
        return pd.Series(dtype=float)
    arr = ret_window.to_numpy(dtype=float)
    if arr.shape[0] < 60:
        return pd.Series(np.nan, index=ret_window.columns)
    m = np.nanmean(arr, axis=1)
    good = np.isfinite(m)
    if int(good.sum()) < 60:
        return pd.Series(np.nan, index=ret_window.columns)
    arr = arr[good]
    m = m[good]
    m_cent = m - np.mean(m)
    var_m = float(np.mean(m_cent ** 2))
    if not np.isfinite(var_m) or var_m <= 0.0:
        return pd.Series(np.nan, index=ret_window.columns)
    col_mean = np.nanmean(arr, axis=0)
    centered = arr - col_mean
    prod = centered * m_cent[:, None]
    count = np.sum(np.isfinite(prod), axis=0)
    cov = np.nansum(prod, axis=0) / np.maximum(count - 1, 1)
    beta = cov / var_m
    beta = np.where(count >= 30, beta, np.nan)
    beta = np.where(np.isfinite(beta), beta, np.nan)
    return pd.Series(beta, index=ret_window.columns)

def _risk_weights(names, vols, gross=0.5, cap=0.02):
    names = pd.Index(names)
    if len(names) == 0:
        return pd.Series(dtype=float)
    v = vols.reindex(names).astype(float).replace([0.0, np.inf, -np.inf], np.nan)
    med = v.median()
    if not np.isfinite(med) or med <= 0.0:
        med = 1.0
    v = v.fillna(med)
    v = v.clip(lower=1e-8)
    inv = 1.0 / v
    w = inv / inv.sum() * gross
    if len(names) * cap <= gross + 1e-12:
        return pd.Series(cap, index=names, dtype=float)
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = (~over) & (w > 0.0)
        if not under.any():
            break
        w[under] = w[under] + excess * w[under] / w[under].sum()
    for _ in range(10):
        diff = gross - float(w.sum())
        if abs(diff) < 1e-10:
            break
        under = w < cap - 1e-12
        if not under.any():
            break
        w[under] = w[under] + diff * w[under] / w[under].sum()
    w = w.clip(upper=cap)
    return w

def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero
    close = history.get('adjusted_close')
    if close is None:
        close = history.get('close')
    if close is None or close.empty:
        return zero
    dates = close.index
    rebalance = 21
    p = len(dates) - 1
    anchor = (p // rebalance) * rebalance
    if anchor < 130:
        return zero
    anchor_date = dates[anchor]
    cols = [c for c in universe if c in close.columns]
    if len(cols) < 20:
        return zero
    prices = close.iloc[:anchor + 1][cols].ffill()
    last = prices.iloc[-1]
    elig_df = history.get('eligible')
    if elig_df is not None and not elig_df.empty:
        sub = elig_df.loc[:anchor_date]
        if len(sub) > 0:
            elig = sub.iloc[-1].reindex(cols).fillna(False).astype(bool)
        else:
            elig = pd.Series(True, index=cols)
    else:
        elig = pd.Series(True, index=cols)
    valid = elig & last.notna() & (last > 0.0) & np.isfinite(last)
    if int(valid.sum()) < 20:
        return zero
    ret = prices / prices.shift(1) - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)
    vol = ret.iloc[-60:].std()
    max20 = ret.iloc[-20:].max()
    rev = prices.iloc[-1] / prices.iloc[-22] - 1.0
    ma_win = min(200, len(prices))
    ma = prices.iloc[-ma_win:].mean()
    trend = prices.iloc[-1] / ma - 1.0
    mom = prices.iloc[-22] / prices.iloc[-127] - 1.0
    beta = _market_beta(ret.iloc[-252:])
    f = pd.DataFrame({
        'vol': vol,
        'beta': beta,
        'max20': max20,
        'rev': rev,
        'trend': trend,
        'mom': mom
    }).reindex(cols)
    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.loc[valid]
    f = f[f['vol'].notna()]
    if len(f) < 20:
        return zero
    def rank_low(s):
        return s.rank(pct=True, method='average', ascending=False).fillna(0.5)
    def rank_high(s):
        return s.rank(pct=True, method='average', ascending=True).fillna(0.5)
    score = (0.25 * rank_low(f['vol']) +
             0.20 * rank_low(f['beta']) +
             0.15 * rank_low(f['max20']) +
             0.20 * rank_high(f['trend']) +
             0.12 * rank_low(f['rev']) +
             0.08 * rank_high(f['mom']))
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    score = score.sort_values(ascending=False)
    n = min(40, len(score) // 2)
    if n < 5:
        return zero
    longs = score.index[:n]
    shorts = score.index[-n:]
    lw = _risk_weights(longs, f['vol'], gross=0.5, cap=0.02)
    sw = _risk_weights(shorts, f['vol'], gross=0.5, cap=0.02)
    out = zero.copy()
    out.loc[lw.index] = lw.values
    out.loc[sw.index] = -sw.values
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out