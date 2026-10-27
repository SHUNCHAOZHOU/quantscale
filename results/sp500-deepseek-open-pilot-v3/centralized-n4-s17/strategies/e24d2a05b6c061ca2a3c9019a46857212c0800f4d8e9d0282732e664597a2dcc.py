import numpy as np
import pandas as pd

def _zscore(s):
    s = s.astype(float)
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - mu) / sd

def generate_weights(history, universe):
    idx_univ = pd.Index(list(universe))
    zeros = pd.Series(0.0, index=idx_univ, dtype=float)
    if len(idx_univ) == 0:
        return zeros
    adj = history.get('adjusted_close')
    if adj is None:
        adj = history.get('close')
    if adj is None or adj.empty:
        return zeros
    cols = [c for c in idx_univ if c in adj.columns]
    if len(cols) == 0:
        return zeros
    ac = adj[cols].tail(320)
    if len(ac) < 30:
        return zeros
    rets = ac.pct_change(fill_method=None)
    rets = rets.replace([np.inf, -np.inf], np.nan)
    mkt = rets.mean(axis=1)
    idx_level = (1.0 + mkt.fillna(0.0)).cumprod()
    ma = idx_level.rolling(window=200, min_periods=150).mean()
    risk_off = True
    if len(idx_level) > 0 and np.isfinite(idx_level.iloc[-1]) and np.isfinite(ma.iloc[-1]):
        risk_off = bool(idx_level.iloc[-1] < ma.iloc[-1])
    vol_win = rets.tail(60)
    count = vol_win.count()
    vol = vol_win.std(ddof=0) * np.sqrt(252.0)
    vol[count < 30] = np.nan
    r120 = rets.tail(120)
    m120 = mkt.tail(120)
    R = r120.to_numpy(dtype=float)
    M = m120.to_numpy(dtype=float)
    beta_vals = []
    for j in range(R.shape[1]):
        x = R[:, j]
        mask = np.isfinite(x) & np.isfinite(M)
        if int(mask.sum()) >= 40:
            xv = x[mask]
            yv = M[mask]
            vy = float(np.var(yv, ddof=0))
            if vy > 0.0:
                beta_vals.append(float(np.cov(xv, yv, ddof=0)[0, 1] / vy))
            else:
                beta_vals.append(np.nan)
        else:
            beta_vals.append(np.nan)
    beta = pd.Series(beta_vals, index=cols, dtype=float)
    if len(ac) > 20:
        ret20 = ac.iloc[-1] / ac.iloc[-21] - 1.0
    else:
        ret20 = pd.Series(np.nan, index=cols, dtype=float)
    valid = vol.notna() & ret20.notna() & np.isfinite(ret20)
    beta_filled = beta.copy()
    if beta_filled.notna().any():
        beta_filled = beta_filled.fillna(beta_filled.median())
    else:
        beta_filled = beta_filled.fillna(1.0)
    vol_z = _zscore(vol)
    beta_z = _zscore(beta_filled)
    ret_z = _zscore(ret20)
    if risk_off:
        score = -(0.6 * vol_z + 0.4 * beta_z)
    else:
        score = -0.7 * ret_z - 0.3 * vol_z
    score = score.where(valid, np.nan)
    n = int(score.notna().sum())
    if n < 2:
        return zeros
    K = min(40, n // 2)
    if K < 1:
        return zeros
    ranked = score.dropna().sort_values(ascending=False, kind='mergesort')
    longs = ranked.index[:K]
    shorts = ranked.index[-K:]
    w = min(0.02, 0.5 / float(K))
    weights = zeros.copy()
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    return weights