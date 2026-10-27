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

    ac_full = adj[cols]
    n_full = len(ac_full)
    if n_full < 30:
        return zeros

    # Rebalance every 5 trading days: use data up to the last row whose
    # integer position is a multiple of 5.  This holds weights for 5 days.
    rebal_pos = ((n_full - 1) // 5) * 5
    ac = ac_full.iloc[:rebal_pos + 1].tail(320)
    if len(ac) < 30:
        return zeros

    rets = ac / ac.shift(1) - 1.0
    rets = rets.replace([np.inf, -np.inf], np.nan)

    mkt = rets.mean(axis=1)
    idx_level = (1.0 + mkt.fillna(0.0)).cumprod()
    ma = idx_level.rolling(window=200, min_periods=150).mean()

    risk_off = True
    if len(idx_level) > 0 and np.isfinite(idx_level.iloc[-1]) and np.isfinite(ma.iloc[-1]):
        risk_off = bool(idx_level.iloc[-1] < ma.iloc[-1])

    # 60-day realized volatility
    vol_win = rets.tail(60)
    count = vol_win.count()
    vol60 = vol_win.std(ddof=0) * np.sqrt(252.0)
    vol60[count < 30] = np.nan

    # 20-day realized volatility (for risk-off quality momentum)
    vol_win20 = rets.tail(20)
    count20 = vol_win20.count()
    vol20 = vol_win20.std(ddof=0) * np.sqrt(252.0)
    vol20[count20 < 10] = np.nan

    # 120-day beta vs equal-weight market
    r120 = rets.tail(120)
    m120 = mkt.tail(120)
    R = r120.to_numpy(dtype=float)
    M = m120.to_numpy(dtype=float)
    n_cols = R.shape[1]
    beta_vals = np.full(n_cols, np.nan)
    for j in range(n_cols):
        x = R[:, j]
        mask = np.isfinite(x) & np.isfinite(M)
        if int(mask.sum()) >= 40:
            xv = x[mask]
            yv = M[mask]
            vy = float(np.var(yv, ddof=0))
            if vy > 0.0:
                beta_vals[j] = float(np.cov(xv, yv, ddof=0)[0, 1] / vy)
    beta = pd.Series(beta_vals, index=cols, dtype=float)

    # Momentum
    if len(ac) >= 121:
        mom120 = ac.iloc[-1] / ac.iloc[-121] - 1.0
    else:
        mom120 = pd.Series(np.nan, index=cols)
    if len(ac) >= 61:
        mom60 = ac.iloc[-1] / ac.iloc[-61] - 1.0
    else:
        mom60 = pd.Series(np.nan, index=cols)

    mom120 = mom120.replace([np.inf, -np.inf], np.nan)
    mom60 = mom60.replace([np.inf, -np.inf], np.nan)

    beta_filled = beta.copy()
    if beta_filled.notna().any():
        beta_filled = beta_filled.fillna(beta_filled.median())
    else:
        beta_filled = beta_filled.fillna(1.0)

    z_mom120 = _zscore(mom120)
    z_mom60 = _zscore(mom60)
    z_beta = _zscore(beta_filled)
    z_vol60 = _zscore(vol60)
    z_vol20 = _zscore(vol20)

    if risk_off:
        # Quality momentum: high long-term momentum, low short-term vol
        score = 0.6 * z_mom120 + 0.4 * z_mom60 - 0.3 * z_vol20
        valid = mom120.notna() & mom60.notna() & vol20.notna()
        gross_scale = 0.5
    else:
        # High-beta / high-vol / high-momentum tilt
        score = 0.4 * z_mom120 + 0.3 * z_mom60 + 0.5 * z_beta + 0.4 * z_vol60
        valid = mom120.notna() & mom60.notna() & vol60.notna() & beta_filled.notna()
        gross_scale = 1.0

    score = score.where(valid, np.nan)
    n = int(score.notna().sum())
    if n < 2:
        return zeros

    K = min(30, n // 2)
    if K < 1:
        return zeros

    ranked = score.dropna().sort_values(ascending=False, kind='mergesort')
    longs = ranked.index[:K]
    shorts = ranked.index[-K:]

    w = min(0.02, 0.5 / float(K)) * gross_scale
    weights = zeros.copy()
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    return weights
