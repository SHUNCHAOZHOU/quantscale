import numpy as np
import pandas as pd


def generate_weights(history, universe):
    u = pd.Index(universe, dtype=object)
    m = len(u)
    if m == 0:
        return pd.Series(dtype=float)

    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if not isinstance(ac, pd.DataFrame):
        return pd.Series(0.0, index=u, dtype=float)

    ac = ac.reindex(columns=u).tail(400)
    n = ac.shape[0]
    if n < 200:
        return pd.Series(0.0, index=u, dtype=float)

    close = ac.to_numpy(dtype=float, copy=False)
    if close.ndim != 2 or close.shape[0] != n or close.shape[1] != m:
        return pd.Series(0.0, index=u, dtype=float)

    with np.errstate(divide='ignore', invalid='ignore'):
        rets = close[1:] / close[:-1] - 1.0
    rets = np.vstack([np.full((1, m), np.nan), rets])
    rets = np.where(np.isfinite(rets), np.clip(rets, -0.5, 0.5), np.nan)

    dollar = None
    vol = history.get('volume')
    if isinstance(vol, pd.DataFrame):
        vol = vol.reindex(columns=u).tail(400)
        if vol.shape[0] == n:
            v = vol.to_numpy(dtype=float, copy=False)
            with np.errstate(invalid='ignore'):
                dv = close * v
            dv_df = pd.DataFrame(dv).rolling(20, min_periods=10).median()
            dollar = dv_df.iloc[-1].to_numpy()

    elig = np.ones(m, dtype=bool)
    e = history.get('eligible')
    if isinstance(e, pd.DataFrame):
        e = e.reindex(columns=u).tail(1)
        if e.shape[0] == 1:
            arr = e.iloc[0].to_numpy()
            elig = np.where(pd.isna(arr), True, arr).astype(bool)

    df_close = pd.DataFrame(close, columns=u)
    mom = df_close.shift(21) / df_close.shift(126) - 1.0
    ma200 = df_close.rolling(200, min_periods=150).mean()
    trend = df_close / ma200 - 1.0
    vol60 = df_close.pct_change().rolling(60, min_periods=30).std()
    vol60 = vol60.clip(lower=0.005)
    score = mom / vol60
    score_smooth = score.tail(5).mean(axis=0, skipna=True)

    last = df_close.iloc[-1].to_numpy()
    trend_last = trend.iloc[-1].to_numpy()
    score_arr = score_smooth.to_numpy()

    mkt_ret = np.nanmean(rets, axis=1)
    mkt_vol = pd.Series(mkt_ret).rolling(20, min_periods=10).std()
    cur_mkt_vol = mkt_vol.iloc[-1]
    hist_mkt_vol = mkt_vol.iloc[:-1].dropna().tail(250)
    if len(hist_mkt_vol) >= 20 and np.isfinite(cur_mkt_vol):
        mkt_vol_pct = float((hist_mkt_vol <= cur_mkt_vol).mean())
    else:
        mkt_vol_pct = 0.5

    valid = elig & np.isfinite(last) & (last > 5) & np.isfinite(score_arr) & np.isfinite(trend_last)
    if dollar is not None:
        valid = valid & np.isfinite(dollar) & (dollar > 0)
        if valid.sum() >= 20:
            thr = np.nanpercentile(dollar[valid], 25)
            valid = valid & (dollar >= thr)

    long_ok = valid & (trend_last > 0) & (score_arr > 0)
    short_ok = valid & (trend_last < 0) & (score_arr < 0)
    if long_ok.sum() < 10 or short_ok.sum() < 10:
        long_ok = valid & (score_arr > 0)
        short_ok = valid & (score_arr < 0)
    if long_ok.sum() < 5 or short_ok.sum() < 5:
        return pd.Series(0.0, index=u, dtype=float)

    long_idx = np.where(long_ok)[0]
    short_idx = np.where(short_ok)[0]
    long_sorted = long_idx[np.argsort(-score_arr[long_idx])]
    short_sorted = short_idx[np.argsort(score_arr[short_idx])]

    K = min(40, len(long_sorted), len(short_sorted))
    if K < 1:
        return pd.Series(0.0, index=u, dtype=float)

    gross_target = 1.0
    if mkt_vol_pct >= 0.90:
        gross_target = 0.5
    elif mkt_vol_pct >= 0.75:
        gross_target = 0.75

    per_side = 0.5 * gross_target
    per_w = min(0.02, per_side / K)

    weights = np.zeros(m, dtype=float)
    weights[long_sorted[:K]] = per_w
    weights[short_sorted[:K]] = -per_w

    long_sum = weights[weights > 0].sum()
    short_sum = -weights[weights < 0].sum()
    target = min(long_sum, short_sum)
    if target > 0:
        if long_sum > 0:
            weights[weights > 0] *= target / long_sum
        if short_sum > 0:
            weights[weights < 0] *= target / short_sum

    weights = np.clip(weights, -0.02, 0.02)
    gross = np.abs(weights).sum()
    if gross > 1.0:
        weights *= 1.0 / gross

    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(weights, index=u, dtype=float)