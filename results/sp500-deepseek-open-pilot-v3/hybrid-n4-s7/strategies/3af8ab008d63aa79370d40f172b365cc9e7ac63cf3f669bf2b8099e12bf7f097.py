import numpy as np
import pandas as pd


def _ret_lag(px, start_lag, end_lag, lag):
    if len(px) < start_lag + lag + 1:
        return pd.Series(np.nan, index=px.columns, dtype=float)
    i0 = -1 - lag - start_lag
    i1 = -1 - lag - end_lag
    return px.iloc[i1] / px.iloc[i0] - 1.0


def _mom_best(px, lag, specs):
    for start, end in specs:
        if len(px) >= start + lag + 1:
            return _ret_lag(px, start, end, lag)
    return pd.Series(np.nan, index=px.columns, dtype=float)


def _z_for(series, idx):
    s = series.reindex(idx).astype(float).replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 2:
        return pd.Series(0.0, index=idx)
    mu = s.mean(skipna=True)
    sd = s.std(ddof=0, skipna=True)
    if (not np.isfinite(sd)) or sd <= 1e-12:
        return pd.Series(0.0, index=idx)
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-3.0, 3.0)


def generate_weights(history, universe):
    u = pd.Index(universe)
    w = pd.Series(0.0, index=u, dtype=float)
    if len(u) == 0:
        return w

    ac = history.get('adjusted_close')
    if not isinstance(ac, pd.DataFrame):
        return w

    px = ac.reindex(columns=u).tail(320).astype(float)
    if px.shape[0] < 80:
        return w

    last = px.iloc[-1]
    rets = px.pct_change()
    vol60 = rets.tail(60).std(ddof=0)

    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame) and len(elig) > 0:
        e = elig.reindex(columns=u).iloc[-1]
        eligible = e.fillna(True).astype(bool)
    else:
        eligible = pd.Series(True, index=u)

    vol_df = history.get('volume')
    has_volume = isinstance(vol_df, pd.DataFrame)
    if has_volume:
        vv = vol_df.reindex(columns=u).reindex(index=px.index)
        dollar = (px * vv).tail(60).median()
    else:
        dollar = pd.Series(np.nan, index=u)

    valid = eligible & last.notna() & np.isfinite(last) & (last > 5.0)
    valid = valid & vol60.notna() & np.isfinite(vol60)
    if has_volume:
        valid = valid & dollar.notna() & (dollar > 0.0)

    if valid.sum() >= 20 and has_volume:
        dv = dollar[valid]
        if dv.notna().sum() >= 20:
            thr = dv.quantile(0.25)
            liq_valid = valid & (dollar >= thr)
            if liq_valid.sum() >= 12:
                valid = liq_valid

    v_idx = valid[valid].index
    if len(v_idx) < 10:
        return w

    mkt_ret = rets.mean(axis=1, skipna=True)
    mkt_price = (1.0 + mkt_ret.fillna(0.0)).cumprod()
    mkt_ma = mkt_price.tail(min(200, len(mkt_price))).mean()
    mkt_trend = float(mkt_price.iloc[-1] / mkt_ma - 1.0) if np.isfinite(mkt_ma) and mkt_ma > 0 else 0.0

    mv = mkt_ret.rolling(20, min_periods=10).std(ddof=0)
    cur_mv = mv.iloc[-1]
    if not np.isfinite(cur_mv) or cur_mv <= 0:
        cur_mv = 0.015
    cur_mv_ann = float(cur_mv * np.sqrt(252.0))
    hist_mv = mv.dropna().tail(200)
    mvol_pct = float((hist_mv <= cur_mv).mean()) if len(hist_mv) >= 20 else 0.5

    ma50 = px.tail(min(50, len(px))).mean()
    breadth = float((last[valid] > ma50[valid]).mean()) if valid.sum() > 0 else 0.5

    if mkt_trend > 0.02 and cur_mv_ann < 0.20 and 0.40 <= breadth <= 0.75:
        rev_w = 0.70
    elif mkt_trend > 0.0 and cur_mv_ann < 0.26 and 0.35 <= breadth <= 0.80:
        rev_w = 0.35
    elif (mkt_trend < -0.04) or (cur_mv_ann > 0.30):
        rev_w = 0.0
    else:
        rev_w = 0.15

    if cur_mv_ann > 0.35:
        gross_target = 0.45
    elif cur_mv_ann > 0.28:
        gross_target = 0.60
    elif cur_mv_ann > 0.22:
        gross_target = 0.80
    else:
        gross_target = 1.0

    if mkt_trend < -0.10 and breadth < 0.35:
        gross_target = min(gross_target, 0.70)
    if mvol_pct > 0.90:
        gross_target = min(gross_target, 0.60)

    ma200 = px.tail(min(200, len(px))).mean()
    trend = last / ma200 - 1.0
    z_trend = _z_for(trend, v_idx)
    vol_for_score = vol60.reindex(u).clip(lower=0.005)

    score_acc = pd.Series(0.0, index=v_idx, dtype=float)
    n_acc = 0
    for lag in (0, 2, 4):
        m12 = _mom_best(px, lag, [(252, 21), (126, 21), (63, 0), (21, 0)])
        m6 = _mom_best(px, lag, [(126, 21), (63, 0), (21, 0)])
        if len(px) >= 22 + lag:
            rev = _ret_lag(px, 21, 0, lag)
        else:
            rev = pd.Series(np.nan, index=u, dtype=float)

        z_m12 = _z_for(m12, v_idx)
        z_m6 = _z_for(m6, v_idx)
        z_rev = _z_for(-rev / vol_for_score, v_idx)

        mom_l = 0.55 * z_m12 + 0.30 * z_m6 + 0.15 * z_trend
        score_l = (1.0 - rev_w) * mom_l + rev_w * z_rev
        score_acc = score_acc + score_l
        n_acc += 1

    if n_acc == 0:
        return w
    score = score_acc / float(n_acc)
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 6:
        return w

    above = (last > ma200).reindex(score.index).fillna(False).astype(bool)
    long_pool = score[above]
    short_pool = score[~above]
    same_pool = False
    if len(long_pool) < 8 or len(short_pool) < 8:
        long_pool = score
        short_pool = score
        same_pool = True

    k = min(40, len(long_pool), len(short_pool))
    if same_pool:
        k = min(k, len(score) // 2)
    if k < 3:
        return w

    longs = list(long_pool.sort_values(ascending=False).index[:k])
    shorts = list(short_pool.sort_values(ascending=True).index[:k])
    if set(longs) & set(shorts):
        k = min(k, len(score) // 2)
        if k < 3:
            return w
        longs = list(score.sort_values(ascending=False).index[:k])
        shorts = list(score.sort_values(ascending=True).index[:k])

    k = min(k, len(longs), len(shorts))
    if k < 1:
        return w
    per_w = min(0.02, 0.5 * gross_target / float(k))

    w.loc[longs[:k]] = per_w
    w.loc[shorts[:k]] = -per_w
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)

    gross = float(w.abs().sum())
    if gross > 1.0 and gross > 0:
        w = w * (1.0 / gross)

    return w