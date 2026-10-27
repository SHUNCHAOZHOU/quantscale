import numpy as np
import pandas as pd

def _pct_rank(s):
    return s.rank(pct=True, method='average').fillna(0.5)

def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    close_df = history.get('adjusted_close')
    if close_df is None or close_df.empty:
        close_df = history.get('close')
    if close_df is None or close_df.empty:
        return zero

    dates = close_df.index
    n_obs = len(dates)
    if n_obs < 80:
        return zero

    rebalance = 21
    pos = n_obs - 1
    anchor = (pos // rebalance) * rebalance
    if anchor < 60:
        return zero
    anchor_date = dates[anchor]

    cols = [c for c in universe if c in close_df.columns]
    if len(cols) < 6:
        return zero

    prices = close_df.iloc[:anchor + 1][cols].ffill()
    last = prices.iloc[-1]

    eligible = pd.Series(True, index=cols, dtype=bool)
    elig_df = history.get('eligible')
    if elig_df is not None and not elig_df.empty:
        sub = elig_df.loc[:anchor_date]
        if len(sub) > 0:
            er = sub.iloc[-1].reindex(cols)
            eligible = er.fillna(False).astype(bool)

    valid = eligible & last.notna() & np.isfinite(last) & (last > 0.0)
    if int(valid.sum()) < 6:
        return zero

    ret = prices / prices.shift(1) - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)

    p21 = prices.shift(21).iloc[-1]
    p126 = prices.shift(126).iloc[-1]
    p252 = prices.shift(252).iloc[-1]

    rev1m = -(last / p21 - 1.0)
    mom12 = p21 / p252 - 1.0
    mom6 = p21 / p126 - 1.0
    mom = mom12.where(np.isfinite(mom12), mom6)
    trend = last / prices.rolling(200, min_periods=100).mean().iloc[-1] - 1.0
    vol = ret.rolling(60, min_periods=30).std().iloc[-1]

    f = pd.DataFrame({
        'rev': rev1m,
        'mom': mom,
        'trend': trend,
        'vol': vol
    }).reindex(cols)
    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.loc[valid]
    if len(f) < 6:
        return zero

    r_mom = _pct_rank(f['mom'])
    r_trend = _pct_rank(f['trend'])
    r_rev = _pct_rank(f['rev'])
    r_lowvol = _pct_rank(-f['vol'])

    mkt_ret = ret.mean(axis=1)
    mkt_vol = mkt_ret.iloc[-60:].std() * np.sqrt(252.0)
    if not np.isfinite(mkt_vol) or mkt_vol <= 0:
        mkt_vol = 0.15
    vol_scale = np.clip(0.10 / mkt_vol, 0.40, 1.00)

    cs_disp = ret.std(axis=1).replace([np.inf, -np.inf], np.nan)
    if cs_disp.notna().sum() > 60:
        cur_disp = cs_disp.iloc[-5:].mean()
        hist_disp = cs_disp.iloc[-252:].median()
        if np.isfinite(cur_disp) and np.isfinite(hist_disp) and hist_disp > 0 and (cur_disp / hist_disp) > 1.15:
            w_rev = 0.18
            w_mom = 0.56
            vol_scale = vol_scale * 0.85
        else:
            w_rev = 0.06
            w_mom = 0.68
    else:
        w_rev = 0.10
        w_mom = 0.64
    w_trend = 0.20
    w_low = 0.06

    score = (w_mom * r_mom + w_trend * r_trend +
             w_rev * r_rev + w_low * r_lowvol)
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    score = score.sort_values(ascending=False)

    n_side = min(50, len(score) // 2)
    if n_side < 3:
        return zero
    longs = score.index[:n_side]
    shorts = score.index[-n_side:]

    gross = float(np.clip(vol_scale, 0.0, 1.0))
    if gross <= 0.0:
        return zero
    side = min(0.02, 0.5 * gross / float(n_side))

    out = zero.copy()
    out.loc[longs] = side
    out.loc[shorts] = -side
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out