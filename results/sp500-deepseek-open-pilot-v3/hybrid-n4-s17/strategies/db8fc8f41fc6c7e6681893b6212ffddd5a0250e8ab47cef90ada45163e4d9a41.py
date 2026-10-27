import numpy as np
import pandas as pd


def generate_weights(history, universe):
    u = pd.Index(universe)
    zeros = pd.Series(0.0, index=u, dtype=float)
    if len(u) == 0:
        return zeros

    ac = history.get('adjusted_close')
    if ac is None or ac.empty:
        ac = history.get('close')
    if ac is None or ac.empty:
        return zeros

    ac = ac.reindex(columns=u)
    if ac.shape[0] < 60:
        return zeros

    idx = ac.index
    current_month = (idx[-1].year, idx[-1].month)
    prev_dates = [d for d in idx if (d.year, d.month) < current_month]
    if not prev_dates:
        return zeros

    reb_date = prev_dates[-1]
    try:
        pos = idx.get_loc(reb_date)
    except Exception:
        return zeros

    ac_hist = ac.iloc[:pos + 1]
    if len(ac_hist) < 60:
        return zeros

    ret = ac_hist.pct_change()
    if len(ret) < 20:
        return zeros

    vol = ret.tail(60).std(ddof=0)
    vol = vol.replace([np.inf, -np.inf], np.nan)
    last_vol = vol.copy()

    mkt = ret.mean(axis=1, skipna=True).fillna(0.0)
    cum = (1.0 + mkt).cumprod()
    if len(cum) >= 200:
        ma = cum.tail(200).mean()
    elif len(cum) >= 100:
        ma = cum.tail(100).mean()
    else:
        return zeros
    risk_on = bool(cum.iloc[-1] > ma)

    elig_df = history.get('eligible')
    if elig_df is not None and not elig_df.empty:
        try:
            cur_elig = elig_df.reindex(columns=u).iloc[-1].fillna(False).astype(bool)
        except Exception:
            cur_elig = pd.Series(True, index=u)
    else:
        cur_elig = pd.Series(True, index=u)

    if elig_df is not None and not elig_df.empty:
        try:
            reb_elig = elig_df.reindex(columns=u).loc[reb_date].fillna(False).astype(bool)
        except Exception:
            reb_elig = pd.Series(True, index=u)
    else:
        reb_elig = pd.Series(True, index=u)

    last_ac_current = ac.iloc[-1]
    last_ac_reb = ac_hist.iloc[-1]

    valid = (
        cur_elig
        & reb_elig
        & last_vol.notna()
        & np.isfinite(last_vol)
        & (last_vol > 0)
        & last_ac_current.notna()
        & last_ac_reb.notna()
    )
    valid_names = last_vol[valid].dropna()
    if len(valid_names) < 6:
        return zeros

    n_side = min(25, len(valid_names) // 2)
    if n_side < 1:
        return zeros

    ordered = valid_names.sort_values(kind='mergesort')
    if risk_on:
        longs = ordered.index[-n_side:]
        shorts = ordered.index[:n_side]
    else:
        longs = ordered.index[:n_side]
        shorts = ordered.index[-n_side:]

    w = min(0.02, 0.25 / float(n_side))
    weights = zeros.copy()
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.clip(-0.02, 0.02).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights
