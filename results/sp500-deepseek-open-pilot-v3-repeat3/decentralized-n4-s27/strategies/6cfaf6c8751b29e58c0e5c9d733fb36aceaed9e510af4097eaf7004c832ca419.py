import numpy as np
import pandas as pd

def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)
    zeros = pd.Series(0.0, index=universe)
    if 'adjusted_close' in history:
        ac = history['adjusted_close']
    elif 'close' in history:
        ac = history['close']
    else:
        return zeros
    if ac is None or ac.shape[0] == 0:
        return zeros
    dates = ac.index
    wd = np.asarray(dates.weekday)
    fri = np.where(wd == 4)[0]
    if len(fri) > 0:
        reb_idx = int(fri[-1])
    else:
        reb_idx = len(dates) - 1
    if reb_idx < 60:
        return zeros
    start_idx = max(0, reb_idx - 300)
    ac_s = ac.iloc[start_idx:reb_idx+1]
    if ac_s.shape[0] < 60:
        return zeros
    ret = ac_s.pct_change()
    n = ac_s.shape[0]
    if n >= 253:
        p_recent = ac_s.iloc[-22]
        p_base = ac_s.iloc[-253]
    else:
        p_recent = ac_s.iloc[-1]
        p_base = ac_s.iloc[0]
    mom = p_recent / p_base - 1.0
    vol = ret.tail(60).std().replace(0.0, np.nan)
    score = mom / vol
    finite_score = pd.Series(np.isfinite(score.values), index=score.index)
    dv = pd.Series(1.0, index=ac_s.columns)
    if 'volume' in history:
        vdf = history['volume']
        v_s = vdf.iloc[max(0, reb_idx-20):reb_idx+1]
        if 'close' in history:
            cdf = history['close']
            c_s = cdf.iloc[max(0, reb_idx-20):reb_idx+1]
        else:
            c_s = ac_s
        v_s = v_s.reindex(columns=ac_s.columns)
        c_s = c_s.reindex(columns=ac_s.columns)
        dv = (v_s * c_s).median()
    if 'eligible' in history:
        edf = history['eligible']
        reb_date = dates[reb_idx]
        if reb_date in edf.index:
            elig = edf.loc[reb_date]
        else:
            sub = edf.loc[:reb_date]
            if len(sub) > 0:
                elig = sub.iloc[-1]
            else:
                elig = pd.Series(True, index=ac_s.columns)
    else:
        elig = pd.Series(True, index=ac_s.columns)
    elig = elig.reindex(ac_s.columns).fillna(False).astype(bool)
    univ_mask = pd.Series(ac_s.columns.isin(universe), index=ac_s.columns)
    last_price = ac_s.iloc[-1]
    valid = elig & univ_mask & finite_score & (dv > 0) & (last_price > 0)
    scores = score[valid]
    if len(scores) < 6:
        return zeros
    N = min(25, len(scores) // 2)
    if N < 1:
        return zeros
    top = list(scores.nlargest(N).index)
    bottom = list(scores.nsmallest(N).index)
    top_set = set(top)
    bottom = [t for t in bottom if t not in top_set]
    N2 = min(len(top), len(bottom))
    if N2 < 1:
        return zeros
    w = zeros.copy()
    for t in top[:N2]:
        w[t] = 0.02
    for t in bottom[:N2]:
        w[t] = -0.02
    return w