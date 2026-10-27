import numpy as np
import pandas as pd

def generate_weights(history, universe):
    universe = pd.Index([str(t) for t in universe])
    if len(universe) == 0:
        return pd.Series(dtype=float)
    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None:
        return pd.Series(0.0, index=universe)
    ac = ac.reindex(columns=universe)
    dates = ac.index
    if len(dates) == 0:
        return pd.Series(0.0, index=universe)
    latest = dates[-1]
    months = dates.year * 100 + dates.month
    cur = latest.year * 100 + latest.month
    prior = dates[months < cur]
    if len(prior) > 0:
        sig = prior[-1]
    else:
        sig = dates[0]
    pos = dates.get_loc(sig)
    if isinstance(pos, slice):
        pos = pos.start
    pos = int(pos)
    w = ac.iloc[max(0, pos - 260):pos + 1]
    rets = w.pct_change()
    vol = rets.tail(60).std(ddof=1)
    if len(w) >= 253:
        mom = w.iloc[-22] / w.iloc[-253] - 1.0
    elif len(w) >= 130:
        mom = w.iloc[-1] / w.iloc[-126] - 1.0
    else:
        mom = pd.Series(np.nan, index=universe)
    if len(w) >= 6:
        rev = w.iloc[-1] / w.iloc[-6] - 1.0
    else:
        rev = pd.Series(np.nan, index=universe)
    trend = pd.Series(0.0, index=universe)
    if len(w) >= 100:
        trend = w.iloc[-1] / w.tail(100).mean() - 1.0
    volume = history.get('volume')
    if volume is not None:
        volume = volume.reindex(columns=universe)
        dwin = ac.iloc[max(0, pos - 19):pos + 1] * volume.iloc[max(0, pos - 19):pos + 1]
        dv = dwin.median(axis=0)
    else:
        dv = pd.Series(np.inf, index=universe)
    valid = mom.notna() & rev.notna() & vol.notna() & (vol > 0) & (dv > 1e6)
    if int(valid.sum()) < 4:
        return pd.Series(0.0, index=universe)
    sub = pd.DataFrame({'mom': mom[valid], 'rev': rev[valid], 'vol': vol[valid], 'trend': trend[valid]})
    score = (0.40 * sub['mom'].rank(pct=True) +
             0.30 * (-sub['rev']).rank(pct=True) +
             0.20 * (-sub['vol']).rank(pct=True) +
             0.10 * sub['trend'].rank(pct=True))
    ordered = score.sort_values(ascending=False)
    n = min(30, len(ordered) // 2)
    weights = pd.Series(0.0, index=universe)
    if n <= 0:
        return weights
    wgt = min(0.02, 1.0 / (2.0 * n))
    long_names = ordered.index[:n]
    short_names = ordered.index[-n:]
    weights.loc[long_names] = wgt
    weights.loc[short_names] = -wgt
    return weights.astype(float)