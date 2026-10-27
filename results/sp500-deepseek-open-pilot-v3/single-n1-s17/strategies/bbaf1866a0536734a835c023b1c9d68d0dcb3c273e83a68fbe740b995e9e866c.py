import numpy as np
import pandas as pd

def generate_weights(history, universe):
    universe = pd.Index(universe)
    w = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return w
    close = history.get('adjusted_close')
    if close is None or close.empty:
        return w
    cols = [c for c in universe if c in close.columns]
    if len(cols) < 10:
        return w
    close = close[cols]
    idx = close.index
    pos = len(idx) - 1
    for i in range(len(idx) - 1, -1, -1):
        if idx[i].dayofweek == 0:
            pos = i
            break
    start = max(0, pos - 299)
    hist_close = close.iloc[start:pos + 1]
    if len(hist_close) < 30:
        return w
    last_date = idx[pos]
    elig_df = history.get('eligible')
    if elig_df is not None:
        e = elig_df.reindex(index=pd.Index([last_date]), columns=cols).iloc[0]
        e = e.fillna(False).astype(bool)
    else:
        e = hist_close.iloc[-1].notna()
    p_last = hist_close.iloc[-1]
    e = e & p_last.notna() & (p_last > 2.0)
    vol_df = history.get('volume')
    if vol_df is not None:
        v = vol_df.reindex(index=hist_close.index, columns=cols)
        dv = (hist_close * v).rolling(20, min_periods=10).mean().iloc[-1]
        e = e & dv.notna() & (dv > 1e6)
    rets = hist_close.pct_change(fill_method=None)
    rets = rets.replace([np.inf, -np.inf], np.nan)
    n = len(hist_close)
    if n >= 252:
        mom = hist_close.iloc[-22] / hist_close.iloc[-252] - 1.0
    elif n >= 60:
        mom = hist_close.iloc[-min(22, n - 1)] / hist_close.iloc[0] - 1.0
    else:
        mom = hist_close.iloc[-1] / hist_close.iloc[0] - 1.0
    if n >= 22:
        rev = hist_close.iloc[-1] / hist_close.iloc[-22] - 1.0
    else:
        rev = hist_close.iloc[-1] / hist_close.iloc[0] - 1.0
    vol = rets.iloc[-60:].std()
    mom = mom.replace([np.inf, -np.inf], np.nan)
    rev = rev.replace([np.inf, -np.inf], np.nan)
    vol = vol.replace([np.inf, -np.inf], np.nan)
    valid = e & mom.notna() & rev.notna() & vol.notna()
    if valid.sum() < 20:
        return w
    mom_r = mom[valid].rank(pct=True)
    rev_r = rev[valid].rank(pct=True)
    vol_r = vol[valid].rank(pct=True)
    score = mom_r - 0.5 * rev_r - 0.1 * vol_r
    k = min(50, len(score) // 2)
    if k < 5:
        return w
    ordered = score.sort_values(ascending=False)
    top = ordered.index[:k]
    bot = ordered.index[-k:]
    mag = min(0.02, 0.5 / float(k))
    w.loc[top] = mag
    w.loc[bot] = -mag
    w = w.fillna(0.0)
    w = w.clip(-0.02, 0.02)
    gross = w.abs().sum()
    if gross > 1.0:
        w = w * (1.0 / gross)
    return w