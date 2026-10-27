import numpy as np
import pandas as pd

def _anchor_position(n, rebalance=21, min_obs=60):
    if n <= 0:
        return None
    p = n - 1
    a = (p // rebalance) * rebalance
    if a < min_obs:
        return None
    return a

def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero
    close_df = history.get('adjusted_close')
    if close_df is None:
        close_df = history.get('close')
    if close_df is None or close_df.empty:
        return zero
    dates = close_df.index
    apos = _anchor_position(len(dates))
    if apos is None:
        return zero
    anchor_date = dates[apos]
    cols = [c for c in universe if c in close_df.columns]
    if len(cols) < 2:
        return zero
    prices = close_df.iloc[:apos + 1][cols].ffill()
    last = prices.iloc[-1]
    e = pd.Series(True, index=cols)
    elig_df = history.get('eligible')
    if elig_df is not None and not elig_df.empty:
        sub = elig_df.loc[:anchor_date]
        if len(sub) > 0:
            e_raw = sub.iloc[-1].reindex(cols)
            e = e_raw.fillna(False).astype(bool)
    valid = e & last.notna() & (last > 0.0) & (last < np.inf)
    if int(valid.sum()) < 4:
        return zero
    ret = prices / prices.shift(1) - 1.0
    rev1m = -(prices / prices.shift(21) - 1.0)
    mom12 = prices.shift(21) / prices.shift(252) - 1.0
    mom6 = prices.shift(21) / prices.shift(126) - 1.0
    mom = mom12.where(mom12.notna(), mom6)
    trend = prices / prices.rolling(200, min_periods=100).mean() - 1.0
    vol = ret.rolling(60, min_periods=30).std()
    f = pd.DataFrame({
        'rev': rev1m.iloc[-1],
        'mom': mom.iloc[-1],
        'trend': trend.iloc[-1],
        'vol': vol.iloc[-1]
    }).reindex(cols)
    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.loc[valid]
    if len(f) < 4:
        return zero
    def rk(s):
        return s.rank(pct=True, method='average').fillna(0.5)
    score = (0.45 * rk(f['rev']) + 0.35 * rk(f['mom']) +
             0.15 * rk(f['trend']) + 0.05 * rk(-f['vol']))
    score = score.sort_values(ascending=False)
    n = min(50, len(score) // 2)
    if n < 1:
        return zero
    longs = score.index[:n]
    shorts = score.index[-n:]
    side = min(0.02, 0.5 / float(n))
    out = zero.copy()
    out.loc[longs] = side
    out.loc[shorts] = -side
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out