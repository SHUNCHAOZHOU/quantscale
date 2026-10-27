import numpy as np
import pandas as pd


def _pct_rank(s, high_good=True):
    if s is None or len(s) == 0:
        return s
    r = s.rank(pct=True, method='average', ascending=bool(high_good))
    return r.fillna(0.5)


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    close = history.get('adjusted_close')
    if close is None:
        close = history.get('close')
    if close is None or close.empty:
        return zero

    dates = close.index
    rebalance = 21
    p = len(dates) - 1
    anchor = (p // rebalance) * rebalance
    if anchor < 260:
        return zero
    anchor_date = dates[anchor]

    cols = [c for c in universe if c in close.columns]
    if len(cols) < 10:
        return zero

    prices = close.iloc[:anchor + 1][cols].ffill()
    last = prices.iloc[-1]

    elig_df = history.get('eligible')
    if elig_df is not None and not elig_df.empty:
        sub = elig_df.loc[:anchor_date]
        if len(sub) > 0:
            elig = sub.iloc[-1].reindex(cols).fillna(False).astype(bool)
        else:
            elig = pd.Series(True, index=cols)
    else:
        elig = pd.Series(True, index=cols)

    valid = elig & last.notna() & (last > 0.0) & np.isfinite(last)
    if int(valid.sum()) < 10:
        return zero

    ret = prices / prices.shift(1) - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)

    mom12 = prices.shift(21) / prices.shift(252) - 1.0
    mom6 = prices.shift(21) / prices.shift(126) - 1.0

    ma = prices.rolling(200, min_periods=120).mean()
    trend = prices / ma - 1.0

    high252 = prices.rolling(252, min_periods=120).max()
    prox = prices / high252 - 1.0

    f = pd.DataFrame({
        'mom12': mom12.iloc[-1],
        'mom6': mom6.iloc[-1],
        'trend': trend.iloc[-1],
        'prox': prox.iloc[-1]
    }).reindex(cols)
    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.loc[valid]
    if len(f) < 10:
        return zero

    score = (
        0.45 * _pct_rank(f['mom12'], True) +
        0.25 * _pct_rank(f['mom6'], True) +
        0.20 * _pct_rank(f['trend'], True) +
        0.10 * _pct_rank(f['prox'], True)
    )
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    score = score.sort_values(ascending=False)

    n = min(25, len(score) // 2)
    if n < 5:
        return zero

    longs = score.index[:n]
    shorts = score.index[-n:]
    side = min(0.02, 0.5 / float(n))

    out = zero.copy()
    out.loc[longs] = side
    out.loc[shorts] = -side
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out