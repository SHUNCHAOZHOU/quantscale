import numpy as np
import pandas as pd


def generate_weights(history, universe):
    weights = pd.Series(0.0, index=universe, dtype=float)
    tickers = list(universe)
    if not tickers:
        return weights

    close = history.get('adjusted_close')
    if close is None or close.shape[0] < 30:
        return weights

    cols = [t for t in tickers if t in close.columns]
    if len(cols) < 20:
        return weights

    C = close[cols].astype(float)
    if C.shape[0] < 30:
        return weights

    last = C.iloc[-1]
    r5 = last / C.iloc[-6] - 1.0
    if C.shape[0] >= 22:
        r21 = last / C.iloc[-22] - 1.0
    else:
        r21 = r5

    rets = C.pct_change()
    vol = rets.iloc[-60:].std(min_periods=30)

    volume = history.get('volume')
    if volume is not None:
        V = volume.reindex(columns=cols).astype(float)
        av = V.iloc[-20:].mean()
        dv = av * last
    else:
        dv = pd.Series(np.inf, index=cols)

    mask = last.notna() & r5.notna() & r21.notna() & vol.notna()
    mask &= last > 1.0
    mask &= (vol >= 0.005) & (vol <= 0.08)
    mask &= dv > 1e6

    if mask.sum() < 20:
        return weights

    sub5 = r5[mask]
    sub21 = r21[mask]
    comp = 0.6 * sub5.rank(pct=True) + 0.4 * sub21.rank(pct=True)

    q_lo = comp.quantile(0.10)
    q_hi = comp.quantile(0.90)
    longs = comp[comp <= q_lo].sort_values().index.tolist()
    shorts = comp[comp >= q_hi].sort_values(ascending=False).index.tolist()

    N = min(len(longs), len(shorts), 50)
    if N < 10:
        return weights
    longs = longs[:N]
    shorts = shorts[:N]
    w = min(0.02, 0.5 / N)
    weights[longs] = w
    weights[shorts] = -w
    return weights
