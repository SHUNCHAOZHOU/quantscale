import numpy as np
import pandas as pd


def generate_weights(history, universe):
    weights = pd.Series(0.0, index=universe, dtype=float)
    tickers = list(universe)
    if len(tickers) == 0:
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
    r21 = last / C.iloc[-22] - 1.0

    rets = C / C.shift(1) - 1.0
    tail = rets.tail(60)
    vol = tail.std()
    valid_vol_days = tail.notna().sum()

    volume = history.get('volume')
    if volume is not None:
        V = volume.reindex(index=C.index, columns=cols).astype(float)
        avg_vol = V.tail(20).mean()
        dollar_vol = avg_vol * last
    else:
        dollar_vol = pd.Series(np.inf, index=cols, dtype=float)

    mask = last.notna() & r5.notna() & r21.notna() & vol.notna() & dollar_vol.notna()
    mask &= valid_vol_days >= 30
    mask &= last > 1.0
    mask &= (vol >= 0.005) & (vol <= 0.08)
    mask &= dollar_vol > 1e6

    eligible = history.get('eligible')
    if eligible is not None:
        try:
            e = eligible.reindex(index=C.index, columns=cols).iloc[-1]
            mask &= e.fillna(False).astype(bool)
        except Exception:
            pass

    if int(mask.sum()) < 20:
        return weights

    sub5 = r5[mask]
    sub21 = r21[mask]
    comp = 0.6 * sub5.rank(pct=True) + 0.4 * sub21.rank(pct=True)

    q_lo = comp.quantile(0.10)
    q_hi = comp.quantile(0.90)
    long_candidates = comp[comp <= q_lo].sort_values().index.tolist()
    short_candidates = comp[comp >= q_hi].sort_values(ascending=False).index.tolist()

    n_names = min(len(long_candidates), len(short_candidates), 50, len(comp) // 2)
    if n_names < 10:
        return weights

    ordered = comp.sort_values()
    longs = ordered.index[:n_names].tolist()
    shorts = ordered.index[-n_names:].tolist()[::-1]

    w = min(0.02, 0.5 / n_names)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    return weights
