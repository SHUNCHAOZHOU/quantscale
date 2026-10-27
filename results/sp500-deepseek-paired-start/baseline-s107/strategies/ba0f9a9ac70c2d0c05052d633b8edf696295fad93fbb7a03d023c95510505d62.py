import numpy as np
import pandas as pd

def _rank_pct(s):
    s = s.dropna()
    if len(s) == 0:
        return s
    return s.rank(method='average', pct=True) - 0.5

def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    if len(tickers) == 0:
        return pd.Series(dtype=float)

    px = history.get('adjusted_close')
    if px is None:
        return pd.Series(0.0, index=tickers)

    px = px.reindex(columns=tickers).astype(float)
    px = px.replace([np.inf, -np.inf], np.nan)
    if px.shape[0] < 30:
        return pd.Series(0.0, index=tickers)

    last_px = px.iloc[-1]

    ret = px / px.shift(1) - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)

    r10 = px / px.shift(10) - 1.0
    r80 = px / px.shift(80) - 1.0
    vol20 = ret.rolling(20, min_periods=15).std()

    last_r10 = r10.iloc[-1]
    last_r80 = r80.iloc[-1]
    last_vol = vol20.iloc[-1]

    valid = (
        last_px.notna()
        & (last_px > 0)
        & last_r10.notna()
        & last_r80.notna()
        & last_vol.notna()
        & (last_vol > 0)
    )

    elig = history.get('eligible')
    if elig is not None:
        elig = elig.reindex(columns=tickers)
        if elig.shape[0] > 0:
            e_last = elig.iloc[-1]
            valid = valid & e_last.fillna(False).astype(bool)

    valid_t = valid[valid].index
    if len(valid_t) < 6:
        return pd.Series(0.0, index=tickers)

    rr10 = _rank_pct(last_r10.reindex(valid_t))
    rr80 = _rank_pct(last_r80.reindex(valid_t))
    rv = _rank_pct(last_vol.reindex(valid_t))

    score = -1.0 * rr10 + 0.55 * rr80 - 0.20 * rv
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 4:
        return pd.Series(0.0, index=tickers)

    n_side = min(30, len(score) // 2)
    if n_side < 1:
        return pd.Series(0.0, index=tickers)

    order = score.sort_values(ascending=False)
    longs = order.index[:n_side]
    shorts = order.index[-n_side:]

    w = min(0.02, 0.5 / float(n_side))
    weights = pd.Series(0.0, index=tickers)
    weights.loc[longs] = w
    weights.loc[shorts] = -w

    gross = float(weights.abs().sum())
    if gross > 1.0:
        weights = weights * (1.0 / gross)

    return weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)