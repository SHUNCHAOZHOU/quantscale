import pandas as pd
import numpy as np


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zeros

    ac = history.get('adjusted_close')
    if ac is None or len(ac) == 0:
        ac = history.get('close')
    if ac is None or len(ac) == 0:
        return zeros
    ac = ac.reindex(columns=universe)
    if ac.shape[0] < 2:
        return zeros

    close = history.get('close')
    if close is None or len(close) == 0:
        close = ac
    else:
        close = close.reindex(columns=universe)

    vol = history.get('volume')
    if vol is None or len(vol) == 0:
        vol = pd.DataFrame(1.0, index=ac.index, columns=universe)
    else:
        vol = vol.reindex(columns=universe)

    eligible = pd.Series(True, index=universe)
    el = history.get('eligible')
    if el is not None and len(el) > 0:
        last_el = el.reindex(columns=universe).iloc[-1]
        if last_el.notna().any():
            eligible = last_el.fillna(False).astype(bool)

    last_price = close.iloc[-1]
    valid = eligible & last_price.notna() & ac.iloc[-1].notna()
    valid = valid.fillna(False).astype(bool)

    liq_n = min(20, len(close))
    dv = (close * vol).iloc[-liq_n:]
    med_dv = dv.median()
    if med_dv.notna().any():
        dv_thresh = med_dv.quantile(0.2)
        valid = valid & (med_dv >= dv_thresh) & (med_dv > 0)

    valid = valid & (last_price > 1.0)

    lb = min(11, len(ac))
    if lb < 2:
        return zeros
    p0 = ac.iloc[-lb]
    p1 = ac.iloc[-1]
    ret = p1 / p0 - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)
    valid = valid & ret.notna()

    cnt = ac.iloc[-lb:].notna().sum()
    valid = valid & (cnt >= max(3, int(lb * 0.8)))
    valid = valid.fillna(False).astype(bool)
    if valid.sum() < 4:
        return zeros

    candidates = universe[valid]
    r = ret.loc[candidates]
    order = r.sort_values(kind='mergesort')

    n_side = min(40, len(order) // 2)
    if n_side < 1:
        return zeros

    long_tickers = order.index[:n_side]
    short_tickers = order.index[-n_side:]

    w = min(0.02, 0.95 / (2.0 * n_side))
    weights = pd.Series(0.0, index=universe, dtype=float)
    weights.loc[long_tickers] = w
    weights.loc[short_tickers] = -w
    return weights
