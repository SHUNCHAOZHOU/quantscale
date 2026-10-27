import numpy as np
import pandas as pd

_STATE = {'month': None, 'weights': None, 'universe': None}


def _rank_centered(s):
    s = pd.Series(s, dtype=float).replace([np.inf, -np.inf], np.nan)
    return s.rank(pct=True) - 0.5


def _compute(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)

    close = history.get('adjusted_close')
    if close is None or close.shape[1] == 0:
        close = history.get('close')
    if close is None or close.shape[1] == 0:
        return zeros
    try:
        close = close.reindex(columns=universe).astype(float)
    except Exception:
        return zeros

    openp = history.get('adjusted_open')
    if openp is None or openp.shape[1] == 0:
        openp = history.get('open')
    if openp is None or openp.shape[1] == 0:
        return zeros
    try:
        openp = openp.reindex(columns=universe).reindex(index=close.index).astype(float)
    except Exception:
        return zeros

    n = close.shape[0]
    if n < 90:
        return zeros

    last_c = close.iloc[-1]
    valid = last_c.notna() & (last_c > 0.0) & (close.notna().sum() >= 80)

    prev_close = close.shift(1)
    overnight = openp / prev_close - 1.0
    intraday = close / openp - 1.0
    overnight = overnight.replace([np.inf, -np.inf], np.nan)
    intraday = intraday.replace([np.inf, -np.inf], np.nan)

    window = min(60, n - 1)
    if window < 20:
        return zeros
    on_mean = overnight.tail(window).mean()
    id_mean = intraday.tail(window).mean()

    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    vol = ret.tail(window).std()

    valid = valid & on_mean.notna() & id_mean.notna()
    if int(valid.sum()) < 12:
        return zeros

    volm = history.get('volume')
    if volm is not None:
        try:
            volm = volm.reindex(columns=universe).reindex(index=close.index).astype(float)
            dollar = (close * volm).tail(40).median()
            dollar = dollar.reindex(universe).fillna(0.0)
            liq = dollar > 2e6
            if int(liq.sum()) >= 15:
                valid = valid & liq
        except Exception:
            pass

    el = history.get('eligible')
    if el is not None:
        try:
            el = el.reindex(columns=universe).reindex(index=close.index)
            el_last = el.ffill().iloc[-1]
            flag = el_last.fillna(1.0).astype(float) > 0.5
            if int(flag.sum()) >= 15:
                valid = valid & flag
        except Exception:
            pass

    if int(valid.sum()) < 12:
        return zeros

    z_on = _rank_centered(on_mean)
    z_id = _rank_centered(id_mean)
    z_vol = _rank_centered(vol)

    score = 0.7 * z_on.fillna(0.0) - 0.7 * z_id.fillna(0.0) - 0.2 * z_vol.fillna(0.0)
    score = score.where(z_on.notna() & z_id.notna())
    score = score[valid].replace([np.inf, -np.inf], np.nan).dropna()

    k = len(score)
    if k < 8:
        return zeros

    n_side = min(50, k // 2)
    if n_side < 2:
        return zeros

    order = score.sort_values(kind='mergesort')
    shorts = order.index[:n_side]
    longs = order.index[-n_side:]

    w = min(0.5 / float(n_side), 0.02)
    weights = zeros.copy()
    weights.loc[longs] = w
    weights.loc[shorts] = -w

    gross = float(weights.abs().sum())
    if gross > 1.0 + 1e-12:
        weights = weights * (1.0 / gross)

    weights = weights.clip(-0.02, 0.02)
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
    return weights


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)

    close = history.get('adjusted_close')
    if close is None or close.shape[1] == 0:
        close = history.get('close')
    if close is None:
        return zeros

    try:
        last_date = close.index[-1]
        month_key = (last_date.year, last_date.month)
    except Exception:
        month_key = None

    if month_key is not None and _STATE.get('month') == month_key and _STATE.get('weights') is not None:
        cached = _STATE.get('universe')
        if cached is not None and cached.equals(universe):
            return _STATE['weights'].reindex(universe).fillna(0.0).astype(float).clip(-0.02, 0.02)

    w = _compute(history, universe)
    if month_key is not None:
        _STATE['month'] = month_key
        _STATE['weights'] = w.copy()
        _STATE['universe'] = universe.copy()
    return w
