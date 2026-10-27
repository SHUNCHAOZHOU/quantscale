import numpy as np
import pandas as pd

_STATE = {'key': None, 'weights': None, 'universe': None}


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

    n = close.shape[0]
    if n < 130:
        return zeros

    ret = close.pct_change()
    ret = ret.replace([np.inf, -np.inf], np.nan)
    mkt = ret.mean(axis=1)

    win = 120
    R = ret.iloc[:-1].tail(win).fillna(0.0)
    m = mkt.iloc[:-1].tail(win).fillna(0.0)
    if len(m) < 60:
        return zeros

    m_mean = float(m.mean())
    R_mean = R.mean()
    cov = ((R.sub(R_mean, axis=1)).mul(m - m_mean, axis=0)).sum(axis=0) / float(len(m))
    var = float(((m - m_mean) ** 2).sum()) / float(len(m))
    if not np.isfinite(var) or var <= 0.0:
        return zeros

    beta = cov / var
    beta = beta.reindex(universe).replace([np.inf, -np.inf], np.nan)

    R60 = ret.tail(60)
    m60 = mkt.tail(60)
    resid = R60.values - np.outer(m60.values, beta.values)
    resid = pd.DataFrame(resid, index=R60.index, columns=R60.columns)
    resid = resid.replace([np.inf, -np.inf], np.nan)

    cum = resid.tail(10).sum(min_count=6)
    vol = resid.tail(60).std()
    vol = vol.replace(0.0, np.nan)
    score = -cum / vol
    score = score.replace([np.inf, -np.inf], np.nan)

    last_px = close.iloc[-1]
    valid = last_px.notna() & (last_px > 3.0) & (close.notna().sum() >= 80)

    volm = history.get('volume')
    if volm is not None:
        try:
            volm = volm.reindex(columns=universe).reindex(index=close.index).astype(float)
            dollar = (close * volm).tail(60).median()
            dollar = dollar.reindex(universe).fillna(0.0)
            liq = dollar > 5e6
            if int(liq.sum()) >= 20:
                valid = valid & liq
        except Exception:
            pass

    el = history.get('eligible')
    if el is not None:
        try:
            el = el.reindex(columns=universe).reindex(index=close.index)
            el_last = el.ffill().iloc[-1]
            flag = el_last.fillna(1.0).astype(float) > 0.5
            if int(flag.sum()) >= 20:
                valid = valid & flag
        except Exception:
            pass

    cand = valid & score.notna() & beta.notna() & (vol > 0.0)
    if int(cand.sum()) < 20:
        return zeros

    b_rank = beta[cand].rank(pct=True)
    b_id = pd.Series(-1, index=universe, dtype=int)
    b_id.loc[cand] = np.minimum((b_rank * 5.0).astype(int), 4)

    longs = []
    shorts = []
    for b in range(5):
        idx = cand & (b_id == b)
        if int(idx.sum()) < 4:
            continue
        s = score[idx].dropna().sort_values(kind='mergesort')
        if len(s) < 4:
            continue
        q = min(5, len(s) // 2)
        if q < 1:
            continue
        shorts.extend(s.index[:q].tolist())
        longs.extend(s.index[-q:].tolist())

    n_side = len(longs)
    if n_side < 6 or len(shorts) != n_side:
        return zeros

    w = min(0.5 / float(n_side), 0.02)
    weights = zeros.copy()
    weights.loc[longs] = w
    weights.loc[shorts] = -w

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
    gross = float(weights.abs().sum())
    if gross > 1.0 + 1e-12:
        weights = weights * (1.0 / gross)

    net = float(weights.sum())
    if abs(net) > 1e-12:
        weights = weights - net / float(len(universe))
        weights = weights.clip(-0.02, 0.02)

    return weights.astype(float)


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)

    close = history.get('adjusted_close')
    if close is None or close.shape[1] == 0:
        close = history.get('close')
    if close is None:
        return zeros

    key = None
    try:
        last_date = close.index[-1]
        iso = last_date.isocalendar()
        key = (iso[0], iso[1] // 2)
    except Exception:
        key = None

    if key is not None and _STATE.get('key') == key and _STATE.get('weights') is not None:
        cached = _STATE.get('universe')
        if cached is not None and cached.equals(universe):
            return _STATE['weights'].reindex(universe).fillna(0.0).astype(float).clip(-0.02, 0.02)

    w = _compute(history, universe)
    if key is not None:
        _STATE['key'] = key
        _STATE['weights'] = w.copy()
        _STATE['universe'] = universe.copy()

    return w
