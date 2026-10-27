import numpy as np
import pandas as pd

_STATE = {'key': None, 'weights': None, 'last_date': None}


def _compute_target(history, universe):
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
    if n < 150:
        return zeros

    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    mkt = ret.mean(axis=1)

    win_beta = min(120, n - 1)
    R = ret.tail(win_beta)
    m = mkt.tail(win_beta)
    Rf = R.fillna(0.0)
    mf = m.fillna(0.0)
    if len(mf) < 60:
        return zeros

    m_mean = float(mf.mean())
    R_mean = Rf.mean()
    cov = Rf.sub(R_mean, axis=1).mul(mf - m_mean, axis=0).sum(axis=0) / float(len(mf))
    var = float(((mf - m_mean) ** 2).sum()) / float(len(mf))
    if not np.isfinite(var) or var <= 0.0:
        return zeros

    beta = cov / var
    beta = beta.reindex(universe).replace([np.inf, -np.inf], np.nan)

    win_res = min(60, n - 1)
    Rres = ret.tail(win_res).fillna(0.0)
    mres = mkt.tail(win_res).fillna(0.0)
    resid = Rres.values - np.outer(mres.values, beta.values)
    resid = pd.DataFrame(resid, index=Rres.index, columns=Rres.columns)
    resid = resid.replace([np.inf, -np.inf], np.nan)

    cum = resid.tail(21).sum(min_count=15)
    vol = resid.tail(60).std()
    vol = vol.replace(0.0, np.nan)
    score = -cum / vol
    score = score.replace([np.inf, -np.inf], np.nan)

    last_px = close.iloc[-1]
    valid = last_px.notna() & (last_px > 3.0) & (close.notna().sum() >= 100)

    volm = history.get('volume')
    if volm is not None:
        try:
            volm = volm.reindex(columns=universe).reindex(index=close.index).astype(float)
            dollar = (close * volm).tail(60).median()
            dollar = dollar.reindex(universe).fillna(0.0)
            liq = dollar > 5e6
            if int(liq.sum()) >= 30:
                valid = valid & liq
        except Exception:
            pass

    el = history.get('eligible')
    if el is not None:
        try:
            el = el.reindex(columns=universe).reindex(index=close.index)
            el_last = el.ffill().iloc[-1]
            flag = el_last.fillna(1.0).astype(float) > 0.5
            if int(flag.sum()) >= 30:
                valid = valid & flag
        except Exception:
            pass

    cand = valid & score.notna() & beta.notna() & (vol > 0.0)
    if int(cand.sum()) < 30:
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
        q = min(10, len(s) // 2)
        if q < 1:
            continue
        shorts.extend(s.index[:q].tolist())
        longs.extend(s.index[-q:].tolist())

    n_side = len(longs)
    if n_side < 25 or len(shorts) != n_side:
        return zeros

    w = min(0.5 / float(n_side), 0.02)
    weights = zeros.copy()
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
    weights = weights.clip(-0.02, 0.02)

    gross = float(weights.abs().sum())
    if gross > 1.0 + 1e-12:
        weights = weights * (1.0 / gross)

    return weights.astype(float)


def _repair_cached(w, universe):
    w = w.reindex(universe).fillna(0.0).astype(float)
    net = float(w.sum())
    if abs(net) > 1e-12:
        active = w != 0.0
        n_active = int(active.sum())
        if n_active > 0:
            w.loc[active] = w.loc[active] - net / float(n_active)
    w = w.clip(-0.02, 0.02)
    gross = float(w.abs().sum())
    if gross > 1.0 + 1e-12:
        w = w * (1.0 / gross)
    return w.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)

    close = history.get('adjusted_close')
    if close is None or close.shape[1] == 0:
        close = history.get('close')
    if close is None or close.shape[1] == 0:
        return zeros

    last_date = None
    key = None
    try:
        last_date = close.index[-1]
        key = (last_date.year, last_date.month)
    except Exception:
        key = None

    if key is not None and _STATE.get('key') == key and _STATE.get('weights') is not None:
        stored = _STATE.get('last_date')
        if stored is None or last_date is None or last_date >= stored:
            today = zeros.reindex(universe).fillna(0.0) if len(zeros) else zeros
            cached = _STATE['weights']
            w = _repair_cached(cached, universe)
            return w.clip(-0.02, 0.02)

    w = _compute_target(history, universe)
    if key is not None:
        _STATE['key'] = key
        _STATE['weights'] = w.copy()
        _STATE['last_date'] = last_date
    return w
