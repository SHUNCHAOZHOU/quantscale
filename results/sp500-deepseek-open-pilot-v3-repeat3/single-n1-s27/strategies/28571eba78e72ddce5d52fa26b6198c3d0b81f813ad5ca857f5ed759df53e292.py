import numpy as np
import pandas as pd

_STATE = {'weights': None, 'universe': None, 'last_rebal_date': None}
_REBAL_EVERY = 5
_CARRY = 0.4
_THRESH = 0.7
_VOL_WIN = 60
_CUM_WIN = 10
_BETA_WIN = 120
_MAX_W = 0.02


def _clean(w, universe):
    w = pd.Series(w, dtype=float).reindex(universe).fillna(0.0)
    w = w.replace([np.inf, -np.inf], 0.0)
    return w.clip(-_MAX_W, _MAX_W)


def _normalize(w, universe):
    w = _clean(w, universe)
    for _ in range(3):
        act = w != 0.0
        na = int(act.sum())
        if na == 0:
            return pd.Series(0.0, index=universe, dtype=float)
        net = float(w.sum())
        if abs(net) > 1e-10:
            w.loc[act] = w.loc[act] - net / float(na)
        gross = float(w.abs().sum())
        if gross <= 1e-12:
            return pd.Series(0.0, index=universe, dtype=float)
        w = w * (1.0 / gross)
        w = w.clip(-_MAX_W, _MAX_W)
    act = w != 0.0
    na = int(act.sum())
    if na > 0:
        net = float(w.sum())
        if abs(net) > 1e-10:
            w.loc[act] = w.loc[act] - net / float(na)
    return w.clip(-_MAX_W, _MAX_W).fillna(0.0)


def _target(history, universe, prev):
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
    if n < 160:
        return zeros

    ret = close.pct_change()
    ret = ret.where(np.isfinite(ret))
    mkt = ret.mean(axis=1)

    R = ret.tail(_BETA_WIN)
    m = mkt.tail(_BETA_WIN).fillna(0.0)
    if len(m) < 60:
        return zeros
    counts = R.notna().sum()
    Rf = R.fillna(0.0)
    m_mean = float(m.mean())
    dm = m - m_mean
    var = float((dm * dm).sum()) / float(len(m))
    if not np.isfinite(var) or var <= 1e-18:
        return zeros
    Rc = Rf.sub(Rf.mean(), axis=1)
    cov = Rc.mul(dm, axis=0).sum(axis=0) / float(len(m))
    beta = (cov / var).reindex(universe)
    beta = beta.where(counts.reindex(universe) >= 60)

    Rr = ret.tail(_VOL_WIN)
    mr = mkt.tail(_VOL_WIN).fillna(0.0)
    if len(mr) < _VOL_WIN:
        return zeros
    resid = pd.DataFrame(Rr.values - np.outer(mr.values, beta.values),
                         index=Rr.index, columns=Rr.columns)
    resid = resid.where(np.isfinite(resid))

    cum = resid.tail(_CUM_WIN).sum(min_count=8)
    vol = resid.std()
    vol = vol.where(vol > 1e-8)
    score = -cum / vol
    score = score.where(np.isfinite(score))

    last_px = close.iloc[-1]
    valid = last_px.notna() & (last_px > 3.0) & (close.notna().sum() >= 100)
    valid = valid & score.reindex(universe).notna() & beta.notna() & vol.reindex(universe).notna()

    volm = history.get('volume')
    if volm is not None:
        try:
            volm = volm.reindex(columns=universe).reindex(index=close.index).astype(float)
            dollar = (close * volm).tail(60).median()
            liq = dollar.reindex(universe).fillna(0.0) > 5e6
            if int(liq.sum()) >= 30:
                valid = valid & liq
        except Exception:
            pass

    el = history.get('eligible')
    if el is not None:
        try:
            el = el.reindex(columns=universe).reindex(index=close.index)
            flag = el.ffill().iloc[-1].fillna(1.0).astype(float) > 0.5
            if int(flag.sum()) >= 30:
                valid = valid & flag
        except Exception:
            pass

    valid = valid.reindex(universe).fillna(False).astype(bool)
    if int(valid.sum()) < 40:
        return zeros

    s = score.reindex(universe)[valid]
    b = beta.reindex(universe)[valid]
    s_std = float(s.std())
    b_std = float(b.std())
    if not np.isfinite(s_std) or s_std <= 1e-12 or not np.isfinite(b_std) or b_std <= 1e-12:
        return zeros
    zs = ((s - s.mean()) / s_std).clip(-3.0, 3.0)
    zb = ((b - b.mean()) / b_std).clip(-3.0, 3.0)
    rho = float((zs * zb).mean())
    if not np.isfinite(rho):
        rho = 0.0
    z = (zs - rho * zb).clip(-3.0, 3.0)

    raw = np.sign(z) * np.maximum(np.abs(z) - _THRESH, 0.0)
    raw = raw[raw != 0.0]
    if len(raw) < 20:
        return zeros

    w = zeros.copy()
    w.loc[raw.index] = raw.values
    w = _normalize(w, universe)

    if prev is not None:
        p = _clean(prev, universe)
        w = _normalize(_CARRY * p + (1.0 - _CARRY) * w, universe)

    return w


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zeros

    close = history.get('adjusted_close')
    if close is None or close.shape[1] == 0:
        close = history.get('close')
    if close is None or close.shape[1] == 0:
        return zeros

    last_date = None
    try:
        last_date = pd.Timestamp(close.index[-1])
    except Exception:
        last_date = None

    cached = _STATE.get('weights')
    last_reb = _STATE.get('last_rebal_date')

    need = False
    if cached is None or last_reb is None:
        need = True
    elif last_date is None:
        need = True
    else:
        try:
            since = int((close.index > last_reb).sum())
        except Exception:
            since = 10 ** 6
        if since >= _REBAL_EVERY:
            need = True

    if not need:
        w = _normalize(cached, universe)
        return w.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)

    w = _target(history, universe, cached)
    _STATE['weights'] = w.copy()
    _STATE['universe'] = universe.copy()
    _STATE['last_rebal_date'] = last_date
    return w.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
