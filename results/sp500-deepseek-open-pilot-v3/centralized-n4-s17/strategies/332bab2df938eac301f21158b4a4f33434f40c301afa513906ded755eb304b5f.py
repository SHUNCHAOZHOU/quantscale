import numpy as np
import pandas as pd

_STATE = {'last_date': None, 'last_reb': None, 'weights': None}
_MAX_NAMES = 12
_SIDE_GROSS = 0.20
_CAP = 0.02
_REBAL_DAYS = 7

def _map_weights(prev, univ):
    out = pd.Series(0.0, index=univ, dtype=float)
    if prev is None:
        return out
    for t, w in prev.items():
        if t in out.index:
            out.loc[t] = float(w)
    return out

def _reset_if_needed(dt):
    if _STATE['last_date'] is not None and dt < _STATE['last_date']:
        _STATE['last_date'] = None
        _STATE['last_reb'] = None
        _STATE['weights'] = None

def _rank_centered(s):
    return s.rank(pct=True) - 0.5

def _compute_scores(history, cols):
    adj_df = history.get('adjusted_close')
    if adj_df is None:
        adj_df = history.get('close')
    if adj_df is None or len(cols) == 0:
        return None
    use_cols = [c for c in cols if c in adj_df.columns]
    if len(use_cols) == 0:
        return None
    adj = adj_df.loc[:, use_cols].tail(95).apply(pd.to_numeric, errors='coerce')
    if adj.shape[0] < 6:
        return None
    last = adj.iloc[-1]
    ret = adj.pct_change()
    r5 = last / adj.iloc[-6] - 1.0
    if adj.shape[0] >= 22:
        r21 = last / adj.iloc[-22] - 1.0
    else:
        r21 = pd.Series(np.nan, index=use_cols)
    vol = ret.tail(20).std(ddof=0)
    pos = vol[vol > 0]
    med = pos.median() if len(pos) > 0 else np.nan
    if (not np.isfinite(med)) or med <= 0:
        vol = pd.Series(1.0, index=use_cols)
    else:
        vol = vol.fillna(med)
        vol = vol.where(vol > 0, med)
    r5v = r5 / (vol * np.sqrt(5.0))
    r21v = r21 / (vol * np.sqrt(21.0))
    if r21v.notna().sum() >= max(5, len(use_cols) // 3):
        score = -0.65 * _rank_centered(r5v) - 0.35 * _rank_centered(r21v)
    else:
        score = -_rank_centered(r5v)
    valid = last.notna() & r5.notna() & score.replace([np.inf, -np.inf], np.nan).notna()
    elig_df = history.get('eligible')
    if elig_df is not None:
        ec = [c for c in use_cols if c in elig_df.columns]
        if len(ec) > 0:
            e = elig_df.loc[:, ec].tail(1).iloc[-1].reindex(use_cols).fillna(False).astype(bool)
            valid = valid & e
    vol_df = history.get('volume')
    if vol_df is not None:
        vc = [c for c in use_cols if c in vol_df.columns]
        if len(vc) > 0:
            vv = vol_df.loc[:, vc].tail(20).apply(pd.to_numeric, errors='coerce')
            pp = adj.tail(20).reindex(columns=vc)
            dollar = (vv * pp).median().reindex(use_cols)
            valid = valid & (dollar.fillna(0.0) > 1000000.0)
    score = score[valid].dropna()
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    return score

def _select(score, prev_weights):
    n_total = len(score)
    if n_total < 2:
        return [], [], 0.0
    N = min(_MAX_NAMES, n_total // 2)
    if N < 1:
        return [], [], 0.0
    buffer = 2 * N if n_total >= 4 * N else N
    ordered_long = score.sort_values(ascending=False).index.tolist()
    ordered_short = score.sort_values(ascending=True).index.tolist()
    prev_long = set()
    prev_short = set()
    if prev_weights is not None:
        prev_long = set(prev_weights[prev_weights > 0].index)
        prev_short = set(prev_weights[prev_weights < 0].index)
    keep_long = [t for t in ordered_long[:buffer] if t in prev_long][:N]
    longs = keep_long[:]
    long_set = set(longs)
    for t in ordered_long:
        if len(longs) >= N:
            break
        if t not in long_set:
            longs.append(t)
            long_set.add(t)
    keep_short = [t for t in ordered_short[:buffer] if (t in prev_short) and (t not in long_set)][:N]
    shorts = keep_short[:]
    short_set = set(shorts)
    for t in ordered_short:
        if len(shorts) >= N:
            break
        if (t not in short_set) and (t not in long_set):
            shorts.append(t)
            short_set.add(t)
    M = min(len(longs), len(shorts), N)
    if M < 1:
        return [], [], 0.0
    longs = longs[:M]
    shorts = shorts[:M]
    base = min(_CAP, _SIDE_GROSS / float(M))
    return longs, shorts, base

def generate_weights(history, universe):
    univ = pd.Index(universe)
    zero = pd.Series(0.0, index=univ, dtype=float)
    if len(univ) == 0:
        return zero
    ref = None
    for k in ('adjusted_close', 'close', 'open'):
        df = history.get(k)
        if df is not None and len(df.index) > 0:
            ref = df
            break
    if ref is None:
        return zero
    dt = ref.index[-1]
    _reset_if_needed(dt)
    if _STATE['last_date'] is not None and dt == _STATE['last_date'] and _STATE['weights'] is not None:
        return _map_weights(_STATE['weights'], univ)
    due = _STATE['last_reb'] is None
    if not due:
        due = (dt - _STATE['last_reb']).days >= _REBAL_DAYS
    if (not due) and (_STATE['weights'] is not None):
        _STATE['last_date'] = dt
        return _map_weights(_STATE['weights'], univ)
    cols = list(univ)
    score = _compute_scores(history, cols)
    if score is None or len(score) < 2:
        w = zero.copy()
        _STATE['weights'] = w
        _STATE['last_reb'] = dt
        _STATE['last_date'] = dt
        return w
    longs, shorts, base = _select(score, _STATE['weights'])
    w = pd.Series(0.0, index=univ, dtype=float)
    if len(longs) > 0:
        for t in longs:
            if t in w.index:
                w.loc[t] = base
        for t in shorts:
            if t in w.index:
                w.loc[t] = -base
    w = w.clip(-_CAP, _CAP)
    net = float(w.sum())
    if abs(net) > 1e-12:
        nz = w[w != 0.0]
        if len(nz) > 0:
            w.loc[nz.index] -= net / float(len(nz))
    w = w.clip(-_CAP, _CAP)
    if abs(float(w.sum())) > 0.01:
        w = zero.copy()
    gross = float(w.abs().sum())
    if gross > 1.0:
        w = w / gross
    _STATE['weights'] = w.copy()
    _STATE['last_reb'] = dt
    _STATE['last_date'] = dt
    return w