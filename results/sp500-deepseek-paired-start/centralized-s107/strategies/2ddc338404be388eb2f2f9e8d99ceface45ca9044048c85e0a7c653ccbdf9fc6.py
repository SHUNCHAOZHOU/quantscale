import numpy as np
import pandas as pd

_REB = 12
_FORM = 21
_NLONG = 40
_BUF = 2.0
_TARGET_VOL = 0.10
_MIN_HIST = 130
_MAX_W = 0.02
_GROSS_LIMIT = 1.0


def _zeros(tickers):
    return pd.Series(0.0, index=tickers, dtype=float)


def _scores(P, elig_last, form):
    n = P.shape[0]
    if n < form + 2:
        return None
    last = P.iloc[-1]
    ret = last / P.iloc[-1 - form] - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)
    rets = P.pct_change()
    vol = rets.iloc[-120:].std()
    hist_ok = P.iloc[-120:].notna().mean() > 0.8
    valid = last.notna() & (last > 0) & ret.notna() & vol.notna() & (vol > 0) & hist_ok
    if elig_last is not None:
        valid = valid & elig_last.reindex(valid.index).fillna(False).astype(bool)
    valid = valid.fillna(False)
    idx = valid[valid].index
    if len(idx) < 2:
        return None
    score = -ret.reindex(idx)
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    return score


def _select(scores, n_long, prev_long):
    if scores is None or len(scores) < 2:
        return []
    order = scores.sort_values(ascending=False)
    k = len(order)
    n = min(n_long, k - 1)
    if n < 1:
        return []
    kbuf = min(k, int(round(n * _BUF)))
    top = list(order.index[:kbuf])
    prev_set = set(prev_long) if prev_long is not None else set()
    keep = [t for t in top if t in prev_set]
    new = [t for t in top if t not in prev_set]
    return (keep + new)[:n]


def _compute_target_weights(reb, px, elig, tickers):
    P = px.iloc[:reb + 1]
    e_cur = None
    if elig is not None and elig.shape[0] > reb:
        e_cur = elig.iloc[reb].reindex(tickers).fillna(False).astype(bool)
    scores = _scores(P, e_cur, _FORM)
    if scores is None:
        return _zeros(tickers), []

    prev_long = []
    prev_reb = reb - _REB
    if prev_reb >= _MIN_HIST:
        Pprev = px.iloc[:prev_reb + 1]
        e_prev = None
        if elig is not None and elig.shape[0] > prev_reb:
            e_prev = elig.iloc[prev_reb].reindex(tickers).fillna(False).astype(bool)
        s_prev = _scores(Pprev, e_prev, _FORM)
        if s_prev is not None:
            prev_long = _select(s_prev, _NLONG, [])

    longs = _select(scores, _NLONG, prev_long)
    if len(longs) < 1:
        return _zeros(tickers), []

    if e_cur is not None:
        short_ok = e_cur.copy()
    else:
        short_ok = pd.Series(True, index=tickers)
    last = P.iloc[-1]
    short_ok = short_ok & last.notna() & (last > 0)
    short_ok = short_ok & (P.iloc[-20:].notna().mean() > 0.8)
    short_ok = short_ok.fillna(False)
    long_set = set(longs)
    short_universe = [t for t in tickers if short_ok.get(t, False) and t not in long_set]

    n_long = len(longs)
    n_short = len(short_universe)
    if n_long < 1 or n_short < 1:
        return _zeros(tickers), []

    base_L = min(0.4, _MAX_W * n_long, _MAX_W * n_short)
    if base_L <= 0:
        return _zeros(tickers), []

    w = _zeros(tickers)
    w.loc[longs] = base_L / n_long
    w.loc[short_universe] = -base_L / n_short

    rets = px.pct_change().iloc[:reb + 1].iloc[-120:]
    nz = w[w != 0.0]
    if len(nz) > 0:
        sub = rets.reindex(columns=nz.index).fillna(0.0)
        port = sub.mul(nz, axis=1).sum(axis=1)
        sd = float(port.std(ddof=0))
        vol_ann = sd * np.sqrt(252.0) if np.isfinite(sd) else np.nan
    else:
        vol_ann = np.nan

    if np.isfinite(vol_ann) and vol_ann > 1e-8:
        vol_scale = _TARGET_VOL / vol_ann
    else:
        vol_scale = 1.0
    vol_scale = float(np.clip(vol_scale, 0.2, 1.25))

    max_base_w = max(base_L / n_long, base_L / n_short)
    scale_weight = _MAX_W / max_base_w if max_base_w > 0 else 1.0
    scale_gross = _GROSS_LIMIT / (2.0 * base_L) if base_L > 0 else 1.0
    final_scale = min(vol_scale, scale_weight, scale_gross)
    final_scale = max(final_scale, 0.0)

    w = w * final_scale
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return w, longs


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    zeros = _zeros(tickers)

    px = history.get('adjusted_close')
    if px is None:
        return zeros
    px = px.reindex(columns=tickers).astype(float)
    px = px.replace([np.inf, -np.inf], np.nan)
    n = px.shape[0]
    if n < _MIN_HIST:
        return zeros

    elig = history.get('eligible')
    if elig is not None:
        elig = elig.reindex(columns=tickers)

    reb = n - 1 - ((n - 1) % _REB)
    if reb < _MIN_HIST:
        return zeros

    w_reb, longs = _compute_target_weights(reb, px, elig, tickers)
    if w_reb is None or w_reb.abs().sum() == 0:
        return zeros

    if n - 1 == reb:
        return w_reb.reindex(tickers).fillna(0.0)

    t = n - 1
    w = w_reb.copy()
    nz = w[w != 0.0].index
    if len(nz) == 0:
        return zeros

    P_reb = px.iloc[reb]
    P_t = px.iloc[t]
    R = (P_t / P_reb - 1.0).reindex(nz).fillna(0.0)
    R_p = float((w.loc[nz] * R).sum())
    if not np.isfinite(R_p) or abs(1.0 + R_p) < 1e-8:
        return w_reb.reindex(tickers).fillna(0.0)

    w_drift = w.copy()
    w_drift.loc[nz] = w.loc[nz] * (1.0 + R) / (1.0 + R_p)

    net = float(w_drift.sum())
    if abs(net) > 0.01:
        pos = w_drift[w_drift > 0]
        neg = w_drift[w_drift < 0]
        if len(pos) > 0 and len(neg) > 0:
            long_sum = float(pos.sum())
            short_sum = float(-neg.sum())
            L = 0.5 * (long_sum + short_sum)
            if long_sum > 0 and short_sum > 0:
                w_drift.loc[pos.index] = w_drift.loc[pos.index] * (L / long_sum)
                w_drift.loc[neg.index] = w_drift.loc[neg.index] * (L / short_sum)

    max_abs = float(w_drift.abs().max()) if len(w_drift) > 0 else 0.0
    if max_abs > _MAX_W and max_abs > 0:
        w_drift = w_drift * (_MAX_W / max_abs)

    gross = float(w_drift.abs().sum())
    if gross > _GROSS_LIMIT and gross > 0:
        w_drift = w_drift * (_GROSS_LIMIT / gross)

    w_drift = w_drift.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return w_drift.reindex(tickers).fillna(0.0)