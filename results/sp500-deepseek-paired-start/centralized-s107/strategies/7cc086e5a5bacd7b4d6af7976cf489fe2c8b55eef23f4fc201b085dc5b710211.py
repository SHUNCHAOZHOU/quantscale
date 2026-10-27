import numpy as np
import pandas as pd

_REB = 5          # weekly rebalance (trading days)
_R21 = 21         # reversal formation
_R252 = 252       # momentum formation
_NSIDE = 40       # names per leg
_BUF = 1.6        # no-trade band multiplier on leg size
_TARGET_VOL = 0.12


def _ranks(s):
    out = pd.Series(np.nan, index=s.index, dtype=float)
    d = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) == 0:
        return out
    out.loc[d.index] = d.rank(method='average', pct=True) - 0.5
    return out


def _scores(P, elig_last):
    """Residual 1-month reversal score (reversal orthogonal to 12-1 momentum)."""
    if P.shape[0] < _R252 + 2:
        return None
    last = P.iloc[-1]
    with np.errstate(all='ignore'):
        r21 = last / P.iloc[-1 - _R21] - 1.0
        mom = P.iloc[-1 - _R21] / P.iloc[-1 - _R252] - 1.0
        rets = P.pct_change()
        vol = rets.iloc[-60:].std()
    r21 = r21.replace([np.inf, -np.inf], np.nan)
    mom = mom.replace([np.inf, -np.inf], np.nan)
    ok = (last.notna() & (last > 0) & r21.notna() & mom.notna()
          & vol.notna() & (vol > 0))
    ok = ok & (P.iloc[-_R252:].notna().mean() > 0.9)
    if elig_last is not None:
        ok = ok & elig_last.reindex(ok.index).fillna(False).astype(bool)
    ok = ok.fillna(False)
    idx = ok[ok].index
    if len(idx) < 20:
        return None
    rev = _ranks(-r21.reindex(idx))
    mo = _ranks(mom.reindex(idx))
    x = mo.values.astype(float)
    y = rev.values.astype(float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return None
    xm = x[m]
    ym = y[m]
    vx = float(np.var(xm))
    beta = float(np.cov(xm, ym)[0, 1] / vx) if vx > 1e-12 else 0.0
    beta = float(np.clip(beta, -1.5, 1.5))
    resid = (rev - beta * mo).replace([np.inf, -np.inf], np.nan).dropna()
    return resid


def _select(scores, n_side, prev_long, prev_short):
    order = scores.sort_values(ascending=False)
    k = len(order)
    nl = min(n_side, k // 2)
    if nl < 1:
        return [], []
    kbuf = min(k, int(round(nl * _BUF)))
    top = list(order.index[:kbuf])
    bot = list(order.index[-kbuf:][::-1])
    prev_l = set(prev_long)
    prev_s = set(prev_short)
    lc = [t for t in top if t in prev_l] + [t for t in top if t not in prev_l]
    sc = [t for t in bot if t in prev_s] + [t for t in bot if t not in prev_s]
    return lc[:nl], sc[:nl]


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    if len(tickers) == 0:
        return pd.Series(dtype=float)
    zeros = pd.Series(0.0, index=tickers, dtype=float)

    px = history.get('adjusted_close')
    if px is None:
        return zeros
    px = px.reindex(columns=tickers).astype(float)
    px = px.replace([np.inf, -np.inf], np.nan)
    n = px.shape[0]
    if n < _R252 + 3:
        return zeros

    reb = n - 1 - ((n - 1) % _REB)
    if reb < _R252 + 1:
        return zeros

    elig = history.get('eligible')
    e = None
    if elig is not None:
        e = elig.reindex(columns=tickers)

    P = px.iloc[:reb + 1]
    e_cur = None
    if e is not None and e.shape[0] > reb:
        e_cur = e.iloc[reb].fillna(False).astype(bool)
    scores = _scores(P, e_cur)
    if scores is None:
        return zeros

    prev_long, prev_short = [], []
    prev_reb = reb - _REB
    if prev_reb >= _R252 + 1:
        Pprev = px.iloc[:prev_reb + 1]
        e_prev = None
        if e is not None and e.shape[0] > prev_reb:
            e_prev = e.iloc[prev_reb].fillna(False).astype(bool)
        s_prev = _scores(Pprev, e_prev)
        if s_prev is not None:
            prev_long, prev_short = _select(s_prev, _NSIDE, [], [])

    longs, shorts = _select(scores, _NSIDE, prev_long, prev_short)
    nl = len(longs)
    ns = len(shorts)
    if nl < 1 or ns < 1:
        return zeros

    base_w = min(0.02, 0.5 / float(max(nl, ns)))

    w = pd.Series(0.0, index=tickers, dtype=float)
    w.loc[list(longs)] = base_w
    w.loc[list(shorts)] = -base_w

    # volatility targeting on the proposed book (60-day window)
    scale = 1.0
    try:
        rets = P.pct_change().iloc[-60:]
        nz = w[w != 0.0]
        sub = rets.reindex(columns=nz.index).fillna(0.0)
        port = sub.mul(nz, axis=1).sum(axis=1)
        sd = float(port.std(ddof=0))
        if np.isfinite(sd) and sd > 1e-8:
            vol_ann = sd * np.sqrt(252.0)
            scale = float(np.clip(_TARGET_VOL / vol_ann, 0.35, 1.0))
    except Exception:
        scale = 1.0

    w = w * scale
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return w.reindex(tickers).fillna(0.0)