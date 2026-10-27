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
    zero = pd.Series(0.0, index=tickers, dtype=float)

    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None or not isinstance(ac, pd.DataFrame) or len(ac.index) == 0:
        return zero

    if not ac.index.is_monotonic_increasing:
        ac = ac.sort_index()

    n = len(ac.index)
    if n < 260:
        return zero

    reb = ((n - 1) // 10) * 10
    if reb < 252:
        return zero

    cols = [c for c in tickers if c in ac.columns]
    if len(cols) < 20:
        return zero

    start = max(0, reb - 300)
    px = ac.iloc[start:reb + 1][cols].astype(float).copy()
    px = px.replace([np.inf, -np.inf], np.nan)
    if px.shape[0] < 260:
        return zero

    last = px.iloc[-1]
    valid = last.notna() & (last > 0)
    valid = valid & (px.notna().sum(axis=0) >= 240)

    elig = history.get('eligible')
    if elig is not None and isinstance(elig, pd.DataFrame):
        if not elig.index.is_monotonic_increasing:
            elig = elig.sort_index()
        ecols = [c for c in cols if c in elig.columns]
        if len(ecols) > 0:
            elig_al = elig.reindex(ac.index)
            e_last = elig_al.iloc[reb][ecols].reindex(cols).fillna(False).astype(bool)
            valid = valid & e_last

    vol = history.get('volume')
    if vol is not None and isinstance(vol, pd.DataFrame):
        if not vol.index.is_monotonic_increasing:
            vol = vol.sort_index()
        vcols = [c for c in cols if c in vol.columns]
        if len(vcols) > 0:
            vol_al = vol.reindex(ac.index)
            vsub = vol_al.iloc[start:reb + 1][vcols].astype(float).reindex(columns=cols)
            dollar = vsub * px
            liq = dollar.tail(20).median(axis=0)
            liq = liq.replace([np.inf, -np.inf], np.nan)
            valid = valid & liq.notna() & (liq > 0)
            if liq.notna().sum() >= 10:
                q = liq.quantile(0.30)
                valid = valid & (liq >= q)

    rets = px.pct_change()
    rets = rets.replace([np.inf, -np.inf], np.nan)
    vol20 = rets.tail(20).std()
    max20 = rets.tail(20).max()

    def _ret(lag):
        if px.shape[0] <= lag:
            return pd.Series(np.nan, index=cols)
        return px.iloc[-1] / px.iloc[-1 - lag] - 1.0

    r252 = _ret(252)
    r126 = _ret(126)
    r21 = _ret(21)
    mom12 = (1.0 + r252) / (1.0 + r21) - 1.0

    high252 = px.tail(252).max()
    prox = px.iloc[-1] / high252

    valid = valid & mom12.notna() & r126.notna() & prox.notna() & vol20.notna() & max20.notna()
    idx = valid[valid].index
    if len(idx) < 12:
        return zero

    rr12 = _rank_pct(mom12.reindex(idx))
    rr6 = _rank_pct(r126.reindex(idx))
    rprox = _rank_pct(prox.reindex(idx))
    rvol = _rank_pct(vol20.reindex(idx))
    rmax = _rank_pct(max20.reindex(idx))

    score = 0.45 * rr12 + 0.25 * rr6 + 0.25 * rprox - 0.15 * rvol - 0.15 * rmax
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 12:
        return zero

    n_side = min(30, len(score) // 2)
    if n_side < 1:
        return zero

    order = score.sort_values(ascending=False)
    longs = order.index[:n_side]
    shorts = order.index[-n_side:]

    w = min(0.02, 0.5 / float(n_side))
    weights = pd.Series(0.0, index=tickers, dtype=float)
    weights.loc[longs] = w
    weights.loc[shorts] = -w

    gross = float(weights.abs().sum())
    if gross > 1.0 + 1e-12:
        weights = weights * (1.0 / gross)

    weights = weights.clip(-0.02, 0.02).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights.reindex(tickers).fillna(0.0)