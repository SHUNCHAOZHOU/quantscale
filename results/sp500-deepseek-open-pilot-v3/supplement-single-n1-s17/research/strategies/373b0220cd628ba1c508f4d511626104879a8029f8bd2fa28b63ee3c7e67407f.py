import numpy as np
import pandas as pd


def _get_df(history, name):
    x = history.get(name)
    if isinstance(x, pd.DataFrame):
        return x
    return None


def generate_weights(history, universe):
    cols = pd.Index(pd.unique(pd.Index(universe).astype(str)))
    if len(cols) == 0:
        return pd.Series(dtype=float)
    zeros = pd.Series(0.0, index=cols, dtype=float)

    ac = _get_df(history, 'adjusted_close')
    if ac is None:
        ac = _get_df(history, 'close')
    if ac is None or ac.empty:
        return zeros

    available = [c for c in cols if c in ac.columns]
    if len(available) < 10:
        return zeros

    ac = ac.reindex(columns=available).sort_index()
    dates = ac.index
    if len(dates) < 70:
        return zeros

    # Rebalance on the last completed calendar month-end. This keeps weights
    # stable through each month and cuts turnover relative to weekly signals.
    months = dates.month.values
    completed = np.zeros(len(dates), dtype=bool)
    completed[:-1] = months[1:] != months[:-1]
    if completed.any():
        sig_date = dates[np.where(completed)[0][-1]]
    else:
        sig_date = dates[-1]

    sub = ac.loc[:sig_date].tail(320)
    if len(sub.index) < 70:
        sub = ac.loc[:sig_date]
    if len(sub.index) < 70:
        return zeros

    cdf = _get_df(history, 'close')
    if cdf is None:
        close = sub.copy()
    else:
        close = cdf.reindex(columns=available).reindex(index=sub.index)
        close = close.where(close.notna(), sub)

    vdf = _get_df(history, 'volume')
    adv20 = pd.Series(np.nan, index=available, dtype=float)
    if vdf is not None:
        vol = vdf.reindex(columns=available).reindex(index=sub.index)
        dv = vol.mul(close, axis=0)
        adv20 = dv.rolling(20, min_periods=10).median().iloc[-1]

    rets = sub.pct_change()
    vol20 = rets.tail(20).std(ddof=0)
    px = sub.iloc[-1]
    pclose = close.iloc[-1]
    r5 = px / sub.iloc[-6] - 1.0
    r63 = px / sub.iloc[-64] - 1.0

    edf = _get_df(history, 'eligible')
    if edf is not None:
        e = edf.reindex(columns=available).reindex(index=sub.index).ffill().iloc[-1]
        e = e.fillna(False).astype(bool)
        if int(e.sum()) < 10:
            e = pd.Series(True, index=available, dtype=bool)
    else:
        e = pd.Series(True, index=available, dtype=bool)

    base = px.notna() & pclose.notna() & r5.notna() & r63.notna() & vol20.notna() & e
    base = base & np.isfinite(px) & np.isfinite(pclose) & np.isfinite(r5)
    base = base & np.isfinite(r63) & np.isfinite(vol20)
    base = base & (pclose > 5.0) & (vol20 > 0.004) & (vol20 < 0.09) & (r5.abs() < 0.60)

    adv = adv20.reindex(available)
    if adv.notna().any():
        base = base & (adv.fillna(0.0) > 5e6)

    if int(base.sum()) < 40:
        base = px.notna() & pclose.notna() & r5.notna() & r63.notna() & vol20.notna() & e
        base = base & np.isfinite(px) & np.isfinite(pclose) & np.isfinite(r5)
        base = base & np.isfinite(r63) & np.isfinite(vol20)
        base = base & (pclose > 3.0) & (vol20 > 0.003) & (vol20 < 0.12) & (r5.abs() < 0.80)
        if adv.notna().any():
            base = base & (adv.fillna(0.0) > 1e6)

    cand = list(base[base].index)
    if len(cand) < 10:
        return zeros

    r5c = r5.loc[cand].astype(float)
    r63c = r63.loc[cand].astype(float)
    vc = vol20.loc[cand].astype(float)

    # Cross-sectional ranks: long oversold, positive intermediate trend, lower vol.
    rev = 1.0 - r5c.rank(method='average', pct=True)
    mom = r63c.rank(method='average', pct=True)
    lowvol = 1.0 - vc.rank(method='average', pct=True)
    score = 0.70 * rev + 0.20 * mom + 0.10 * lowvol
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 10:
        return zeros

    n_side = min(30, len(score) // 2)
    if n_side < 5:
        return zeros

    ordered = score.sort_values(ascending=True)
    shorts = list(ordered.index[:n_side])
    longs = list(ordered.index[-n_side:])

    side_gross = min(0.475, 0.02 * n_side)
    if side_gross <= 0.0:
        return zeros
    w = min(0.02, side_gross / n_side)

    weights = zeros.copy()
    for c in longs:
        if c in weights.index:
            weights.loc[c] = w
    for c in shorts:
        if c in weights.index:
            weights.loc[c] = -w

    net = float(weights.sum())
    if abs(net) > 1e-12:
        nz = weights.index[weights != 0.0]
        if len(nz) > 0:
            weights.loc[nz] -= net / len(nz)

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    weights = weights.clip(-0.02, 0.02)
    gross = float(weights.abs().sum())
    if gross > 1.0:
        weights = weights * (1.0 / gross)
    return weights.fillna(0.0)