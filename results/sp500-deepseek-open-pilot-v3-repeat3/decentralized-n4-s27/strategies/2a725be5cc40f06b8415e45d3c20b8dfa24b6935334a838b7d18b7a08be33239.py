import numpy as np
import pandas as pd


def generate_weights(history, universe):
    tickers = pd.Index(universe)
    zero = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return zero

    ac_all = history.get('adjusted_close')
    if ac_all is None:
        ac_all = history.get('close')
    if ac_all is None or not isinstance(ac_all, pd.DataFrame) or ac_all.shape[0] == 0:
        return zero

    dates = ac_all.index
    current = dates[-1]
    try:
        periods = dates.to_period('M')
        curr_p = current.to_period('M')
        prev = np.where(periods < curr_p)[0]
    except Exception:
        prev = np.array([], dtype=int)
    if len(prev) > 0:
        pos = int(prev[-1])
    else:
        wd = np.asarray(dates.weekday)
        fri = np.where(wd == 4)[0]
        pos = int(fri[-1]) if len(fri) > 0 else len(dates) - 1
    if pos < 25:
        return zero

    start = max(0, pos - 320)
    c = ac_all.iloc[start:pos + 1].reindex(columns=tickers)
    if c.shape[0] < 30:
        return zero
    ref = dates[pos]

    elig = pd.Series(True, index=tickers)
    edf = history.get('eligible')
    if isinstance(edf, pd.DataFrame) and edf.shape[0] > 0:
        if ref in edf.index:
            e = edf.loc[ref].reindex(tickers)
        else:
            sub = edf.loc[:ref].reindex(columns=tickers)
            if len(sub) > 0:
                e = sub.iloc[-1]
            else:
                e = pd.Series(True, index=tickers)
        elig = e.fillna(False).astype(bool)

    last = pd.to_numeric(c.iloc[-1], errors='coerce')
    valid = last.notna() & (last > 1.0) & elig
    if valid.sum() < 10:
        return zero

    rets = c.pct_change()
    vol_cnt = rets.tail(60).notna().sum()
    vol = rets.tail(60).std()
    vol = vol.where(vol_cnt >= 20)
    vol = vol.where(vol > 1e-8)

    n = c.shape[0]
    if n >= 252:
        mom = c.iloc[-22] / c.iloc[-252] - 1.0
    elif n >= 126:
        mom = c.iloc[-22] / c.iloc[-126] - 1.0
    else:
        mom = c.iloc[-22] / c.iloc[0] - 1.0
    rev = c.iloc[-1] / c.iloc[-22] - 1.0

    liq = pd.Series(1.0, index=tickers)
    vdf = history.get('volume')
    has_volume = isinstance(vdf, pd.DataFrame) and vdf.shape[0] > 0
    if has_volume:
        v = vdf.reindex(index=c.index, columns=tickers)
        cdf = history.get('close')
        if isinstance(cdf, pd.DataFrame) and cdf.shape[0] > 0:
            raw = cdf.reindex(index=c.index, columns=tickers)
        else:
            raw = c
        dollar = raw * v
        liq = dollar.tail(60).median()
        liq = liq.where(liq > 0)

    mom = pd.to_numeric(mom, errors='coerce').reindex(tickers)
    rev = pd.to_numeric(rev, errors='coerce').reindex(tickers)
    vol = pd.to_numeric(vol, errors='coerce').reindex(tickers)
    liq = pd.to_numeric(liq, errors='coerce').reindex(tickers)

    valid = valid & mom.notna() & rev.notna() & vol.notna() & liq.notna()
    valid = valid & np.isfinite(mom) & np.isfinite(rev) & np.isfinite(vol)
    if has_volume:
        valid = valid & (liq > 250000.0)
    if valid.sum() < 10:
        return zero

    rm = (mom / vol).where(valid).rank(pct=True, method='average')
    rr = rev.where(valid).rank(pct=True, method='average')
    rv = vol.where(valid).rank(pct=True, method='average')
    rl = liq.where(valid).rank(pct=True, method='average')
    score = 0.75 * rm - 0.15 * rr - 0.05 * rv + 0.05 * rl
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 10:
        return zero

    n_side = min(25, len(score) // 2)
    if n_side < 3:
        return zero

    order = score.sort_values(ascending=False).index
    top = order[:n_side]
    bottom = order[-n_side:]

    w = min(0.02, 0.5 / float(n_side))
    weights = pd.Series(0.0, index=tickers, dtype=float)
    weights.loc[top] = w
    weights.loc[bottom] = -w
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights