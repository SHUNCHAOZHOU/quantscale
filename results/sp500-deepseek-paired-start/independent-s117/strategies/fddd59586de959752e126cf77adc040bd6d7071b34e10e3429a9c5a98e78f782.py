import numpy as np
import pandas as pd
import math


def _get_df(history, names):
    for name in names:
        obj = history.get(name)
        if isinstance(obj, pd.DataFrame):
            return obj
    return None


def _zscore(s):
    s = s.replace([np.inf, -np.inf], np.nan)
    x = s.dropna()
    if len(x) < 2:
        return pd.Series(np.nan, index=s.index)
    mu = x.mean()
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)

    price = _get_df(history, ['adjusted_close', 'close'])
    if price is None:
        return pd.Series(0.0, index=universe)

    price = price.reindex(columns=universe)
    if price.shape[0] < 160:
        return pd.Series(0.0, index=universe)

    idx = price.index
    last_date = idx[-1]
    days_since_monday = last_date.dayofweek
    asof_pos = price.shape[0] - 1 - days_since_monday
    if asof_pos < 150:
        asof_pos = price.shape[0] - 1

    p = price.iloc[:asof_pos + 1].tail(400)
    if p.shape[0] < 150:
        return pd.Series(0.0, index=universe)
    asof_date = p.index[-1]

    last = p.iloc[-1]
    cnt = p.notna().sum()
    valid = last.notna() & (last > 2.0) & (cnt >= 120)

    vol_df = _get_df(history, ['volume'])
    if vol_df is not None:
        vol_df = vol_df.reindex(index=idx, columns=universe).loc[:asof_date].tail(60)
        vmed = vol_df.tail(20).median()
        valid = valid & (vmed > 100000)

    elig_df = _get_df(history, ['eligible'])
    if elig_df is not None:
        elig_df = elig_df.reindex(index=idx, columns=universe).loc[:asof_date].tail(1)
        if elig_df.shape[0] > 0:
            e = elig_df.iloc[0]
            valid = valid & e.fillna(True).astype(bool)

    rets = p.pct_change()
    vol = rets.tail(60).std()
    vol = vol.replace([0.0, np.inf, -np.inf], np.nan).clip(lower=0.005)
    valid = valid & vol.notna()

    n = len(p)

    def form(k):
        if n >= k + 1:
            return p.iloc[-1] / p.iloc[-k - 1] - 1.0
        return pd.Series(np.nan, index=universe)

    ret126 = form(126)
    ret252 = form(252)

    valid = valid & ret126.notna() & (ret126 < 3.0) & (ret126 > -0.85)
    if valid.sum() < 4:
        return pd.Series(0.0, index=universe)

    z126 = ret126 / (vol * math.sqrt(126.0))
    z252 = ret252 / (vol * math.sqrt(252.0))
    z126s = _zscore(z126.where(valid))
    z252s = _zscore(z252.where(valid))

    if z252s.notna().sum() >= max(10, int(0.5 * valid.sum())):
        mom = 0.65 * z126s + 0.35 * z252s.fillna(0.0)
    else:
        mom = z126s

    volz = _zscore(vol.where(valid))
    score = 0.78 * mom - 0.22 * volz
    score = score.where(valid).replace([np.inf, -np.inf], np.nan).dropna()

    if len(score) < 10:
        alt = (ret126 / vol).where(valid)
        score = alt.replace([np.inf, -np.inf], np.nan).dropna()

    if len(score) < 2:
        return pd.Series(0.0, index=universe)

    m = len(score)
    n_side = min(50, m // 2)
    if n_side < 1:
        n_side = 1

    srt = score.sort_values(ascending=False)
    long_names = srt.index[:n_side]
    short_names = srt.index[-n_side:]

    w = min(0.5 / n_side, 0.02)
    weights = pd.Series(0.0, index=universe)
    weights.loc[long_names] += w
    weights.loc[short_names] -= w

    net = weights.sum()
    if abs(net) > 1e-12:
        active = weights.index[weights != 0.0]
        if len(active) > 0:
            weights.loc[active] -= net / len(active)

    weights = weights.clip(-0.02, 0.02)
    gross = weights.abs().sum()
    if gross > 1.0:
        weights = weights * (1.0 / gross)

    return weights.fillna(0.0).astype(float)