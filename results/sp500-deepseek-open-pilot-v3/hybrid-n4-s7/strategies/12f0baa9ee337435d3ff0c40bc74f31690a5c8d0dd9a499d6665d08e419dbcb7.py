import numpy as np
import pandas as pd


def generate_weights(history, universe):
    u = pd.Index(universe)
    zeros = pd.Series(0.0, index=u)

    try:
        ac = history['adjusted_close']
        vol = history['volume']
    except Exception:
        return zeros

    if ac is None or vol is None:
        return zeros
    if not isinstance(ac, pd.DataFrame) or not isinstance(vol, pd.DataFrame):
        return zeros
    if ac.shape[0] < 25:
        return zeros

    ac = ac.reindex(columns=u)
    vol = vol.reindex(columns=u)
    ac_tail = ac.iloc[-31:].copy()
    vol_tail = vol.iloc[-31:].copy()
    if len(ac_tail) < 21:
        return zeros

    elig = history.get('eligible', None)
    if isinstance(elig, pd.DataFrame) and len(elig) > 0:
        elig = elig.reindex(columns=u)
        elig_last = elig.iloc[-1].fillna(False).astype(bool)
    else:
        elig_last = pd.Series(True, index=u)

    last_close = ac_tail.iloc[-1]
    rets = ac_tail.pct_change().replace([np.inf, -np.inf], np.nan)

    r5 = ac_tail / ac_tail.shift(5) - 1.0
    r10 = ac_tail / ac_tail.shift(10) - 1.0
    ma5 = ac_tail.rolling(5).mean()
    dist = ac_tail / ma5 - 1.0

    vol20 = rets.rolling(20).std()
    vol20 = vol20.mask(vol20 < 0.005, 0.005)

    raw = -(0.5 * r5 + 0.2 * r10 + 0.3 * dist)
    sig_series = raw / vol20
    sig = sig_series.iloc[-3:].mean(axis=0, skipna=True)

    dollar = (ac_tail * vol_tail).iloc[-20:].median()

    valid = (
        elig_last
        & last_close.notna()
        & sig.notna()
        & np.isfinite(sig)
        & (last_close > 5.0)
    )
    valid = valid & r5.iloc[-1].notna() & dist.iloc[-1].notna() & vol20.iloc[-1].notna()

    if valid.sum() >= 20:
        dv = dollar[valid]
        thresh = dv.quantile(0.25)
        valid = valid & (dollar >= thresh)

    if valid.sum() < 6:
        valid = (
            elig_last
            & last_close.notna()
            & sig.notna()
            & np.isfinite(sig)
            & (last_close > 5.0)
        )
        valid = valid & r5.iloc[-1].notna() & dist.iloc[-1].notna() & vol20.iloc[-1].notna()

    s = sig[valid]
    if len(s) < 6:
        return zeros

    mu = s.mean()
    sd = s.std(ddof=0)
    if (not np.isfinite(sd)) or sd <= 1e-12:
        return zeros

    z = (s - mu) / sd
    pos = z[z > 0.75].sort_values(ascending=False)
    neg = z[z < -0.75].sort_values(ascending=True)

    n = min(25, len(pos), len(neg))
    if n == 0:
        return zeros

    w = pd.Series(0.0, index=u)
    w.loc[pos.index[:n]] = 0.02
    w.loc[neg.index[:n]] = -0.02
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return w