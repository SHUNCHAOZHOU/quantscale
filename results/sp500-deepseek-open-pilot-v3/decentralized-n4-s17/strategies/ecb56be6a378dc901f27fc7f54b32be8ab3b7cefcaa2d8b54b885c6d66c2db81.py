import numpy as np
import pandas as pd


def generate_weights(history, universe):
    u = pd.Index(universe)
    w = pd.Series(0.0, index=u, dtype=float)
    if len(u) == 0:
        return w

    ac = history.get('adjusted_close')
    if not isinstance(ac, pd.DataFrame) or ac.empty:
        ac = history.get('close')
    if not isinstance(ac, pd.DataFrame) or ac.empty:
        return w

    cols = [c for c in u if c in ac.columns]
    if len(cols) < 10:
        return w

    px = ac.loc[:, cols].copy()
    if not px.index.is_monotonic_increasing:
        px = px.sort_index()

    n = len(px)
    if n < 260:
        return w

    last_pos = n - 1
    reb_pos = last_pos - (last_pos % 21)
    if reb_pos < 252:
        return w

    p = px.iloc[:reb_pos + 1].ffill()
    if len(p) < 252:
        return w

    last = p.iloc[-1]
    last_num = pd.to_numeric(last, errors='coerce')
    valid = last_num.notna() & np.isfinite(last_num) & (last_num > 0.5)
    valid = valid & (p.notna().sum() >= 252)

    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame) and not elig.empty:
        try:
            e = elig.reindex(index=p.index, columns=cols).ffill().iloc[-1]
            valid = valid & e.fillna(False).astype(bool)
        except Exception:
            pass

    vol = history.get('volume')
    if isinstance(vol, pd.DataFrame) and not vol.empty:
        try:
            v = vol.reindex(index=p.index, columns=cols).ffill()
            close = history.get('close')
            if isinstance(close, pd.DataFrame) and not close.empty:
                c = close.reindex(index=p.index, columns=cols).ffill()
            else:
                c = p
            if len(v) >= 20:
                dv = (v.iloc[-20:] * c.iloc[-20:]).median()
                valid = valid & dv.notna() & np.isfinite(dv) & (dv > 1e6)
        except Exception:
            pass

    if int(valid.sum()) < 10:
        return w

    rets = p / p.shift(1) - 1.0
    vol60 = rets.iloc[-60:].std(ddof=0)
    vol60 = vol60.replace([np.inf, -np.inf], np.nan)
    vol60 = vol60.where(vol60 > 1e-12)

    mom12 = p.iloc[-21] / p.iloc[-252] - 1.0
    mom6 = p.iloc[-21] / p.iloc[-126] - 1.0
    rev21 = -(p.iloc[-1] / p.iloc[-21] - 1.0)
    rev5 = -(p.iloc[-1] / p.iloc[-6] - 1.0)
    ma200 = p.iloc[-200:].mean()
    trend = p.iloc[-1] / ma200 - 1.0
    maxret = rets.iloc[-21:].max()
    mom12_vol = mom12 / vol60

    def rk(s):
        s = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)
        r = s.where(valid).rank(method='average', pct=True)
        return r.fillna(0.5)

    score = (
        0.30 * rk(mom12)
        + 0.10 * rk(mom12_vol)
        + 0.15 * rk(mom6)
        + 0.15 * rk(rev21)
        + 0.05 * rk(rev5)
        + 0.10 * rk(-vol60)
        + 0.10 * rk(trend)
        + 0.05 * rk(-maxret)
    )
    score = score.where(valid)
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 10:
        return w

    ordered = score.sort_values(ascending=False, kind='mergesort')
    k = min(25, len(ordered) // 2)
    if k < 1:
        return w

    longs = ordered.index[:k]
    shorts = ordered.index[-k:]
    per = min(0.02, 0.5 / float(k))

    w.loc[longs] = per
    w.loc[shorts] = -per
    return w.astype(float)