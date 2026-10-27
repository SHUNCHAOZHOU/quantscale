import numpy as np
import pandas as pd


def _df(history, key, universe):
    x = history.get(key)
    if isinstance(x, pd.DataFrame) and not x.empty:
        return x.reindex(columns=universe)
    return None


def generate_weights(history, universe):
    u = pd.Index(universe)
    zero = pd.Series(0.0, index=u, dtype=float)
    if len(u) == 0:
        return zero

    ac = _df(history, 'adjusted_close', u)
    if ac is None:
        ac = _df(history, 'close', u)
    if ac is None or ac.shape[0] < 80:
        return zero

    tail = ac.tail(300).astype(float)
    if len(tail) < 80:
        return zero

    last_px = tail.iloc[-1]

    elig_df = history.get('eligible')
    if isinstance(elig_df, pd.DataFrame) and not elig_df.empty:
        try:
            e = elig_df.reindex(columns=u).iloc[-1].fillna(False).astype(bool)
        except Exception:
            e = pd.Series(True, index=u)
    else:
        e = pd.Series(True, index=u)

    ret = tail.pct_change().replace([np.inf, -np.inf], np.nan)
    mkt = ret.mean(axis=1, skipna=True)

    mkt_fill = mkt.fillna(0.0)
    idx = (1.0 + mkt_fill).cumprod()
    if len(idx) < 60:
        return zero

    win = min(200, len(idx))
    ma = idx.rolling(win, min_periods=min(60, win)).mean().iloc[-1]
    last_idx = idx.iloc[-1]
    if not np.isfinite(ma) or not np.isfinite(last_idx):
        return zero

    if last_idx > ma * 1.01:
        sign = 1.0
    elif last_idx < ma * 0.99:
        sign = -1.0
    else:
        return zero

    bwin = min(120, len(ret) - 1)
    if bwin < 40:
        return zero

    sub = ret.iloc[-bwin:]
    m = mkt.iloc[-bwin:]
    ok = m.notna()
    sub = sub.loc[ok]
    m = m.loc[ok]
    if len(m) < 40:
        return zero

    m_c = m - m.mean()
    r_c = sub.sub(sub.mean(axis=0), axis=1)
    cnt = sub.notna().sum(axis=0)
    cov = (r_c.fillna(0.0).mul(m_c, axis=0)).sum(axis=0) / (cnt - 1).clip(lower=1)
    var = (m_c ** 2).sum() / max(len(m) - 1, 1)
    if not np.isfinite(var) or var <= 0:
        return zero

    beta = (cov / var).replace([np.inf, -np.inf], np.nan)
    enough = cnt >= max(40, bwin // 2)
    valid = e & last_px.notna() & (last_px > 1.0) & beta.notna() & np.isfinite(beta) & enough
    names = valid[valid].index
    if len(names) < 20:
        return zero

    b = beta.loc[names].astype(float)
    ranks = b.rank(method='average', pct=True)
    score = ranks - ranks.mean()
    raw = sign * score
    if raw.abs().sum() <= 0:
        return zero

    mvol = mkt.iloc[-60:].std() * np.sqrt(252.0) if len(mkt) >= 60 else np.nan
    gross = 0.8
    if np.isfinite(mvol):
        if mvol > 0.30:
            gross = 0.45
        elif mvol > 0.20:
            gross = 0.65

    w = raw / raw.abs().sum() * gross
    w = w.clip(-0.02, 0.02)
    for _ in range(3):
        net = w.sum()
        if abs(net) < 1e-8:
            break
        w = (w - net / len(w)).clip(-0.02, 0.02)

    weights = pd.Series(0.0, index=u, dtype=float)
    weights.loc[w.index] = w.values
    return weights.fillna(0.0)