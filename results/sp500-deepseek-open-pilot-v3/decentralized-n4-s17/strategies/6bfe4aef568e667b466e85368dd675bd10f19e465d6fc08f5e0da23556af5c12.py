import numpy as np
import pandas as pd


def generate_weights(history, universe):
    u = pd.Index(universe)
    zero = pd.Series(0.0, index=u, dtype=float)
    if len(u) == 0:
        return zero

    adj = history.get('adjusted_close')
    if not isinstance(adj, pd.DataFrame) or adj.empty:
        adj = history.get('close')
    if not isinstance(adj, pd.DataFrame) or adj.empty:
        return zero

    adj = adj.reindex(columns=u)
    n = len(adj)
    if n < 260:
        return zero

    last_pos = n - 1
    reb_pos = last_pos - (last_pos % 21)
    if reb_pos < 252:
        return zero

    start = max(0, reb_pos - 260)
    p_full = adj.iloc[start:reb_pos + 1].ffill()
    if len(p_full) < 253:
        return zero

    p = p_full.iloc[-253:]
    last = p.iloc[-1]
    valid = last.notna() & (last > 0.5)
    valid &= np.isfinite(last.astype(float))

    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame) and not elig.empty:
        e = elig.reindex(index=p.index, columns=u).ffill().iloc[-1]
        valid &= e.fillna(False).astype(bool)

    vol = history.get('volume')
    if isinstance(vol, pd.DataFrame) and not vol.empty:
        v = vol.reindex(index=p.index, columns=u).ffill().iloc[-20:]
        close = history.get('close')
        if isinstance(close, pd.DataFrame) and not close.empty:
            px = close.reindex(index=p.index, columns=u).ffill().iloc[-20:]
        else:
            px = p.iloc[-20:]
        dollar_vol = (v * px).median()
        valid &= dollar_vol.notna() & np.isfinite(dollar_vol) & (dollar_vol > 1e6)

    cols = valid[valid].index
    if len(cols) < 10:
        return zero

    pv = p.loc[:, cols]
    logp = np.log(pv.replace([np.inf, -np.inf], np.nan))
    rets = logp.diff().iloc[1:]
    if len(rets) < 60:
        return zero

    counts = rets.notna().sum()
    good = counts >= 200
    cols = cols[good]
    if len(cols) < 10:
        return zero

    rets = rets.loc[:, cols]
    mkt = rets.mean(axis=1)
    resid = rets.sub(mkt, axis=0)

    mom = resid.iloc[:-21].sum()
    rev = resid.iloc[-21:].sum()
    volr = resid.std(ddof=0)

    mom_z = mom / (volr * np.sqrt(max(1, len(resid) - 21)))
    rev_z = rev / (volr * np.sqrt(21))

    r_mom = mom_z.rank(pct=True)
    r_rev = (-rev_z).rank(pct=True)
    score = 0.6 * r_mom + 0.4 * r_rev
    score = score.replace([np.inf, -np.inf], np.nan).dropna()

    if len(score) < 10:
        return zero

    ordered = score.sort_values(ascending=False, kind='mergesort')
    n_each = min(25, len(ordered) // 2)
    if n_each < 1:
        return zero

    longs = ordered.index[:n_each]
    shorts = ordered.index[-n_each:]
    w = min(0.02, 0.5 / float(n_each))

    weights = pd.Series(0.0, index=u, dtype=float)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    return weights.astype(float)
