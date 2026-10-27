import numpy as np
import pandas as pd


def _rank_pct(s):
    return s.rank(method='average', pct=True)


def generate_weights(history, universe):
    u = pd.Index(universe)
    zero = pd.Series(0.0, index=u, dtype=float)
    if len(u) == 0:
        return zero

    ac = history.get('adjusted_close')
    if not isinstance(ac, pd.DataFrame) or ac.empty:
        ac = history.get('close')
    if not isinstance(ac, pd.DataFrame) or ac.empty:
        return zero

    cols = [c for c in u if c in ac.columns]
    if len(cols) < 10:
        return zero

    ac = ac.reindex(columns=cols)
    nrows = len(ac)
    if nrows < 130:
        return zero

    last_pos = nrows - 1
    reb_pos = last_pos - (last_pos % 21)
    if reb_pos < 129:
        reb_pos = last_pos

    start = max(0, reb_pos - 260)
    p = ac.iloc[start:reb_pos + 1].copy()
    if len(p) < 70:
        return zero
    p = p.ffill()

    last = p.iloc[-1]
    valid = last.notna() & (last > 0.5)

    raw = ac.iloc[start:reb_pos + 1]
    need = min(60, len(raw))
    valid = valid & (raw.notna().sum() >= need)

    vol_df = history.get('volume')
    if isinstance(vol_df, pd.DataFrame):
        v = vol_df.reindex(index=ac.index, columns=cols).iloc[start:reb_pos + 1].ffill()
        close_df = history.get('close')
        if isinstance(close_df, pd.DataFrame):
            px = close_df.reindex(index=ac.index, columns=cols).iloc[start:reb_pos + 1].ffill()
        else:
            px = p
        if len(v) >= 20:
            dv = (v.iloc[-20:] * px.iloc[-20:]).median()
            valid = valid & dv.notna() & (dv > 1e6)

    elig_df = history.get('eligible')
    if isinstance(elig_df, pd.DataFrame):
        e = elig_df.reindex(index=ac.index, columns=cols).iloc[start:reb_pos + 1].ffill().iloc[-1]
        valid = valid & e.fillna(False).astype(bool)

    p = p.loc[:, valid]
    if p.shape[1] < 10:
        return zero

    rets = p.pct_change()
    vol = rets.iloc[-60:].std(ddof=0)
    maxret = rets.iloc[-21:].max()

    if len(p) >= 22:
        rev = -(p.iloc[-1] / p.iloc[-22] - 1.0)
    else:
        rev = pd.Series(np.nan, index=p.columns)

    if len(p) >= 252:
        mom = p.iloc[-21] / p.iloc[-252] - 1.0
    elif len(p) >= 126:
        mom = p.iloc[-21] / p.iloc[-126] - 1.0
    else:
        mom = pd.Series(np.nan, index=p.columns)

    if len(p) >= 126:
        mom6 = p.iloc[-21] / p.iloc[-126] - 1.0
    else:
        mom6 = pd.Series(np.nan, index=p.columns)

    r_mom = _rank_pct(mom)
    r_mom6 = _rank_pct(mom6)
    r_rev = _rank_pct(rev)
    r_lowvol = _rank_pct(-vol)
    r_lowmax = _rank_pct(-maxret)

    score = (0.40 * r_mom + 0.15 * r_mom6 + 0.20 * r_rev + 0.15 * r_lowvol + 0.10 * r_lowmax)
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 10:
        return zero

    n_each = min(25, len(score) // 2)
    if n_each < 1:
        return zero

    ranked = score.sort_values(ascending=False, kind='mergesort')
    longs = ranked.index[:n_each]
    shorts = ranked.index[-n_each:]

    weight = min(0.02, 0.5 / float(n_each))
    w = pd.Series(0.0, index=u, dtype=float)
    for t in longs:
        if t in w.index:
            w.loc[t] = weight
    for t in shorts:
        if t in w.index:
            w.loc[t] = -weight

    return w.astype(float)
