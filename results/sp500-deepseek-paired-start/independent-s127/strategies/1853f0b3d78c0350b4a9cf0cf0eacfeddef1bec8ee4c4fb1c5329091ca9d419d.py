import numpy as np
import pandas as pd


def _as_frame(df):
    if df is None:
        return None
    if isinstance(df, pd.DataFrame):
        return df
    try:
        return pd.DataFrame(df)
    except Exception:
        return None


def _row_at(df, pos, cols, default=np.nan):
    out = pd.Series(default, index=cols, dtype=float)
    if df is None or df.empty or pos <= 0:
        return out
    try:
        row = df.iloc[pos - 1]
    except Exception:
        return out
    if isinstance(row, pd.Series):
        r = row.reindex(cols)
    else:
        try:
            r = pd.Series(row, index=df.columns, dtype=float).reindex(cols)
        except Exception:
            return out
    return r.astype(float)


def _rank_pct(s):
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    return s.rank(method='average', pct=True, na_option='keep')


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    adj = history.get('adjusted_close')
    adj = _as_frame(adj)
    if adj is None or adj.empty:
        adj = _as_frame(history.get('close'))
    if adj is None or adj.empty:
        return zero

    cols = [c for c in universe if c in adj.columns]
    if len(cols) == 0:
        return zero

    px_all = adj.loc[:, cols].astype(float)
    idx = px_all.index
    if len(idx) < 270:
        return zero

    last_date = idx[-1]
    try:
        months = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max()
        asof = months.iloc[-1]
    except Exception:
        asof = last_date

    pos = int(idx.searchsorted(asof, side='right'))
    if pos < 260:
        return zero

    hist = px_all.iloc[:pos]
    if len(hist) < 260:
        return zero

    ret = hist / hist.shift(1) - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)

    mom = hist.shift(21) / hist.shift(252) - 1.0
    mom = mom.replace([np.inf, -np.inf], np.nan)
    rev = -(hist / hist.shift(21) - 1.0)
    rev = rev.replace([np.inf, -np.inf], np.nan)
    vol = ret.tail(63).std()

    last = hist.iloc[-1].astype(float)
    m = mom.iloc[-1].astype(float)
    r = rev.iloc[-1].astype(float)
    v = vol.iloc[-1].astype(float)

    score = 0.40 * _rank_pct(m) + 0.40 * _rank_pct(r) - 0.20 * _rank_pct(v)
    score = score.reindex(cols)

    valid = last.notna() & (last > 0.0) & score.notna() & m.notna() & r.notna() & v.notna()

    elig = _as_frame(history.get('eligible'))
    if elig is not None and not elig.empty:
        ecols = [c for c in cols if c in elig.columns]
        if len(ecols) > 0:
            epos = int(elig.index.searchsorted(asof, side='right'))
            ev = _row_at(elig, epos, cols, default=np.nan)
            evb = ev.fillna(1.0) > 0.5
            valid = valid & evb

    volume = _as_frame(history.get('volume'))
    if volume is not None and not volume.empty:
        vcols = [c for c in cols if c in volume.columns]
        if len(vcols) > 0:
            try:
                vv = volume.loc[:, vcols].astype(float)
                vv = vv.reindex(index=hist.index)
                pp = hist.loc[:, vcols]
                adv_series = (vv.tail(21) * pp.tail(21)).mean()
                adv = adv_series.reindex(cols)
                liq_ok = adv.fillna(0.0) > 1000000.0
                valid = valid & liq_ok
            except Exception:
                pass

    cand = score[valid].dropna()
    if len(cand) < 2:
        return zero

    n = min(50, len(cand) // 2)
    if n < 1:
        return zero

    order = cand.sort_values(ascending=False)
    longs = list(order.index[:n])
    shorts = list(order.index[-n:])

    w = min(0.02, 0.5 / float(n))
    weights = pd.Series(0.0, index=universe, dtype=float)
    for t in longs:
        weights.loc[t] = weights.loc[t] + w
    for t in shorts:
        weights.loc[t] = weights.loc[t] - w

    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.02, 0.02)
    return weights
