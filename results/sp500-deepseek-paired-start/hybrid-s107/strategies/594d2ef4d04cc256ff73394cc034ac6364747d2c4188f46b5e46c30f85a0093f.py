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

    px = history.get('adjusted_close')
    if px is None:
        px = history.get('close')
    if px is None or not isinstance(px, pd.DataFrame) or px.shape[0] == 0:
        return zero

    px = px.sort_index()
    n = len(px.index)
    period = 20
    lookback = 160
    if n < 130:
        return zero

    reb_idx = ((n - 1) // period) * period
    if reb_idx < 126:
        return zero

    cols = [c for c in tickers if c in px.columns]
    if len(cols) < 20:
        return zero

    start = max(0, reb_idx - lookback + 1)
    sub = px.iloc[start:reb_idx + 1][cols].astype(float)
    sub = sub.replace([np.inf, -np.inf], np.nan)
    if sub.shape[0] < 126:
        return zero

    last = sub.iloc[-1]
    valid = last.notna() & (last > 0)
    valid = valid & (sub.notna().sum(axis=0) >= 100)

    ret = sub / sub.shift(1) - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)
    vol_d = ret.iloc[-60:].std(axis=0, ddof=1)
    vol_d = vol_d.where(vol_d > 0)

    r40 = sub.iloc[-6] / sub.iloc[-46] - 1.0
    r120 = sub.iloc[-6] / sub.iloc[-126] - 1.0
    r40 = r40.replace([np.inf, -np.inf], np.nan)
    r120 = r120.replace([np.inf, -np.inf], np.nan)

    valid = valid & r40.notna() & vol_d.notna()

    vol_df = history.get('volume')
    if vol_df is not None and isinstance(vol_df, pd.DataFrame):
        vcols = [c for c in cols if c in vol_df.columns]
        if len(vcols) > 0:
            vsub = vol_df.iloc[start:reb_idx + 1][vcols].astype(float)
            vsub = vsub.replace([np.inf, -np.inf], np.nan)
            dollar = vsub * sub[vcols]
            liq = dollar.iloc[-60:].median(axis=0)
            liq = liq.reindex(cols)
            valid = valid & (liq > 0)
            if liq.notna().sum() >= 20:
                thresh = liq.quantile(0.10)
                valid = valid & (liq >= thresh)

    elig = history.get('eligible')
    if elig is not None and isinstance(elig, pd.DataFrame):
        edf = elig.reindex(columns=cols)
        if len(edf.index) == n:
            e = edf.iloc[reb_idx]
        else:
            dt = px.index[reb_idx]
            if dt in edf.index:
                e = edf.loc[dt]
            else:
                e = None
        if e is not None:
            valid = valid & e.fillna(False).astype(bool)

    valid = valid.fillna(False)
    valid_t = valid[valid].index
    if len(valid_t) < 8:
        return zero

    r40v = r40.reindex(valid_t)
    r120v = r120.reindex(valid_t)
    volv = vol_d.reindex(valid_t)
    volv = volv.fillna(volv.median())
    if volv.isna().all():
        volv = pd.Series(1.0, index=valid_t)

    z40 = r40v / (volv * np.sqrt(40.0))
    z120 = r120v / (volv * np.sqrt(120.0))
    z120 = z120.where(z120.notna(), z40)
    z40 = z40.replace([np.inf, -np.inf], np.nan)
    z120 = z120.replace([np.inf, -np.inf], np.nan)

    score = (0.60 * _rank_pct(z40) +
             0.40 * _rank_pct(z120) -
             0.15 * _rank_pct(volv))
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 8:
        return zero

    n_side = min(80, len(score) // 2)
    if n_side < 1:
        return zero

    order = score.sort_values(ascending=False)
    longs = order.index[:n_side]
    shorts = order.index[-n_side:]

    wmag = min(0.02, 0.5 / float(n_side))
    weights = pd.Series(0.0, index=tickers, dtype=float)
    weights.loc[longs] = wmag
    weights.loc[shorts] = -wmag

    gross = float(weights.abs().sum())
    if gross > 1.0 + 1e-12:
        weights = weights * (1.0 / gross)
    weights = weights.clip(-0.02, 0.02)
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights.reindex(tickers).fillna(0.0)