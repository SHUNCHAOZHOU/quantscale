import numpy as np
import pandas as pd

def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)

    adj = history.get('adjusted_close')
    if adj is None:
        adj = history.get('close')
    if adj is None or not isinstance(adj, pd.DataFrame):
        return pd.Series(0.0, index=universe)

    cols = [c for c in universe if c in adj.columns]
    if len(cols) == 0:
        return pd.Series(0.0, index=universe)

    if not adj.index.is_monotonic_increasing:
        adj = adj.sort_index()
    px = adj[cols].tail(320).copy()
    n = len(px)
    if n < 40:
        return pd.Series(0.0, index=universe)

    def _mom(start_lag, end_lag):
        if n <= end_lag + 2:
            return pd.Series(np.nan, index=cols)
        end_pos = n - 1 - end_lag
        start_pos = max(0, n - 1 - start_lag)
        if end_pos <= start_pos:
            return pd.Series(np.nan, index=cols)
        return px.iloc[end_pos] / px.iloc[start_pos] - 1.0

    mom12 = _mom(252, 21)
    mom6 = _mom(126, 21)
    if n > 21:
        ret21 = px.iloc[-1] / px.iloc[n - 1 - 21] - 1.0
    else:
        ret21 = pd.Series(np.nan, index=cols)

    rets = px.pct_change()
    vol_window = min(126, max(2, n - 1))
    vol = rets.tail(vol_window).std() * np.sqrt(252.0)
    vol = vol.replace([np.inf, -np.inf], np.nan)

    last_price = px.iloc[-1]
    valid = last_price.notna() & (last_price > 0) & mom12.notna()
    valid = valid & (px.notna().sum() >= 60)

    if 'eligible' in history:
        edf = history['eligible']
        if isinstance(edf, pd.DataFrame):
            if not edf.index.is_monotonic_increasing:
                edf = edf.sort_index()
            ecols = [c for c in cols if c in edf.columns]
            if ecols:
                e_last = edf[ecols].tail(1)
                if len(e_last) > 0:
                    e = e_last.iloc[-1].reindex(cols).fillna(False).astype(bool)
                    valid = valid & e

    idx_valid = valid[valid].index
    if len(idx_valid) < 10:
        return pd.Series(0.0, index=universe)

    mom6 = mom6.where(mom6.notna(), mom12)
    mom12_v = mom12.loc[idx_valid]
    mom6_v = mom6.loc[idx_valid]
    vol_v = vol.loc[idx_valid].fillna(vol.loc[idx_valid].median())
    if vol_v.isna().all():
        vol_v = pd.Series(1.0, index=idx_valid)
    ret21_v = ret21.loc[idx_valid].fillna(ret21.loc[idx_valid].median())
    if ret21_v.isna().all():
        ret21_v = pd.Series(0.0, index=idx_valid)

    def rp(s):
        return s.rank(method='average', pct=True)

    score = (0.55 * rp(mom12_v) +
             0.25 * rp(mom6_v) +
             0.20 * (1.0 - rp(vol_v)) -
             0.20 * rp(ret21_v))
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 10:
        return pd.Series(0.0, index=universe)

    n_side = min(50, len(score) // 2)
    if n_side < 1:
        return pd.Series(0.0, index=universe)

    longs = score.nlargest(n_side).index
    shorts = score.nsmallest(n_side).index
    w = min(0.02, 0.5 / float(n_side))
    weights = pd.Series(0.0, index=universe)
    weights.loc[longs] = w
    weights.loc[shorts] = -w

    gross = weights.abs().sum()
    if gross > 1.0 + 1e-12:
        weights = weights * (1.0 / gross)
    weights = weights.clip(-0.02, 0.02)
    return weights.reindex(universe).fillna(0.0)