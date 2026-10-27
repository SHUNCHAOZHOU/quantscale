import numpy as np
import pandas as pd


def generate_weights(history, universe):
    idx = pd.Index(universe)
    if len(idx) == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=idx, dtype=float)

    adj = history.get('adjusted_close')
    if adj is None:
        adj = history.get('close')
    if adj is None or not isinstance(adj, pd.DataFrame):
        return zero

    if not adj.index.is_monotonic_increasing:
        adj = adj.sort_index()

    cols = [c for c in idx if c in adj.columns]
    if len(cols) < 10:
        return zero

    n = len(adj)
    if n < 80:
        return zero

    anchor = ((n - 1) // 21) * 21
    if anchor < 60:
        return zero

    px = adj.iloc[:anchor + 1][cols].astype(float)
    px = px.replace([np.inf, -np.inf], np.nan)
    if len(px) < 60:
        return zero

    last = px.iloc[-1]
    valid = last.notna() & (last > 0) & (px.notna().sum(axis=0) >= 40)

    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame):
        e = elig.reindex(columns=cols)
        if len(e.index) > 0:
            if not e.index.is_monotonic_increasing:
                e = e.sort_index()
            cur = e.iloc[-1].reindex(cols).fillna(False).astype(bool)
            valid = valid & cur

    vol_df = history.get('volume')
    if isinstance(vol_df, pd.DataFrame):
        v = vol_df.reindex(index=px.index, columns=cols).astype(float)
        dollar = (v * px).iloc[-60:].median(axis=0)
        dollar = dollar.replace([np.inf, -np.inf], np.nan)
        valid = valid & dollar.notna() & (dollar > 0)
        if dollar.notna().sum() >= 20:
            thresh = dollar.quantile(0.25)
            valid = valid & (dollar >= thresh)

    rets = px.pct_change()
    rets = rets.replace([np.inf, -np.inf], np.nan)

    def _std_tail(window):
        return rets.iloc[-min(window, len(rets)):].std(axis=0, ddof=0) * np.sqrt(252.0)

    vol60 = _std_tail(60)
    vol120 = _std_tail(120)
    vol = vol120.where(vol120.notna(), vol60)
    vol = vol.replace([np.inf, -np.inf], np.nan)

    down = rets.where(rets < 0.0)
    dvol = down.iloc[-min(126, len(down)):].std(axis=0, ddof=0) * np.sqrt(252.0)
    dvol = dvol.replace([np.inf, -np.inf], np.nan)
    dvol = dvol.where(dvol.notna(), vol)

    def _ret(lag):
        if len(px) <= lag:
            return pd.Series(np.nan, index=cols, dtype=float)
        return px.iloc[-1] / px.iloc[-1 - lag] - 1.0

    def _skip_ret(lag, skip):
        if len(px) <= lag + skip:
            return pd.Series(np.nan, index=cols, dtype=float)
        return px.iloc[-1 - skip] / px.iloc[-1 - lag - skip] - 1.0

    mom12 = _skip_ret(252, 21)
    mom6 = _skip_ret(126, 21)
    mom12 = mom12.where(mom12.notna(), _ret(252))
    mom6 = mom6.where(mom6.notna(), _ret(126))
    mom = mom12.where(mom12.notna(), mom6)
    mom = mom.where(mom.notna(), _ret(63))
    mom = mom.replace([np.inf, -np.inf], np.nan)

    sma_window = min(200, len(px))
    sma = px.iloc[-sma_window:].mean(axis=0)
    trend = last / sma - 1.0
    trend = trend.replace([np.inf, -np.inf], np.nan)

    rev5 = _ret(5)
    rev5 = rev5.replace([np.inf, -np.inf], np.nan)

    valid_idx = valid[valid].index
    if len(valid_idx) < 10:
        return zero

    vol_v = vol.reindex(valid_idx)
    dvol_v = dvol.reindex(valid_idx)
    mom_v = mom.reindex(valid_idx)
    trend_v = trend.reindex(valid_idx)
    rev_v = rev5.reindex(valid_idx)

    def _fill(s):
        med = s.median()
        if pd.isna(med):
            med = 0.0
        return s.fillna(med)

    vol_v = _fill(vol_v)
    dvol_v = _fill(dvol_v)
    mom_v = _fill(mom_v)
    trend_v = _fill(trend_v)
    rev_v = _fill(rev_v)

    def rp(s):
        return s.rank(method='average', pct=True)

    score = (0.35 * (1.0 - rp(vol_v))
             + 0.15 * (1.0 - rp(dvol_v))
             + 0.30 * rp(mom_v)
             + 0.20 * rp(trend_v)
             - 0.10 * rp(rev_v))
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 10:
        return zero

    n_side = min(30, len(score) // 2)
    if n_side < 1:
        return zero

    longs = score.nlargest(n_side).index
    shorts = score.nsmallest(n_side).index
    w = min(0.02, 0.40 / float(n_side))
    weights = zero.copy()
    weights.loc[longs] = w
    weights.loc[shorts] = -w

    gross = float(weights.abs().sum())
    if gross > 1.0 + 1e-12:
        weights = weights * (1.0 / gross)
    weights = weights.clip(-0.02, 0.02)
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights.reindex(idx).fillna(0.0)