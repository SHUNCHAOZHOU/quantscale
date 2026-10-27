import numpy as np
import pandas as pd
import math

def _zscore(s):
    s = pd.to_numeric(s, errors='coerce')
    s = s.replace([np.inf, -np.inf], np.nan)
    mu = s.mean()
    sd = s.std()
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd

def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    weights = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return weights

    def _get(key):
        df = history.get(key) if isinstance(history, dict) else None
        if df is None:
            return None
        try:
            return df.reindex(columns=tickers)
        except Exception:
            return None

    adj = _get('adjusted_close')
    raw = _get('close')
    vol_df = _get('volume')
    elig_df = _get('eligible')

    close = adj
    if close is None or close.dropna(how='all').shape[1] == 0:
        close = raw
    if close is None or close.empty:
        return weights
    if raw is None:
        raw = close

    n = len(close)
    if n < 30:
        return weights

    last = close.iloc[-1].astype(float)
    raw_last = raw.iloc[-1].astype(float)
    valid = last.notna() & np.isfinite(last) & (last > 0.0)
    valid = valid & raw_last.notna() & np.isfinite(raw_last) & (raw_last > 2.0)

    if elig_df is not None and len(elig_df) > 0:
        e = elig_df.iloc[-1].reindex(tickers)
        try:
            e = e.fillna(False).astype(bool)
        except Exception:
            e = e.fillna(0).astype(float).astype(bool)
        valid = valid & e

    if valid.sum() < 10:
        valid = last.notna() & np.isfinite(last) & (last > 0.0)
    if valid.sum() < 4:
        return weights

    rets = close / close.shift(1) - 1.0

    def shift_ret(d):
        if n > d:
            prev = close.shift(d).iloc[-1].astype(float)
            return last / prev - 1.0
        return pd.Series(np.nan, index=tickers)

    r252 = shift_ret(252)
    r126 = shift_ret(126)
    r21 = shift_ret(21)

    mom = 0.5 * r126 + 0.5 * (r252 - r21)
    mom = mom.where(np.isfinite(mom), r252)
    mom = mom.where(np.isfinite(mom), r126)
    mom = mom.where(np.isfinite(mom), r21)

    vol_win = min(60, max(10, n - 1))
    vol = rets.iloc[-vol_win:].std(ddof=0) * math.sqrt(252.0)
    vol = vol.replace([np.inf, -np.inf], np.nan)

    vol_short_win = min(21, max(5, n - 1))
    vol_short = rets.iloc[-vol_short_win:].std(ddof=0) * math.sqrt(252.0)

    hi_win = min(252, n)
    hi = close.iloc[-hi_win:].max()
    trend = last / hi - 1.0
    trend = trend.replace([np.inf, -np.inf], np.nan)

    if vol_df is not None:
        dv = (raw * vol_df).iloc[-min(60, n):].median()
    else:
        dv = pd.Series(np.nan, index=tickers)

    mom_adj = mom / vol
    rev_adj = (-r21) / vol_short
    score = (0.50 * _zscore(mom_adj)
             + 0.25 * _zscore(trend)
             + 0.25 * _zscore(rev_adj)
             - 0.10 * _zscore(vol))

    valid = valid & np.isfinite(score) & np.isfinite(vol) & (vol > 0.01)

    if dv.notna().any():
        dvv = dv.replace([np.inf, -np.inf], np.nan)
        q = dvv.quantile(0.2)
        threshold = max(250000.0, float(q) if np.isfinite(q) else 0.0)
        valid = valid & (dvv >= threshold)

    if valid.sum() < 10:
        valid = last.notna() & np.isfinite(last) & (last > 0.0) & np.isfinite(score)
    if valid.sum() < 4:
        return weights

    s = score[valid].dropna()
    if len(s) < 4:
        return weights

    ordered = s.sort_values()
    n_side = min(50, len(ordered) // 2)
    if n_side < 1:
        return weights

    wt = min(0.5 / float(n_side), 0.02)
    longs = ordered.index[-n_side:]
    shorts = ordered.index[:n_side]
    weights.loc[longs] = wt
    weights.loc[shorts] = -wt
    return weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
