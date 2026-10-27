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
    if n < 60:
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

    if valid.sum() < 20:
        valid = last.notna() & np.isfinite(last) & (last > 0.0)
    if valid.sum() < 10:
        return weights

    rets = close / close.shift(1) - 1.0
    rets = rets.replace([np.inf, -np.inf], np.nan)

    if vol_df is not None:
        dv = (raw * vol_df).iloc[-min(60, n):].median()
    else:
        dv = pd.Series(np.nan, index=tickers)

    if dv.notna().any():
        dvv = dv.replace([np.inf, -np.inf], np.nan)
        q = dvv.quantile(0.2)
        threshold = max(250000.0, float(q) if np.isfinite(q) else 0.0)
        valid = valid & (dvv >= threshold)

    if valid.sum() < 10:
        valid = last.notna() & np.isfinite(last) & (last > 0.0)

    mom_df = close.shift(21) / close.shift(252) - 1.0
    mom_df = mom_df.where(mom_df.notna(), close.shift(21) / close.shift(126) - 1.0)
    mom_df = mom_df.where(mom_df.notna(), close.shift(21) / close.shift(63) - 1.0)
    mom_df = mom_df.replace([np.inf, -np.inf], np.nan)

    vol_win = min(60, max(10, n - 1))
    vol = rets.iloc[-vol_win:].std(ddof=0) * math.sqrt(252.0)
    vol = vol.replace([np.inf, -np.inf], np.nan)

    vol_short_win = min(21, max(5, n - 1))
    vol_short = rets.iloc[-vol_short_win:].std(ddof=0) * math.sqrt(252.0)
    vol_short = vol_short.replace([np.inf, -np.inf], np.nan)

    r21_df = close / close.shift(21) - 1.0
    r21_df = r21_df.replace([np.inf, -np.inf], np.nan)

    mom_smooth = mom_df.iloc[-5:].mean()
    rev_smooth = -r21_df.iloc[-5:].mean()

    z_mom = _zscore(mom_smooth / vol)
    z_rev = _zscore(rev_smooth / vol_short)
    z_vol = _zscore(vol)

    mkt_ret = rets.mean(axis=1, skipna=True)
    if len(mkt_ret.dropna()) >= 20:
        mkt_vol = mkt_ret.iloc[-20:].std(ddof=0) * math.sqrt(252.0)
    else:
        mkt_vol = np.nan

    if len(mkt_ret.dropna()) >= 60:
        mkt_60 = (1.0 + mkt_ret.iloc[-60:]).prod() - 1.0
    else:
        mkt_60 = np.nan

    disp_series = r21_df.std(axis=1, skipna=True)
    if len(disp_series.dropna()) >= 20:
        disp_now = disp_series.iloc[-20:].mean()
    else:
        disp_now = np.nan

    disp_hist = disp_series.iloc[-252:].dropna()
    if len(disp_hist) > 20 and np.isfinite(disp_now):
        disp_pct = float((disp_hist < disp_now).mean())
    else:
        disp_pct = 0.5

    risk_off = False
    if np.isfinite(mkt_vol) and mkt_vol > 0.25:
        risk_off = True
    if np.isfinite(mkt_60) and mkt_60 < -0.05:
        risk_off = True
    if disp_pct > 0.8:
        risk_off = True

    if risk_off:
        score = 0.3 * z_mom + 0.6 * z_rev - 0.1 * z_vol
        gross_target = 0.4
    else:
        score = 0.7 * z_mom + 0.2 * z_rev - 0.1 * z_vol
        gross_target = 0.8

    if np.isfinite(disp_pct):
        gross_target *= (1.0 - 0.5 * max(0.0, disp_pct - 0.6) / 0.4)
    gross_target = max(0.2, min(0.9, gross_target))

    score = score.replace([np.inf, -np.inf], np.nan)
    valid = valid & score.notna() & vol.notna() & (vol > 0.01)
    if valid.sum() < 10:
        valid = last.notna() & np.isfinite(last) & (last > 0.0) & score.notna()
    if valid.sum() < 6:
        return weights

    s = score[valid].dropna()
    if len(s) < 6:
        return weights

    N = min(80, len(s) // 2)
    if N < 3:
        return weights

    ordered = s.sort_values()
    longs = ordered.index[-N:]
    shorts = ordered.index[:N]

    w = gross_target / (2.0 * N)
    w = min(w, 0.02)

    weights.loc[longs] = w
    weights.loc[shorts] = -w
    return weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)