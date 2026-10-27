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
    if n < 80:
        return weights

    dates = close.index
    last_date = dates[-1]

    # Rebalance monthly: on the first trading day of a month, use the prior
    # month-end close as the signal date. On all other days, use the same
    # prior month-end signal. This keeps weights constant intra-month and
    # makes average daily turnover approximately monthly turnover / 21.
    if n >= 2 and last_date.month != dates[-2].month:
        signal_date = dates[-2]
    else:
        months = dates.to_period('M')
        current_month = months[-1]
        prev_dates = dates[months < current_month]
        if len(prev_dates) > 0:
            signal_date = prev_dates[-1]
        else:
            signal_date = dates[0]

    close_s = close.loc[:signal_date]
    raw_s = raw.loc[:signal_date]
    vol_s = vol_df.loc[:signal_date] if vol_df is not None else None
    elig_s = elig_df.loc[:signal_date] if elig_df is not None else None
    if len(close_s) < 80:
        return weights

    last = close_s.iloc[-1].astype(float)
    raw_last = raw_s.iloc[-1].astype(float)
    valid = last.notna() & np.isfinite(last) & (last > 0.0)
    valid = valid & raw_last.notna() & np.isfinite(raw_last) & (raw_last > 2.0)

    if elig_s is not None and len(elig_s) > 0:
        e = elig_s.iloc[-1].reindex(tickers)
        try:
            e = e.fillna(False).astype(bool)
        except Exception:
            e = e.fillna(0).astype(float).astype(bool)
        valid = valid & e

    if valid.sum() < 30:
        valid = last.notna() & np.isfinite(last) & (last > 0.0)
        valid = valid & raw_last.notna() & np.isfinite(raw_last) & (raw_last > 2.0)
    if valid.sum() < 10:
        return weights

    rets = close_s / close_s.shift(1) - 1.0
    rets = rets.replace([np.inf, -np.inf], np.nan)

    vol_win = min(60, len(rets) - 1)
    if vol_win < 10:
        return weights
    vol = rets.iloc[-vol_win:].std(ddof=0) * math.sqrt(252.0)
    vol = vol.replace([np.inf, -np.inf], np.nan)

    def ret_shift(d):
        if len(close_s) > d:
            prev = close_s.shift(d).iloc[-1].astype(float)
            return last / prev - 1.0
        return pd.Series(np.nan, index=tickers)

    r21 = ret_shift(21)
    rev = -r21.replace([np.inf, -np.inf], np.nan)

    if len(close_s) > 252:
        prev21 = close_s.shift(21).iloc[-1].astype(float)
        prev252 = close_s.shift(252).iloc[-1].astype(float)
        mom = prev21 / prev252 - 1.0
    elif len(close_s) > 126:
        prev21 = close_s.shift(21).iloc[-1].astype(float)
        prev126 = close_s.shift(126).iloc[-1].astype(float)
        mom = prev21 / prev126 - 1.0
    else:
        mom = pd.Series(np.nan, index=tickers)
    mom = mom.replace([np.inf, -np.inf], np.nan)

    # Liquidity: exclude bottom 20% by median dollar volume.
    if vol_s is not None:
        dv = (raw_s * vol_s).iloc[-min(60, len(raw_s)):].median()
        dvv = dv.replace([np.inf, -np.inf], np.nan)
        if dvv.notna().sum() > 20:
            q20 = dvv.quantile(0.2)
            if np.isfinite(q20):
                valid = valid & (dvv >= max(250000.0, float(q20)))
            else:
                valid = valid & dvv.notna()

        # Turnover proxy: recent 20d mean volume / 120d median volume.
        # Exclude top and bottom 20% of this proxy to avoid names where
        # unusual volume spikes or drying liquidity make costs unstable.
        vwin = min(120, len(vol_s))
        if vwin >= 30:
            vmed = vol_s.iloc[-vwin:].median()
            v20 = vol_s.iloc[-min(20, len(vol_s)):].mean()
            tr = v20 / vmed
            tr = tr.replace([np.inf, -np.inf], np.nan)
            if tr.notna().sum() > 20:
                lo = tr.quantile(0.2)
                hi = tr.quantile(0.8)
                if np.isfinite(lo) and np.isfinite(hi):
                    valid = valid & (tr >= lo) & (tr <= hi)

    if valid.sum() < 30:
        valid = last.notna() & np.isfinite(last) & (last > 0.0)
        valid = valid & raw_last.notna() & np.isfinite(raw_last) & (raw_last > 2.0)
        if elig_s is not None and len(elig_s) > 0:
            e = elig_s.iloc[-1].reindex(tickers)
            try:
                e = e.fillna(False).astype(bool)
            except Exception:
                e = e.fillna(0).astype(float).astype(bool)
            valid = valid & e
    if valid.sum() < 10:
        return weights

    z_vol = _zscore(vol)
    z_rev = _zscore(rev)

    # Core: long low volatility, with a modest one-month reversal overlay.
    # Momentum weight is zero: prior high-turnover momentum books lost and
    # the post-turnover-truncation sign is not supported by the evidence.
    score = -0.80 * z_vol + 0.20 * z_rev
    score = score.replace([np.inf, -np.inf], np.nan)

    valid = valid & score.notna() & vol.notna() & (vol > 0.01)
    if valid.sum() < 10:
        valid = last.notna() & np.isfinite(last) & (last > 0.0) & score.notna()
    if valid.sum() < 6:
        return weights

    s = score[valid].dropna()
    if len(s) < 6:
        return weights

    N = min(25, len(s) // 2)
    if N < 3:
        return weights

    ordered = s.sort_values()
    longs = ordered.index[-N:]
    shorts = ordered.index[:N]

    # Equal weight, max 2% per name. With N=25 this is gross 1.0, net 0.
    w = min(0.02, 1.0 / (2.0 * N))

    weights.loc[longs] = w
    weights.loc[shorts] = -w
    return weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)