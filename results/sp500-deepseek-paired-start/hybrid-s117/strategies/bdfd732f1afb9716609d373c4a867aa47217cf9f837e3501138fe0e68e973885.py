import numpy as np
import pandas as pd


def _zscore(s):
    s = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 2:
        return pd.Series(0.0, index=s.index)
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    zero = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return zero

    ac = history.get('adjusted_close')
    if not isinstance(ac, pd.DataFrame):
        ac = history.get('close')
    if not isinstance(ac, pd.DataFrame) or ac.empty:
        return zero

    try:
        ac = ac.reindex(columns=tickers)
    except Exception:
        return zero

    dates = ac.index
    n = len(dates)
    if n < 70:
        return zero

    # Rebalance only at a completed month-end close.  On the month-end day
    # itself use that close; on later days hold the same target until the next
    # month-end.  This is the explicit turnover budget: one rebalance/month.
    asof = None
    try:
        last_date = dates[-1]
        nb = last_date + pd.offsets.BDay(1)
        if nb.month != last_date.month:
            asof = n - 1
    except Exception:
        pass

    if asof is None:
        for i in range(n - 2, -1, -1):
            if dates[i].month != dates[i + 1].month:
                asof = i
                break

    if asof is None or asof < 64:
        return zero

    px = ac.iloc[:asof + 1].tail(300)
    if len(px) < 64:
        return zero

    last_px = px.iloc[-1].astype(float)
    valid = last_px.notna() & np.isfinite(last_px) & (last_px > 2.0)
    cnt = px.notna().sum()
    valid = valid & (cnt >= 60)

    rets = px.pct_change(fill_method=None)
    vol60 = rets.tail(60).std(ddof=0).replace([np.inf, -np.inf], np.nan)
    valid = valid & vol60.notna() & (vol60 > 0.0)
    vol60 = vol60.clip(lower=1e-4)

    elig_df = history.get('eligible')
    if isinstance(elig_df, pd.DataFrame) and len(elig_df) > asof:
        try:
            e = elig_df.reindex(columns=tickers).iloc[asof]
            e = e.fillna(False).astype(bool)
            if e.sum() >= 10:
                valid = valid & e
        except Exception:
            pass

    vdf = history.get('volume')
    adv = pd.Series(np.nan, index=tickers)
    if isinstance(vdf, pd.DataFrame):
        try:
            v = vdf.reindex(columns=tickers).iloc[:asof + 1].tail(20)
            cdf = history.get('close')
            if not isinstance(cdf, pd.DataFrame):
                cdf = ac
            c = cdf.reindex(columns=tickers).iloc[:asof + 1].tail(20)
            if len(v) == len(c) and len(v) > 0:
                adv = (v * c).median(skipna=True)
        except Exception:
            pass

    def form(k):
        if len(px) > k:
            return px.iloc[-1].astype(float) / px.iloc[-k - 1].astype(float) - 1.0
        return pd.Series(np.nan, index=tickers)

    r21 = form(21)
    r63 = form(63)
    valid = valid & r21.notna() & r63.notna()
    valid = valid & (r21 < 1.5) & (r21 > -0.75) & (r63 < 2.0) & (r63 > -0.8)

    # Pure reversal: recent losers are expected to outperform recent winners.
    # Dividing by own volatility prevents the ranking from becoming a pure
    # high-volatility bet.
    base = -(0.70 * r21 + 0.30 * r63)
    score = base / vol60
    score = score.replace([np.inf, -np.inf], np.nan)
    valid = valid & score.notna()

    if adv.notna().sum() >= 10:
        dvv = adv.replace([np.inf, -np.inf], np.nan)
        q = dvv[valid].quantile(0.20) if valid.sum() >= 5 else dvv.quantile(0.20)
        threshold = max(250000.0, float(q) if np.isfinite(q) else 0.0)
        valid = valid & dvv.notna() & (dvv >= threshold)

    if valid.sum() < 10:
        valid = last_px.notna() & np.isfinite(last_px) & (last_px > 2.0) & score.notna()

    s = score[valid].dropna().sort_values(ascending=False)
    m = len(s)
    if m < 4:
        return zero

    n_side = min(50, m // 2)
    if n_side < 2:
        return zero

    w = min(0.02, 0.5 / float(n_side))
    longs = s.index[:n_side]
    shorts = s.index[-n_side:]

    weights = pd.Series(0.0, index=tickers, dtype=float)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.02, 0.02)
    return weights.astype(float)