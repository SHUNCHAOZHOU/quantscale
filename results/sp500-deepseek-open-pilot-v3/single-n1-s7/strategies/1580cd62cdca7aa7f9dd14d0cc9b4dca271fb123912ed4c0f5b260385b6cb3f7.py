import numpy as np
import pandas as pd


def _empty(tickers):
    return pd.Series(0.0, index=pd.Index(tickers), dtype=float)


def generate_weights(history, universe):
    tickers = pd.Index(universe)
    if len(tickers) == 0:
        return pd.Series(dtype=float)

    try:
        adj_full = history['adjusted_close']
    except Exception:
        try:
            adj_full = history['close']
        except Exception:
            return _empty(tickers)

    if adj_full is None or len(adj_full.index) == 0:
        return _empty(tickers)

    adj = adj_full.reindex(columns=tickers).copy()
    dates = adj.index
    latest = dates[-1]

    if 'eligible' in history and history['eligible'] is not None:
        elig_full = history['eligible'].reindex(index=dates, columns=tickers)
    else:
        elig_full = None

    latest_ok = adj.iloc[-1].notna()
    if elig_full is not None:
        latest_ok = latest_ok & elig_full.iloc[-1].fillna(False).astype(bool)

    periods = np.asarray(dates.year) * 12 + np.asarray(dates.month)
    latest_period = int(dates.year[-1]) * 12 + int(dates.month[-1])
    prior = np.where(periods < latest_period)[0]
    if len(prior) > 0:
        sig_idx = int(prior[-1])
    else:
        sig_idx = len(dates) - 1

    start_idx = max(0, sig_idx - 320)
    sub = adj.iloc[start_idx:sig_idx + 1].ffill(limit=5)
    n = len(sub)
    if n < 30:
        return _empty(tickers)

    if elig_full is not None:
        sig_ok = elig_full.iloc[sig_idx].fillna(False).astype(bool)
    else:
        sig_ok = sub.iloc[-1].notna()

    liq_ok = pd.Series(True, index=tickers, dtype=bool)
    if 'volume' in history and 'close' in history:
        try:
            vol = history['volume'].reindex(index=dates, columns=tickers).iloc[start_idx:sig_idx + 1]
            close = history['close'].reindex(index=dates, columns=tickers).iloc[start_idx:sig_idx + 1]
            dollar = close * vol
            liq = dollar.tail(60).median()
            liq_rank = liq.rank(pct=True, na_option='keep')
            liq_ok = liq.notna() & (liq > 0) & (liq_rank >= 0.20)
        except Exception:
            liq_ok = pd.Series(True, index=tickers, dtype=bool)

    mom = pd.Series(np.nan, index=tickers, dtype=float)
    if n >= 253:
        mom = sub.iloc[-22] / sub.iloc[-253] - 1.0
    elif n >= 127:
        mom = sub.iloc[-22] / sub.iloc[-127] - 1.0

    rev = pd.Series(np.nan, index=tickers, dtype=float)
    if n >= 6:
        rev = -(sub.iloc[-1] / sub.iloc[-6] - 1.0)

    vola = pd.Series(np.nan, index=tickers, dtype=float)
    if n >= 20:
        rets = sub.pct_change(fill_method=None)
        vola = rets.tail(60).std(skipna=True) * np.sqrt(252.0)

    high_prox = pd.Series(np.nan, index=tickers, dtype=float)
    if n >= 60:
        high_prox = sub.iloc[-1] / sub.tail(min(n, 252)).max(skipna=True) - 1.0

    mom_r = mom.rank(pct=True, na_option='keep')
    rev_r = rev.rank(pct=True, na_option='keep')
    vol_r = (-vola).rank(pct=True, na_option='keep')
    hp_r = high_prox.rank(pct=True, na_option='keep')

    score = 0.50 * mom_r + 0.25 * rev_r + 0.15 * vol_r + 0.10 * hp_r
    valid = latest_ok & sig_ok & mom.notna() & rev.notna() & vola.notna() & liq_ok & (sub.iloc[-1] > 0)
    score = score[valid].dropna()

    if len(score) < 10:
        return _empty(tickers)

    n_side = min(50, len(score) // 2)
    if n_side < 5:
        return _empty(tickers)

    ranked = score.sort_values(ascending=False)
    longs = list(ranked.index[:n_side])
    shorts = list(ranked.index[-n_side:])

    per_side = min(0.49, 0.02 * n_side)
    w = per_side / float(n_side)

    weights = pd.Series(0.0, index=tickers, dtype=float)
    weights.loc[longs] = w
    weights.loc[shorts] = -w

    s = float(weights.sum())
    if abs(s) > 0.0:
        weights.loc[longs] = weights.loc[longs] - s / float(len(longs))

    weights = weights.clip(-0.02, 0.02)
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights.reindex(tickers)
