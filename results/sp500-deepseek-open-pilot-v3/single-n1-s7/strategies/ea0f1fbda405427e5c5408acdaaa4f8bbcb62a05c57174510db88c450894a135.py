import numpy as np
import pandas as pd


def _empty(tickers):
    return pd.Series(0.0, index=pd.Index(tickers), dtype=float)


def generate_weights(history, universe):
    tickers = pd.Index(universe)
    if len(tickers) == 0:
        return pd.Series(dtype=float)

    adj_full = history.get('adjusted_close')
    if adj_full is None:
        adj_full = history.get('close')
    if adj_full is None or len(adj_full.index) == 0:
        return _empty(tickers)

    adj = adj_full.reindex(columns=tickers).copy()
    dates = adj.index
    if len(dates) < 30:
        return _empty(tickers)

    weekday = np.asarray(dates.weekday)
    fridays = np.where(weekday == 4)[0]
    if len(fridays) > 0:
        sig_idx = int(fridays[-1])
    else:
        sig_idx = len(dates) - 1

    start_idx = max(0, sig_idx - 130)
    sub = adj.iloc[start_idx:sig_idx + 1].ffill(limit=5)
    sub = sub.where(sub > 0)
    n = len(sub)
    if n < 25:
        return _empty(tickers)

    if 'eligible' in history and history['eligible'] is not None:
        try:
            elig = history['eligible'].reindex(index=dates, columns=tickers)
            sig_ok = elig.iloc[sig_idx].fillna(False).astype(bool)
            latest_ok = elig.iloc[-1].fillna(False).astype(bool)
        except Exception:
            sig_ok = sub.iloc[-1].notna()
            latest_ok = adj.iloc[-1].notna()
    else:
        sig_ok = sub.iloc[-1].notna()
        latest_ok = adj.iloc[-1].notna()

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

    last = sub.iloc[-1]

    rev5 = pd.Series(np.nan, index=tickers, dtype=float)
    if n >= 6:
        rev5 = -(last / sub.iloc[-6] - 1.0)

    rev21 = pd.Series(np.nan, index=tickers, dtype=float)
    if n >= 22:
        rev21 = -(last / sub.iloc[-22] - 1.0)

    vola = pd.Series(np.nan, index=tickers, dtype=float)
    try:
        rets = sub.pct_change(fill_method=None)
        vola = rets.tail(60).std(skipna=True) * np.sqrt(252.0)
    except Exception:
        vola = pd.Series(np.nan, index=tickers, dtype=float)

    rev5_r = rev5.rank(pct=True, na_option='keep')
    rev21_r = rev21.rank(pct=True, na_option='keep')
    lowvol_r = (-vola).rank(pct=True, na_option='keep')

    score = 0.55 * rev5_r + 0.25 * rev21_r + 0.20 * lowvol_r

    valid = (
        sig_ok & latest_ok & liq_ok &
        rev5.notna() & rev21.notna() & vola.notna() & last.notna()
    )
    score = score[valid].replace([np.inf, -np.inf], np.nan).dropna()

    if len(score) < 10:
        return _empty(tickers)

    n_side = min(50, len(score) // 2)
    if n_side < 5:
        return _empty(tickers)

    ranked = score.sort_values(ascending=False)
    longs = ranked.index[:n_side]
    shorts = ranked.index[-n_side:]

    per_side = min(0.49, 0.02 * n_side)
    w = per_side / float(n_side)

    weights = pd.Series(0.0, index=tickers, dtype=float)
    weights.loc[longs] = w
    weights.loc[shorts] = -w

    weights = weights.clip(-0.02, 0.02)
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights.reindex(tickers)
