import numpy as np
import pandas as pd


def _zeros(tickers):
    return pd.Series(0.0, index=pd.Index(tickers), dtype=float)


def generate_weights(history, universe):
    tickers = pd.Index(universe)
    if len(tickers) == 0:
        return pd.Series(dtype=float)

    adj_full = history.get('adjusted_close')
    if adj_full is None:
        adj_full = history.get('close')
    if adj_full is None or len(adj_full.index) == 0:
        return _zeros(tickers)

    adj = adj_full.reindex(columns=tickers)
    dates = adj.index
    if len(dates) < 70:
        return _zeros(tickers)

    ym = np.asarray(dates.year) * 12 + np.asarray(dates.month)
    if len(ym) >= 2:
        month_ends = np.where(ym[1:] != ym[:-1])[0]
    else:
        month_ends = np.array([], dtype=int)

    if len(month_ends) > 0:
        sig_idx = int(month_ends[-1])
    else:
        sig_idx = len(dates) - 1

    start_idx = max(0, sig_idx - 140)
    sub = adj.iloc[start_idx:sig_idx + 1].ffill(limit=5)
    sub = sub.where(sub > 0)
    n = len(sub)
    if n < 25:
        return _zeros(tickers)

    elig = history.get('eligible')
    sig_ok = sub.iloc[-1].notna()
    latest_ok = adj.iloc[-1].notna()
    if elig is not None:
        try:
            e = elig.reindex(index=dates, columns=tickers)
            sig_ok = sig_ok & e.iloc[sig_idx].fillna(False).astype(bool)
            latest_ok = latest_ok & e.iloc[-1].fillna(False).astype(bool)
        except Exception:
            pass

    liq_ok = pd.Series(True, index=tickers, dtype=bool)
    vol_df = history.get('volume')
    close_df = history.get('close')
    if vol_df is not None and close_df is not None:
        try:
            v = vol_df.reindex(index=dates, columns=tickers).iloc[start_idx:sig_idx + 1]
            c = close_df.reindex(index=dates, columns=tickers).iloc[start_idx:sig_idx + 1]
            liq = (c * v).tail(60).median()
            liq_rank = liq.rank(pct=True, na_option='keep')
            liq_ok = liq.notna() & (liq > 0) & (liq_rank >= 0.20)
        except Exception:
            pass

    last = sub.iloc[-1]

    rev5 = pd.Series(np.nan, index=tickers, dtype=float)
    if n >= 6:
        rev5 = -(last / sub.iloc[-6] - 1.0)

    rev10 = pd.Series(np.nan, index=tickers, dtype=float)
    if n >= 11:
        rev10 = -(last / sub.iloc[-11] - 1.0)

    rev21 = pd.Series(np.nan, index=tickers, dtype=float)
    if n >= 22:
        rev21 = -(last / sub.iloc[-22] - 1.0)

    rev63 = pd.Series(np.nan, index=tickers, dtype=float)
    if n >= 64:
        rev63 = -(last / sub.iloc[-64] - 1.0)

    vola = pd.Series(np.nan, index=tickers, dtype=float)
    try:
        rets = sub.pct_change(fill_method=None)
        vola = rets.tail(80).std(skipna=True) * np.sqrt(252.0)
    except Exception:
        vola = pd.Series(np.nan, index=tickers, dtype=float)

    r5 = rev5.rank(pct=True, na_option='keep')
    r10 = rev10.rank(pct=True, na_option='keep')
    r21 = rev21.rank(pct=True, na_option='keep')
    r63 = rev63.rank(pct=True, na_option='keep')
    r63 = r63.fillna(r21)
    rvol = (-vola).rank(pct=True, na_option='keep')

    score = 0.30 * r5 + 0.25 * r10 + 0.20 * r21 + 0.10 * r63 + 0.15 * rvol

    valid = (
        sig_ok & latest_ok & liq_ok &
        r5.notna() & r10.notna() & r21.notna() &
        rvol.notna() & last.notna()
    )
    score = score[valid].replace([np.inf, -np.inf], np.nan).dropna()

    if len(score) < 20:
        return _zeros(tickers)

    n_side = min(50, len(score) // 2)
    if n_side < 10:
        return _zeros(tickers)

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
