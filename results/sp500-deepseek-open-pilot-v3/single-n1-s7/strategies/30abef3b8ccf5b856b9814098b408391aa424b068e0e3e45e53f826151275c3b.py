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

    # longer window so a 12-1 momentum can be formed
    start_idx = max(0, sig_idx - 280)
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

    def _rev(back):
        if n >= back + 1:
            return -(last / sub.iloc[-(back + 1)] - 1.0)
        return pd.Series(np.nan, index=tickers, dtype=float)

    rev21 = _rev(21)
    rev63 = _rev(63)
    rev126 = _rev(126)

    # 12-1 momentum: price 21 sessions ago vs 252 sessions ago
    mom = pd.Series(np.nan, index=tickers, dtype=float)
    if n >= 252:
        mom = sub.iloc[-21] / sub.iloc[-252] - 1.0

    vola = pd.Series(np.nan, index=tickers, dtype=float)
    try:
        rets = sub.pct_change(fill_method=None)
        vola = rets.tail(120).std(skipna=True) * np.sqrt(252.0)
    except Exception:
        vola = pd.Series(np.nan, index=tickers, dtype=float)

    r21 = rev21.rank(pct=True, na_option='keep')
    r63 = rev63.rank(pct=True, na_option='keep')
    r126 = rev126.rank(pct=True, na_option='keep')
    rvol = (-vola).rank(pct=True, na_option='keep')
    rmom = mom.rank(pct=True, na_option='keep')

    # tolerate missing longer-horizon info by cascading to shorter ones
    r126 = r126.fillna(r63).fillna(r21)
    r63 = r63.fillna(r21)
    # neutralize momentum for names lacking a full 12-month history
    rmom = rmom.fillna(0.5)

    score = 0.40 * r21 + 0.25 * r63 + 0.12 * r126 + 0.13 * rvol + 0.10 * rmom

    valid = (
        sig_ok & latest_ok & liq_ok &
        r21.notna() & r63.notna() & rvol.notna() & last.notna()
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
    wout = weights.reindex(tickers)
    return wout.fillna(0.0)
