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
    if len(dates) < 120:
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

    start_idx = max(0, sig_idx - 300)
    sub = adj.iloc[start_idx:sig_idx + 1].ffill(limit=5)
    sub = sub.where(sub > 0)
    n = len(sub)
    if n < 40:
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
    rets = sub.pct_change(fill_method=None)
    mkt = rets.mean(axis=1)

    win = rets.tail(150)
    mm = mkt.reindex(win.index)
    mmc = mm - mm.mean()
    denom = float((mmc ** 2).sum())
    if denom <= 0:
        beta = pd.Series(1.0, index=tickers, dtype=float)
    else:
        w = win.fillna(0.0)
        wc = w.sub(w.mean(axis=0), axis=1)
        cov = wc.mul(mmc, axis=0).sum(axis=0)
        beta = cov / denom
        cover = win.notna().mean(axis=0)
        beta = beta.where(cover >= 0.7, 1.0)
        beta = beta.clip(0.2, 3.0).fillna(1.0)

    res = rets.sub(mkt.mul(beta), axis=1)

    def _rsum(k, minc):
        if len(res) < k:
            return pd.Series(np.nan, index=tickers, dtype=float)
        return res.iloc[-k:].sum(min_count=minc, skipna=True)

    r21 = -_rsum(21, 15)
    r63 = -_rsum(63, 40)
    r126 = -_rsum(126, 80)

    vola = rets.tail(120).std(skipna=True) * np.sqrt(252.0)

    mom = pd.Series(np.nan, index=tickers, dtype=float)
    if n >= 252:
        mom = sub.iloc[-21] / sub.iloc[-252] - 1.0

    q21 = r21.rank(pct=True, na_option='keep')
    q63 = r63.rank(pct=True, na_option='keep')
    q126 = r126.rank(pct=True, na_option='keep')
    qvol = (-vola).rank(pct=True, na_option='keep')
    qmom = mom.rank(pct=True, na_option='keep')

    q126 = q126.fillna(q63).fillna(q21)
    q63 = q63.fillna(q21)

    score = 0.45 * q21 + 0.30 * q63 + 0.10 * q126 + 0.15 * qvol

    valid = (
        sig_ok & latest_ok & liq_ok &
        q21.notna() & q63.notna() & qvol.notna() & last.notna()
    )
    score = score[valid].replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 24:
        return _zeros(tickers)

    disp_scale = 1.0
    try:
        cs = rets.std(axis=1, skipna=True).dropna()
        if len(cs) >= 80:
            smooth = cs.rolling(21, min_periods=10).mean().dropna()
            if len(smooth) >= 40:
                cur = float(smooth.iloc[-1])
                window = smooth.tail(min(252, len(smooth)))
                pct = float((window.values <= cur).mean())
                disp_scale = 0.45 + 0.55 * max(0.0, min(1.0, pct))
    except Exception:
        disp_scale = 1.0

    n_side = min(50, len(score) // 2)
    if n_side < 12:
        return _zeros(tickers)

    ranked = score.sort_values(ascending=False)
    longs = list(ranked.index[:n_side])

    short_pool = list(ranked.index[n_side:])
    if len(short_pool) > n_side:
        try:
            mom_slice = qmom.reindex(short_pool)
            filtered = [
                t for t in short_pool
                if not (pd.notna(mom_slice.loc[t]) and float(mom_slice.loc[t]) > 0.90)
            ]
            if len(filtered) >= n_side:
                short_pool = filtered
        except Exception:
            pass
    shorts = short_pool[-n_side:]

    per_side = min(0.49, 0.02 * n_side) * disp_scale
    w = per_side / float(n_side)

    weights = pd.Series(0.0, index=tickers, dtype=float)
    weights.loc[longs] = w
    weights.loc[shorts] = -w

    weights = weights.clip(-0.02, 0.02)
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights.reindex(tickers).fillna(0.0)
