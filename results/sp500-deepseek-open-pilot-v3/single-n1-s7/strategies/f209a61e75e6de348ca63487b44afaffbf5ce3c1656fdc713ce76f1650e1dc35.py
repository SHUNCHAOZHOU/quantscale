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
    if len(dates) < 130:
        return _zeros(tickers)

    yv = np.asarray(dates.year) * 12 + np.asarray(dates.month)
    if len(yv) >= 2:
        breaks = np.where(yv[1:] != yv[:-1])[0]
    else:
        breaks = np.array([], dtype=int)
    sig_idx = int(breaks[-1]) if len(breaks) > 0 else len(dates) - 1

    start_idx = max(0, sig_idx - 300)
    sub = adj.iloc[start_idx:sig_idx + 1].ffill(limit=5)
    sub = sub.where(sub > 0)
    n = len(sub)
    if n < 45:
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
    R = rets.values.astype(float)
    mcols = len(tickers)
    if R.ndim != 2 or R.shape[1] != mcols or R.shape[0] < 30:
        return _zeros(tickers)

    # NaN-aware equal-weighted market return per session
    with np.errstate(invalid='ignore'):
        mk = np.nanmean(R, axis=1)
    mk = np.where(np.isfinite(mk), mk, 0.0)

    # trailing betas vs equal-weighted market (vectorised, NaN-masked)
    w = int(min(150, R.shape[0]))
    Wb = R[-w:]
    mb = mk[-w:]
    mask = np.isfinite(Wb) & np.isfinite(mb)[:, None]
    cnt = mask.sum(axis=0)
    cnts = np.maximum(cnt, 1)
    Wf = np.where(mask, Wb, 0.0)
    mf = np.where(mask, mb[:, None], 0.0)
    wmean = Wf.sum(axis=0) / cnts
    mmean = mf.sum(axis=0) / cnts
    dm = mf - mmean[None, :]
    dw = Wf - wmean[None, :]
    den = (dm * dm).sum(axis=0)
    num = (dm * dw).sum(axis=0)
    beta = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), 1.0)
    beta = np.where((cnt >= 50) & np.isfinite(beta), beta, 1.0)
    beta = np.clip(beta, 0.2, 3.0)

    resid = R - mk[:, None] * beta[None, :]

    def _rsum(k, minc):
        if resid.shape[0] < k:
            return np.full(mcols, np.nan)
        blk = resid[-k:]
        good = np.isfinite(blk)
        s = np.where(good, blk, 0.0).sum(axis=0)
        c = good.sum(axis=0)
        return np.where(c >= minc, s, np.nan)

    rev21 = pd.Series(-_rsum(21, 10), index=tickers, dtype=float)
    rev63 = pd.Series(-_rsum(63, 30), index=tickers, dtype=float)
    rev126 = pd.Series(-_rsum(126, 60), index=tickers, dtype=float)

    vola = pd.Series(np.nan, index=tickers, dtype=float)
    try:
        vola = rets.tail(120).std(skipna=True) * np.sqrt(252.0)
        vola = vola.reindex(tickers)
    except Exception:
        vola = pd.Series(np.nan, index=tickers, dtype=float)
    qvol = (-vola).rank(pct=True, na_option='keep')

    mom = pd.Series(np.nan, index=tickers, dtype=float)
    if n >= 252:
        mom = sub.iloc[-21] / sub.iloc[-252] - 1.0
    qmom = mom.rank(pct=True, na_option='keep')

    def _composite(a21, a63, a126):
        q21 = a21.rank(pct=True, na_option='keep')
        q63 = a63.rank(pct=True, na_option='keep')
        q126 = a126.rank(pct=True, na_option='keep')
        q126 = q126.fillna(q63).fillna(q21)
        q63 = q63.fillna(q21)
        return q21, q63, q126

    q21, q63, q126 = _composite(rev21, rev63, rev126)
    score = 0.45 * q21 + 0.30 * q63 + 0.10 * q126 + 0.15 * qvol

    base_valid = sig_ok & latest_ok & liq_ok & qvol.notna() & last.notna()
    valid = base_valid & q21.notna() & q63.notna()
    score = score[valid].replace([np.inf, -np.inf], np.nan).dropna()

    if len(score) < 24:
        # fallback: plain price reversal (round-6 core) if residual data is thin
        p21 = pd.Series(np.nan, index=tickers, dtype=float)
        p63 = pd.Series(np.nan, index=tickers, dtype=float)
        if n >= 22:
            p21 = -(last / sub.iloc[-22] - 1.0)
        if n >= 64:
            p63 = -(last / sub.iloc[-64] - 1.0)
        p63 = p63.fillna(p21)
        b21, b63, b126 = _composite(p21, p63, p21)
        score2 = 0.55 * b21 + 0.30 * b63 + 0.15 * qvol
        valid2 = base_valid & b21.notna() & b63.notna()
        score = score2[valid2].replace([np.inf, -np.inf], np.nan).dropna()

    if len(score) < 20:
        return _zeros(tickers)

    disp_scale = 1.0
    try:
        with np.errstate(invalid='ignore'):
            cs = pd.Series(np.nanstd(R, axis=1), index=rets.index).replace([np.inf, -np.inf], np.nan).dropna()
        if len(cs) >= 80:
            smooth = cs.rolling(21, min_periods=10).mean().dropna()
            if len(smooth) >= 40:
                cur = float(smooth.iloc[-1])
                window = smooth.tail(int(min(252, len(smooth))))
                pct = float((window.values <= cur).mean())
                disp_scale = 0.45 + 0.55 * max(0.0, min(1.0, pct))
    except Exception:
        disp_scale = 1.0

    n_side = int(min(50, len(score) // 2))
    if n_side < 12:
        return _zeros(tickers)

    ranked = score.sort_values(ascending=False)
    longs = list(ranked.index[:n_side])

    short_pool = list(ranked.index[n_side:])
    if len(short_pool) > n_side:
        try:
            mslice = qmom.reindex(short_pool)
            filtered = [
                t for t in short_pool
                if not (pd.notna(mslice.loc[t]) and float(mslice.loc[t]) > 0.90)
            ]
            if len(filtered) >= n_side:
                short_pool = filtered
        except Exception:
            pass
    shorts = short_pool[-n_side:]

    per_side = min(0.49, 0.02 * n_side) * disp_scale
    wgt = per_side / float(n_side)

    weights = pd.Series(0.0, index=tickers, dtype=float)
    weights.loc[longs] = wgt
    weights.loc[shorts] = -wgt

    weights = weights.clip(-0.02, 0.02)
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights.reindex(tickers).fillna(0.0)
