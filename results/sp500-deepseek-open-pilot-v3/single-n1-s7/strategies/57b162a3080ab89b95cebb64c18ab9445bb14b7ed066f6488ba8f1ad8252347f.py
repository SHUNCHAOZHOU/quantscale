import numpy as np
import pandas as pd


def _zeros(tickers):
    return pd.Series(0.0, index=pd.Index(tickers), dtype=float)


def _rsum(M, k, minc):
    m = M.shape[1]
    if M.shape[0] < k:
        return np.full(m, np.nan)
    blk = M[-k:]
    good = np.isfinite(blk)
    s = np.where(good, blk, 0.0).sum(axis=0)
    c = good.sum(axis=0)
    return np.where(c >= minc, s, np.nan)


def generate_weights(history, universe):
    tickers = pd.Index(universe)
    if len(tickers) == 0:
        return pd.Series(dtype=float)

    adj = history.get('adjusted_close')
    if adj is None:
        adj = history.get('close')
    if adj is None or len(adj.index) == 0:
        return _zeros(tickers)

    adj = adj.reindex(columns=tickers)
    dates = adj.index
    if len(dates) < 130:
        return _zeros(tickers)

    yv = np.asarray(dates.year) * 12 + np.asarray(dates.month)
    if len(yv) >= 2:
        brk = np.where(yv[1:] != yv[:-1])[0]
    else:
        brk = np.array([], dtype=int)
    sig_idx = int(brk[-1]) if len(brk) > 0 else len(dates) - 1

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
            lr = liq.rank(pct=True, na_option='keep')
            liq_ok = liq.notna() & (liq > 0) & (lr >= 0.20)
        except Exception:
            pass

    last = sub.iloc[-1]
    rets = sub.pct_change(fill_method=None)
    R = rets.values.astype(float)
    m = len(tickers)
    if R.ndim != 2 or R.shape[1] != m or R.shape[0] < 30:
        return _zeros(tickers)

    with np.errstate(invalid='ignore'):
        mk = np.nanmean(R, axis=1)
    mk = np.where(np.isfinite(mk), mk, 0.0)

    # ---- trailing betas vs equal-weighted market (NaN-masked, vectorised) ----
    wlen = int(min(150, R.shape[0]))
    Wb = R[-wlen:]
    mb = mk[-wlen:]
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

    # ---- volatility normalisers ----
    with np.errstate(invalid='ignore'):
        sig_r = np.nanstd(R[-120:], axis=0) * np.sqrt(252.0) if R.shape[0] >= 40 else np.full(m, np.nan)
        sig_e = np.nanstd(resid[-120:], axis=0) * np.sqrt(252.0) if R.shape[0] >= 40 else np.full(m, np.nan)
    sig_use = np.where(np.isfinite(sig_r) & (sig_r > 0.02), sig_r, 0.30)
    sig_euse = np.where(np.isfinite(sig_e) & (sig_e > 0.02), sig_e, 0.30)

    def _zrev(M, k, minc, sg):
        return -_rsum(M, k, minc) / (sg * np.sqrt(k / 252.0))

    rv21 = pd.Series(_zrev(R, 21, 10, sig_use), index=tickers, dtype=float)
    rv63 = pd.Series(_zrev(R, 63, 30, sig_use), index=tickers, dtype=float)
    re21 = pd.Series(_zrev(resid, 21, 10, sig_euse), index=tickers, dtype=float)
    re63 = pd.Series(_zrev(resid, 63, 30, sig_euse), index=tickers, dtype=float)

    vola = pd.Series(sig_r, index=tickers, dtype=float)
    qvol = (-vola).rank(pct=True, na_option='keep')

    mom = pd.Series(np.nan, index=tickers, dtype=float)
    if n >= 252:
        mom = sub.iloc[-21] / sub.iloc[-252] - 1.0
    qmom = mom.rank(pct=True, na_option='keep')

    comp = pd.DataFrame(index=tickers)
    comp['re21'] = re21.rank(pct=True, na_option='keep')
    comp['re63'] = re63.rank(pct=True, na_option='keep')
    comp['rv21'] = rv21.rank(pct=True, na_option='keep')
    comp['rv63'] = rv63.rank(pct=True, na_option='keep')
    comp['vol'] = qvol

    comp['re63'] = comp['re63'].fillna(comp['re21'])
    comp['rv63'] = comp['rv63'].fillna(comp['rv21'])

    wts = {'re21': 0.20, 're63': 0.15, 'rv21': 0.28, 'rv63': 0.22, 'vol': 0.15}
    num = pd.Series(0.0, index=tickers, dtype=float)
    cvg = pd.Series(0.0, index=tickers, dtype=float)
    for k, wk in wts.items():
        col = comp[k]
        ok = col.notna()
        num = num.add(col.fillna(0.0) * wk, fill_value=0.0)
        cvg = cvg.add(ok.astype(float) * wk, fill_value=0.0)

    score = num / cvg.where(cvg > 0)

    valid = sig_ok & latest_ok & liq_ok & last.notna() & (cvg >= 0.40)
    score = score[valid].replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 24:
        return _zeros(tickers)

    # ---- dispersion regime overlay ----
    disp_scale = 1.0
    try:
        with np.errstate(invalid='ignore'):
            cs = pd.Series(np.nanstd(R, axis=1), index=rets.index)
        cs = cs.replace([np.inf, -np.inf], np.nan).dropna()
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
    long_set = set(longs)

    # ---- short candidate pool: bottom of the score distribution ----
    pool_size = int(min(3 * n_side, max(0, len(ranked) - n_side)))
    if pool_size < n_side:
        pool = list(ranked.index[n_side:])
    else:
        pool = list(ranked.index[-pool_size:])
    pool = [t for t in pool if t not in long_set]

    if len(pool) > n_side:
        try:
            qs = qmom.reindex(pool)
            filt = [t for t in pool if not (pd.notna(qs.get(t, np.nan)) and float(qs.get(t, np.nan)) > 0.90)]
            if len(filt) >= n_side:
                pool = filt
        except Exception:
            pass

    # ---- beta-matched short selection ----
    betas = pd.Series(beta, index=tickers, dtype=float).fillna(1.0)
    if len(pool) < n_side:
        shorts = list(ranked.index[-n_side:])
    else:
        lb = betas.reindex(longs).fillna(1.0).values.astype(float)
        pb = betas.reindex(pool).fillna(1.0).values.astype(float)
        used = np.zeros(len(pool), dtype=bool)
        shorts = []
        for i in np.argsort(lb, kind='stable'):
            d = np.abs(pb - lb[i])
            d = np.where(used, np.inf, d)
            j = int(np.argmin(d))
            if not np.isfinite(d[j]):
                break
            used[j] = True
            shorts.append(pool[j])
        if len(shorts) < n_side:
            for j in range(len(pool)):
                if len(shorts) >= n_side:
                    break
                if not used[j]:
                    used[j] = True
                    shorts.append(pool[j])
        if len(shorts) < n_side:
            shorts = list(ranked.index[-n_side:])

    per_side = min(0.49, 0.02 * n_side) * disp_scale
    wgt = per_side / float(n_side)

    weights = pd.Series(0.0, index=tickers, dtype=float)
    weights.loc[longs] = wgt
    weights.loc[shorts] = -wgt

    weights = weights.clip(-0.02, 0.02)
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights.reindex(tickers).fillna(0.0)
