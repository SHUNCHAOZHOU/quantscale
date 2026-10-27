import numpy as np
import pandas as pd
import math


def _reindex(df, tickers):
    if df is None:
        return None
    try:
        return df.reindex(columns=tickers)
    except Exception:
        return None


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=tickers, dtype=float)
    if not isinstance(history, dict):
        return zero

    price = _reindex(history.get('adjusted_close'), tickers)
    if price is None or price.dropna(how='all').empty:
        price = _reindex(history.get('close'), tickers)
    if price is None or price.empty:
        return zero
    price = price.astype(float)
    if len(price) < 120:
        return zero

    dates = price.index
    last_pos = len(dates) - 1
    sig_pos = last_pos
    try:
        periods = dates.to_period('M')
        cur = periods[-1]
        prior = np.where(periods < cur)[0]
        if len(prior) > 0:
            sig_pos = int(prior[-1])
    except Exception:
        sig_pos = last_pos
    if sig_pos < 80:
        return zero

    p = price.iloc[:sig_pos + 1].tail(900)
    if len(p) < 80:
        return zero

    raw = _reindex(history.get('close'), tickers)
    if raw is None:
        raw = price

    last_adj = price.iloc[-1].astype(float)
    last_sig = p.iloc[-1].astype(float)
    valid = last_adj.notna() & np.isfinite(last_adj) & (last_adj > 0.0)
    valid = valid & last_sig.notna() & np.isfinite(last_sig) & (last_sig > 0.0)

    if raw is not None and len(raw) == len(price):
        last_raw = raw.iloc[-1].astype(float)
        valid = valid & last_raw.notna() & np.isfinite(last_raw) & (last_raw > 2.0)

    cnt = p.notna().sum()
    valid = valid & (cnt >= min(120, max(60, len(p) // 2)))

    elig = _reindex(history.get('eligible'), tickers)
    if elig is not None and len(elig) > 0:
        try:
            e = elig.iloc[-1].reindex(tickers).fillna(False).astype(bool)
            valid = valid & e
        except Exception:
            pass

    vol_df = _reindex(history.get('volume'), tickers)
    if vol_df is not None:
        try:
            v = vol_df.iloc[:sig_pos + 1].tail(21).astype(float)
            c = raw.iloc[:sig_pos + 1].tail(21).astype(float)
            adv = (v * c).median()
            adv = pd.to_numeric(adv, errors='coerce')
            valid = valid & adv.notna() & np.isfinite(adv) & (adv > 250000.0)
        except Exception:
            pass

    rets = p.pct_change(fill_method=None)
    vol60 = rets.tail(60).std(ddof=0).replace([np.inf, -np.inf], np.nan)
    ann_vol = (vol60 * math.sqrt(252.0)).clip(lower=0.05, upper=2.0)
    valid = valid & ann_vol.notna() & np.isfinite(ann_vol) & (ann_vol > 0)

    mkt = rets.mean(axis=1)
    win = min(252, max(20, len(rets) - 1))
    R = rets.tail(win).to_numpy(dtype=float)
    M = mkt.tail(win).to_numpy(dtype=float)
    mask = np.isfinite(R) & np.isfinite(M)[:, None]
    nobs = mask.sum(axis=0)
    R0 = np.where(mask, R, 0.0)
    M0 = np.where(mask, M[:, None], 0.0)
    meanR = R0.sum(axis=0) / np.maximum(nobs, 1)
    meanM = M0.sum(axis=0) / np.maximum(nobs, 1)
    cov = ((R0 - meanR) * (M0 - meanM)).sum(axis=0) / np.maximum(nobs - 1, 1)
    var = ((M0 - meanM) ** 2).sum(axis=0) / np.maximum(nobs - 1, 1)
    beta_vals = np.where((nobs >= 60) & (var > 1e-12), cov / var, np.nan)
    beta_vals = np.clip(beta_vals, -3.0, 3.0)
    beta = pd.Series(beta_vals, index=tickers)
    valid = valid & beta.notna() & np.isfinite(beta)

    def skip_ret(look, off=22):
        if len(p) > look + off:
            end = p.iloc[-off].astype(float)
            start = p.iloc[-off - look].astype(float)
            return end / start - 1.0
        return pd.Series(np.nan, index=tickers)

    lt1 = skip_ret(252)
    lt2 = skip_ret(504)
    lt3 = skip_ret(756)
    lt = pd.concat([lt1, lt2, lt3], axis=1).mean(axis=1, skipna=True)
    lt = lt.replace([np.inf, -np.inf], np.nan)

    def mkt_cum(recent_lag, old_lag):
        if len(mkt) > old_lag:
            arr = mkt.iloc[-old_lag:-recent_lag].to_numpy(dtype=float)
            return float(np.expm1(np.nansum(np.log1p(arr))))
        return np.nan

    resid12 = pd.Series(np.nan, index=tickers)
    resid6 = pd.Series(np.nan, index=tickers)
    mom3 = pd.Series(np.nan, index=tickers)

    if len(p) > 253:
        mom12 = p.iloc[-22].astype(float) / p.iloc[-253].astype(float) - 1.0
        mc = mkt_cum(21, 252)
        if np.isfinite(mc):
            resid12 = mom12 - beta * mc

    if len(p) > 148:
        mom6 = p.iloc[-22].astype(float) / p.iloc[-148].astype(float) - 1.0
        mc = mkt_cum(21, 147)
        if np.isfinite(mc):
            resid6 = mom6 - beta * mc

    if len(p) > 64:
        mom3 = p.iloc[-1].astype(float) / p.iloc[-64].astype(float) - 1.0

    resid = resid12.copy()
    resid = resid.where(resid.notna(), resid6)
    resid = resid.where(resid.notna(), mom3)
    resid = resid.replace([np.inf, -np.inf], np.nan)

    rev_adj = (-lt) / ann_vol
    mom_adj = resid / ann_vol
    has_sig = (rev_adj.notna() & np.isfinite(rev_adj)) | (mom_adj.notna() & np.isfinite(mom_adj))
    valid = valid & has_sig

    if valid.sum() < 8:
        return zero

    rank_rev = rev_adj.where(valid).rank(method='average', pct=True).fillna(0.5)
    rank_mom = mom_adj.where(valid).rank(method='average', pct=True).fillna(0.5)
    rank_low = (-ann_vol).where(valid).rank(method='average', pct=True).fillna(0.5)
    score = 0.45 * rank_rev + 0.45 * rank_mom + 0.10 * rank_low
    score = score.replace([np.inf, -np.inf], np.nan)

    bvals = beta[valid]
    if len(bvals) < 8:
        return zero
    qs = np.nanpercentile(bvals.to_numpy(dtype=float), [20, 40, 60, 80])
    bucket = pd.Series(np.digitize(beta.to_numpy(dtype=float), qs), index=tickers)

    long_idx = []
    short_idx = []
    for b in range(5):
        sub = score[valid & (bucket == b)].dropna().sort_values()
        if len(sub) < 2:
            continue
        k = min(8, len(sub) // 2)
        if k < 1:
            continue
        long_idx.extend(sub.index[-k:].tolist())
        short_idx.extend(sub.index[:k].tolist())

    total = len(long_idx)
    if total < 4 or len(short_idx) != total:
        return zero

    if mkt.notna().sum() >= 60:
        mvol = float(mkt.tail(60).std(ddof=0) * math.sqrt(252.0))
    else:
        mvol = 0.20
    if not np.isfinite(mvol) or mvol <= 0:
        mvol = 0.20

    scale = min(1.0, max(0.50, 0.20 / mvol))
    side_gross = 0.5 * scale
    w = min(0.02, side_gross / float(total))

    weights = zero.copy()
    weights.loc[long_idx] = w
    weights.loc[short_idx] = -w

    net = weights.sum()
    if abs(net) > 1e-12:
        nz = weights.index[weights != 0.0]
        if len(nz) > 0:
            weights.loc[nz] -= net / len(nz)

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)

    net = weights.sum()
    if abs(net) > 1e-12:
        nz = weights.index[weights != 0.0]
        if len(nz) > 0:
            weights.loc[nz] -= net / len(nz)
            weights = weights.clip(-0.02, 0.02)

    gross = weights.abs().sum()
    if gross > 1.0:
        weights = weights * (1.0 / gross)

    return weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)