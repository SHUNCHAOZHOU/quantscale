import numpy as np
import pandas as pd
import math


def _get_df(history, names):
    for name in names:
        obj = history.get(name) if isinstance(history, dict) else None
        if isinstance(obj, pd.DataFrame):
            return obj
    return None


def _zscore(s):
    s = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 2:
        return pd.Series(0.0, index=s.index)
    mu = s.mean()
    sd = s.std()
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sd).fillna(0.0)


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    zero = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return zero

    price = _get_df(history, ['adjusted_close', 'close'])
    if price is None:
        return zero
    try:
        price = price.reindex(columns=tickers).astype(float)
    except Exception:
        return zero
    if price.shape[0] < 80:
        return zero

    last_pos = price.shape[0] - 1
    block = 21
    signal_pos = (last_pos // block) * block
    if signal_pos < 60:
        return zero

    last_close = price.iloc[last_pos]
    valid = last_close.notna() & np.isfinite(last_close) & (last_close > 2.0)
    counts = price.iloc[:last_pos + 1].notna().sum()
    valid = valid & (counts >= 60)

    elig = _get_df(history, ['eligible'])
    if elig is not None and len(elig.index) > 0:
        try:
            e = elig.reindex(columns=tickers).iloc[-1]
            valid = valid & e.fillna(False).astype(bool)
        except Exception:
            pass

    vol_df = _get_df(history, ['volume'])
    raw_close = _get_df(history, ['close'])
    if vol_df is not None:
        try:
            v = vol_df.reindex(columns=tickers).iloc[-20:].astype(float)
            if raw_close is not None:
                c = raw_close.reindex(columns=tickers).iloc[-20:].astype(float)
            else:
                c = price.iloc[-20:].astype(float)
            adv = (v * c).median()
            valid = valid & adv.notna() & np.isfinite(adv) & (adv > 250000.0)
        except Exception:
            pass

    p = price.iloc[:signal_pos + 1]
    if p.shape[0] < 61:
        return zero
    last_sig = p.iloc[-1]
    rets = p.pct_change(fill_method=None)

    def lag_ret(lag):
        if p.shape[0] > lag:
            return p.iloc[-1] / p.iloc[-1 - lag] - 1.0
        return pd.Series(np.nan, index=tickers)

    def between_ret(recent_lag, old_lag):
        if p.shape[0] > old_lag:
            return p.iloc[-1 - recent_lag] / p.iloc[-1 - old_lag] - 1.0
        return pd.Series(np.nan, index=tickers)

    r6 = lag_ret(126)
    r12 = lag_ret(252)
    mom12 = between_ret(21, 252) if p.shape[0] > 252 else pd.Series(np.nan, index=tickers)
    mom6 = between_ret(21, 126) if p.shape[0] > 126 else pd.Series(np.nan, index=tickers)
    mom = mom12.copy()
    mom = mom.where(mom.notna(), mom6)
    mom = mom.where(mom.notna(), r12)
    mom = mom.where(mom.notna(), r6)

    vol60 = rets.tail(60).std(ddof=0)
    vol60 = vol60.replace([np.inf, -np.inf], np.nan).clip(lower=0.005)
    ann_vol = vol60 * math.sqrt(252.0)

    mkt = rets.mean(axis=1)
    win = min(120, max(20, rets.shape[0] - 1))
    R = rets.tail(win).to_numpy(dtype=float)
    M = mkt.tail(win).to_numpy(dtype=float)
    if R.size == 0:
        beta = pd.Series(1.0, index=tickers)
    else:
        mask = np.isfinite(R) & np.isfinite(M)[:, None]
        nobs = mask.sum(axis=0)
        Rv = np.where(mask, R, np.nan)
        Mv = np.where(mask, M[:, None], np.nan)
        with np.errstate(all='ignore'):
            mR = np.nanmean(Rv, axis=0)
            mM = np.nanmean(Mv, axis=0)
            cov = np.nansum((Rv - mR) * (Mv - mM), axis=0) / np.maximum(nobs - 1, 1)
            var = np.nansum((Mv - mM) ** 2, axis=0) / np.maximum(nobs - 1, 1)
            beta_vals = cov / var
        beta_vals = np.where(np.isfinite(beta_vals) & (nobs >= 40), beta_vals, 1.0)
        beta = pd.Series(beta_vals, index=tickers)

    if p.shape[0] > 252 and rets.shape[0] >= 252:
        mkt_cum = float(np.expm1(np.nansum(np.log1p(mkt.iloc[-252:-21].to_numpy(dtype=float)))))
        resid = mom12 - beta * mkt_cum
        resid = resid.where(resid.notna(), mom - beta * mkt_cum)
    elif p.shape[0] > 126 and rets.shape[0] >= 126:
        mkt_cum = float(np.expm1(np.nansum(np.log1p(mkt.iloc[-126:-21].to_numpy(dtype=float)))))
        resid = mom6 - beta * mkt_cum
        resid = resid.where(resid.notna(), mom - beta * mkt_cum)
    else:
        resid = mom

    mom_adj = mom / ann_vol
    resid_adj = resid / ann_vol
    r6_adj = r6 / ann_vol
    score = 0.55 * _zscore(mom_adj) + 0.30 * _zscore(resid_adj) + 0.15 * _zscore(r6_adj)
    score = score - 0.05 * _zscore(ann_vol)

    valid_sig = valid & last_sig.notna() & np.isfinite(last_sig) & score.notna() & np.isfinite(score) & vol60.notna() & (vol60 > 0)
    if valid_sig.sum() < 6:
        alt = mom.where(valid & mom.notna())
        if alt.notna().sum() < 6:
            return zero
        score = _zscore(alt)
        valid_sig = alt.notna() & np.isfinite(score)

    s = score[valid_sig].replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 4:
        return zero

    n_side = min(25, len(s) // 2)
    if n_side < 1:
        return zero
    ordered = s.sort_values(ascending=False)
    longs = ordered.index[:n_side]
    shorts = ordered.index[-n_side:]

    if len(mkt.dropna()) >= 60:
        mvol = float(mkt.tail(60).std(ddof=0) * math.sqrt(252.0))
    else:
        mvol = 0.20
    if not np.isfinite(mvol) or mvol <= 0:
        mvol = 0.20
    scale = min(1.0, max(0.35, 0.20 / mvol))
    side_gross = 0.5 * scale
    w = min(0.02, side_gross / float(n_side))

    weights = pd.Series(0.0, index=tickers, dtype=float)
    weights.loc[longs] = w
    weights.loc[shorts] = -w

    net = weights.sum()
    if abs(net) > 1e-12:
        nz = weights.index[weights != 0.0]
        if len(nz) > 0:
            weights.loc[nz] -= net / len(nz)

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
    gross = weights.abs().sum()
    if gross > 1.0:
        weights = weights * (1.0 / gross)
    net = weights.sum()
    if abs(net) > 1e-12:
        nz = weights.index[weights != 0.0]
        if len(nz) > 0:
            weights.loc[nz] -= net / len(nz)
    return weights.clip(-0.02, 0.02).fillna(0.0).astype(float)