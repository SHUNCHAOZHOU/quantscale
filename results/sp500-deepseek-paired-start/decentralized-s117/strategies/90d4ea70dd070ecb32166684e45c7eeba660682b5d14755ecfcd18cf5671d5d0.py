import numpy as np
import pandas as pd
import math

def _zscore(s):
    s = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 2:
        return pd.Series(0.0, index=s.index)
    mu = s.mean()
    sd = s.std()
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sd).fillna(0.0)

def _get_df(history, names, tickers):
    for name in names:
        df = history.get(name) if isinstance(history, dict) else None
        if isinstance(df, pd.DataFrame):
            try:
                return df.reindex(columns=tickers)
            except Exception:
                pass
    return None

def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    zero = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return zero

    price = _get_df(history, ['adjusted_close', 'close'], tickers)
    if price is None or price.empty:
        return zero
    price = price.astype(float)
    if price.shape[0] < 280:
        return zero

    last_pos = price.shape[0] - 1
    block = 21
    signal_pos = (last_pos // block) * block
    if signal_pos < 260:
        return zero

    p = price.iloc[:signal_pos + 1]
    if p.shape[0] < 260:
        return zero

    last_sig = p.iloc[-1]
    last_price = price.iloc[-1]
    valid = (last_sig.notna() & np.isfinite(last_sig) & (last_sig > 2.0)
             & last_price.notna() & np.isfinite(last_price) & (last_price > 2.0))
    counts = p.notna().sum()
    valid = valid & (counts >= 120)

    elig = _get_df(history, ['eligible'], tickers)
    if elig is not None and len(elig.index) > 0:
        try:
            e = elig.iloc[-1].fillna(False).astype(bool)
            valid = valid & e
        except Exception:
            pass

    raw = _get_df(history, ['close'], tickers)
    if raw is None:
        raw = price
    vol_df = _get_df(history, ['volume'], tickers)
    if vol_df is not None:
        try:
            v = vol_df.tail(60).astype(float)
            c = raw.tail(60).astype(float)
            adv = (v * c).median()
            valid = valid & adv.notna() & np.isfinite(adv) & (adv > 250000.0)
        except Exception:
            pass

    rets = p.pct_change(fill_method=None)
    mkt = rets.mean(axis=1)

    win = min(252, max(20, rets.shape[0] - 1))
    if win >= 60:
        R = rets.tail(win).to_numpy(dtype=float)
        M = mkt.tail(win).to_numpy(dtype=float)
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
    else:
        beta = pd.Series(1.0, index=tickers)

    vol60 = rets.tail(60).std(ddof=0) * math.sqrt(252.0)
    vol60 = vol60.replace([np.inf, -np.inf], np.nan).clip(lower=0.05, upper=2.0)
    valid = valid & vol60.notna() & (vol60 > 0)

    def lag_ret(lag):
        if p.shape[0] > lag:
            return p.iloc[-1] / p.iloc[-1 - lag] - 1.0
        return pd.Series(np.nan, index=tickers)

    r12 = lag_ret(252)
    r6 = lag_ret(126)

    if p.shape[0] > 253:
        mom12 = p.iloc[-22] / p.iloc[-253] - 1.0
    elif p.shape[0] > 127:
        mom12 = p.iloc[-22] / p.iloc[-127] - 1.0
    else:
        mom12 = r12

    if p.shape[0] > 127:
        mom6 = p.iloc[-22] / p.iloc[-127] - 1.0
    elif p.shape[0] > 64:
        mom6 = p.iloc[-22] / p.iloc[-64] - 1.0
    else:
        mom6 = r6

    def mkt_cum(start, end):
        if len(mkt) >= abs(start):
            arr = mkt.iloc[start:end].to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                return float(np.expm1(np.nansum(np.log1p(arr))))
        return 0.0

    if p.shape[0] > 253:
        mc12 = mkt_cum(-252, -21)
        resid12 = mom12 - beta * mc12
    else:
        resid12 = mom12

    if p.shape[0] > 127:
        mc6 = mkt_cum(-126, -21)
        resid6 = mom6 - beta * mc6
    else:
        resid6 = mom6

    mom12_adj = mom12 / vol60
    resid12_adj = resid12 / vol60
    mom6_adj = mom6 / vol60

    score = (0.50 * _zscore(mom12_adj)
             + 0.30 * _zscore(resid12_adj)
             + 0.20 * _zscore(mom6_adj)
             - 0.10 * _zscore(vol60))
    score = score.replace([np.inf, -np.inf], np.nan)
    valid = valid & score.notna() & np.isfinite(score)

    if valid.sum() < 8:
        alt = mom12.where(valid & mom12.notna())
        if alt.notna().sum() < 8:
            return zero
        score = _zscore(alt)
        valid = alt.notna() & np.isfinite(score)

    beta_valid = beta[valid]
    if beta_valid.notna().sum() < 8:
        return zero
    qs = np.nanpercentile(beta_valid, [20, 40, 60, 80])
    bucket = np.digitize(beta.to_numpy(dtype=float), qs)

    longs = []
    shorts = []
    for b in range(5):
        idx = valid & (bucket == b) & score.notna()
        if idx.sum() < 2:
            continue
        s = score[idx].sort_values()
        k = min(5, len(s) // 2)
        if k < 1:
            continue
        longs.extend(s.index[-k:].tolist())
        shorts.extend(s.index[:k].tolist())

    n_side = min(len(longs), len(shorts), 25)
    if n_side < 1:
        return zero
    longs = longs[:n_side]
    shorts = shorts[:n_side]

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