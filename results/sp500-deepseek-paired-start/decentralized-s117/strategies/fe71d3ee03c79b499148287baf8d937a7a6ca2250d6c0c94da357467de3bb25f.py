import numpy as np
import pandas as pd
import math


def _get_df(history, names, tickers):
    if not isinstance(history, dict):
        return None
    for name in names:
        obj = history.get(name)
        if isinstance(obj, pd.DataFrame):
            try:
                return obj.reindex(columns=tickers)
            except Exception:
                return None
    return None


def _rank_pct(s):
    s = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 2:
        return pd.Series(0.5, index=s.index, dtype=float)
    return s.rank(method='average', pct=True).fillna(0.5)


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    zero = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return zero

    price = _get_df(history, ['adjusted_close', 'close'], tickers)
    if price is None or price.empty:
        return zero
    price = price.astype(float).iloc[-650:]
    if price.shape[0] < 80:
        return zero

    dates = price.index
    last_pos = price.shape[0] - 1
    if last_pos < 60:
        return zero

    signal_pos = last_pos
    cur = dates[last_pos]
    try:
        if isinstance(cur, pd.Timestamp):
            mask = (dates.year < cur.year) | ((dates.year == cur.year) & (dates.month < cur.month))
            pos = np.where(mask)[0]
            if len(pos) > 0:
                signal_pos = int(pos[-1])
    except Exception:
        signal_pos = last_pos

    if signal_pos < 60:
        return zero
    p = price.iloc[:signal_pos + 1]
    if p.shape[0] < 61:
        return zero

    last_close = price.iloc[-1]
    valid = last_close.notna() & np.isfinite(last_close) & (last_close > 2.0)
    cnt = p.notna().sum()
    valid = valid & (cnt >= min(120, max(50, p.shape[0] // 2)))

    elig = _get_df(history, ['eligible'], tickers)
    if elig is not None and len(elig.index) > 0:
        try:
            e = elig.reindex(index=price.index).iloc[-1]
            valid = valid & e.fillna(False).astype(bool)
        except Exception:
            pass

    vol_df = _get_df(history, ['volume'], tickers)
    raw = _get_df(history, ['close'], tickers)
    if vol_df is not None:
        try:
            v = vol_df.reindex(index=price.index, columns=tickers).iloc[-20:].astype(float)
            c = (raw if raw is not None else price).reindex(index=price.index, columns=tickers).iloc[-20:].astype(float)
            adv = (v * c).median()
            valid = valid & adv.notna() & np.isfinite(adv) & (adv > 300000.0)
        except Exception:
            pass

    rets = p.pct_change(fill_method=None)
    vol60 = rets.tail(60).std(ddof=0)
    vol60 = pd.to_numeric(vol60, errors='coerce').replace([np.inf, -np.inf], np.nan).clip(lower=0.005)
    ann_vol = vol60 * math.sqrt(252.0)
    valid = valid & vol60.notna() & np.isfinite(vol60) & (vol60 > 0.0)

    mkt = rets.mean(axis=1)
    win = min(120, max(30, rets.shape[0] - 1))
    R = rets.tail(win).to_numpy(dtype=float)
    M = mkt.tail(win).to_numpy(dtype=float)
    if R.size == 0:
        beta = pd.Series(1.0, index=tickers, dtype=float)
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
        beta = pd.Series(beta_vals, index=tickers, dtype=float)

    def lag_ret(lag):
        if p.shape[0] > lag:
            return p.iloc[-1] / p.iloc[-1 - lag] - 1.0
        return pd.Series(np.nan, index=tickers, dtype=float)

    def between_ret(recent, old):
        if p.shape[0] > old:
            return p.iloc[-1 - recent] / p.iloc[-1 - old] - 1.0
        return pd.Series(np.nan, index=tickers, dtype=float)

    r21 = lag_ret(21)
    r252 = lag_ret(252)
    mom12 = between_ret(21, 252) if p.shape[0] > 252 else pd.Series(np.nan, index=tickers, dtype=float)
    mom6 = between_ret(21, 126) if p.shape[0] > 126 else pd.Series(np.nan, index=tickers, dtype=float)
    mom12 = mom12.where(mom12.notna(), mom6)
    mom12 = mom12.where(mom12.notna(), r252)
    mom6 = mom6.where(mom6.notna(), r21)

    mkt_clean = mkt.dropna()

    def mkt_cum(recent, old):
        if len(mkt_clean) > old:
            arr = mkt_clean.iloc[-old:-recent].to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                return float(np.expm1(np.nansum(np.log1p(arr))))
        return np.nan

    mc12 = mkt_cum(21, 252)
    mc6 = mkt_cum(21, 126)
    resid12 = mom12 - beta * mc12 if np.isfinite(mc12) else mom12.copy()
    resid6 = mom6 - beta * mc6 if np.isfinite(mc6) else mom6.copy()
    resid12 = resid12.where(resid12.notna(), mom12)
    resid6 = resid6.where(resid6.notna(), mom6)

    stress = False
    mvol = 0.20
    if len(mkt_clean) >= 80:
        mvol = float(mkt_clean.tail(60).std(ddof=0) * math.sqrt(252.0))
        if not np.isfinite(mvol) or mvol <= 0:
            mvol = 0.20
        lvl = (1.0 + mkt_clean).cumprod()
        if len(lvl) >= 200:
            trend_up = bool(lvl.iloc[-1] > lvl.tail(200).mean())
        elif len(lvl) >= 100:
            trend_up = bool(lvl.iloc[-1] > lvl.tail(100).mean())
        else:
            trend_up = True
        roll = mkt_clean.rolling(60).std(ddof=0) * math.sqrt(252.0)
        rv = roll.dropna()
        if len(rv) >= 120:
            q = float(rv.tail(252).quantile(0.65))
        elif len(rv) >= 30:
            q = float(rv.quantile(0.65))
        else:
            q = 0.22
        high_vol = bool(mvol > max(0.18, q))
        stress = high_vol or (not trend_up and mvol > 0.16)

    rev_adj = -r21 / (vol60 * math.sqrt(21.0))
    lowvol = -ann_vol
    max21 = rets.tail(21).max()
    longrev = -r252

    if stress:
        score = (0.60 * _rank_pct(resid12 / ann_vol)
                 + 0.25 * _rank_pct(resid6 / ann_vol)
                 + 0.15 * _rank_pct(lowvol))
        valid = valid & (resid12.notna() | resid6.notna())
    else:
        score = (0.50 * _rank_pct(rev_adj)
                 + 0.20 * _rank_pct(lowvol)
                 + 0.15 * _rank_pct(longrev)
                 + 0.15 * _rank_pct(-max21))
        valid = valid & r21.notna()

    score = pd.to_numeric(score, errors='coerce').replace([np.inf, -np.inf], np.nan)
    valid = valid & score.notna() & np.isfinite(score)
    if valid.sum() < 8:
        alt = resid12 / ann_vol if stress else rev_adj
        alt = alt.replace([np.inf, -np.inf], np.nan)
        if alt.notna().sum() < 8:
            return zero
        score = _rank_pct(alt)
        valid = alt.notna() & np.isfinite(score)
        if valid.sum() < 8:
            return zero

    s = score[valid].replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 10:
        return zero
    n_side = min(30, len(s) // 2)
    if n_side < 5:
        return zero

    ordered = s.sort_values(ascending=False)
    longs = ordered.index[:n_side]
    shorts = ordered.index[-n_side:]

    gross_scale = 1.0
    if mvol > 0.30:
        gross_scale = 0.85
    elif mvol > 0.25:
        gross_scale = 0.95
    w = min(0.02, 0.5 * gross_scale / float(n_side))

    weights = zero.copy()
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
    gross = float(weights.abs().sum())
    if gross > 1.0:
        weights = weights * (1.0 / gross)

    net = float(weights.sum())
    if abs(net) > 1e-12:
        nz = weights.index[weights != 0.0]
        if len(nz) > 0:
            weights.loc[nz] = weights.loc[nz] - net / float(len(nz))
            weights = weights.clip(-0.02, 0.02)
            gross = float(weights.abs().sum())
            if gross > 1.0:
                weights = weights * (1.0 / gross)

    return weights.fillna(0.0).astype(float)