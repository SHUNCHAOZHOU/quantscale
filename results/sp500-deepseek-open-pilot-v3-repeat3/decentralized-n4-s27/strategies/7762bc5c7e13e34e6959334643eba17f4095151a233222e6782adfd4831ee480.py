import numpy as np
import pandas as pd


def generate_weights(history, universe):
    tickers = pd.Index(universe)
    zero = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return zero

    ac_all = history.get('adjusted_close')
    if ac_all is None or not isinstance(ac_all, pd.DataFrame) or ac_all.shape[0] == 0:
        ac_all = history.get('close')
    if ac_all is None or not isinstance(ac_all, pd.DataFrame) or ac_all.shape[0] == 0:
        return zero

    ac_all = ac_all.reindex(columns=tickers)
    dates = ac_all.index
    nrows = len(dates)
    if nrows < 30:
        return zero

    wd = np.asarray(dates.weekday)
    fri = np.where(wd == 4)[0]
    ref_idx = int(fri[-1]) if len(fri) > 0 else nrows - 1
    if ref_idx < 21:
        return zero

    start = max(0, ref_idx - 299)
    c = ac_all.iloc[start:ref_idx + 1].copy()
    if c.shape[0] < 22:
        return zero
    n = c.shape[0]
    last = pd.to_numeric(c.iloc[-1], errors='coerce').reindex(tickers)

    def _ret(k):
        if n >= k + 1:
            base = pd.to_numeric(c.iloc[-k - 1], errors='coerce').reindex(tickers)
            r = last / base - 1.0
        else:
            r = pd.Series(np.nan, index=tickers, dtype=float)
        return r.replace([np.inf, -np.inf], np.nan)

    rev5 = _ret(5)
    rev10 = _ret(10)
    rev21 = _ret(21)

    if n >= 252:
        mom12 = (pd.to_numeric(c.iloc[-22], errors='coerce').reindex(tickers)
                 / pd.to_numeric(c.iloc[-252], errors='coerce').reindex(tickers) - 1.0)
    else:
        mom12 = pd.Series(np.nan, index=tickers, dtype=float)
    if n >= 126:
        mom6 = (pd.to_numeric(c.iloc[-22], errors='coerce').reindex(tickers)
                / pd.to_numeric(c.iloc[-126], errors='coerce').reindex(tickers) - 1.0)
    else:
        mom6 = pd.Series(np.nan, index=tickers, dtype=float)
    mom12 = mom12.replace([np.inf, -np.inf], np.nan)
    mom6 = mom6.replace([np.inf, -np.inf], np.nan)

    look = min(252, n)
    max52 = c.iloc[-look:].max()
    high52 = last / max52 - 1.0
    high52 = high52.replace([np.inf, -np.inf], np.nan)

    rets = c.pct_change()
    if n >= 60:
        vol = rets.iloc[-60:].std()
    else:
        vol = rets.std()
    vol = vol.reindex(tickers).replace([np.inf, -np.inf], np.nan)

    close_all = history.get('close')
    if close_all is not None and isinstance(close_all, pd.DataFrame):
        close_ref = close_all.reindex(columns=tickers).reindex(index=c.index)
    else:
        close_ref = c
    px_for_dv = close_ref.where(close_ref.notna(), c)

    vol_all = history.get('volume')
    if vol_all is not None and isinstance(vol_all, pd.DataFrame):
        vol_ref = vol_all.reindex(columns=tickers).reindex(index=c.index)
        dv = (px_for_dv * vol_ref).iloc[-min(60, n):].median()
    else:
        dv = pd.Series(1.0, index=tickers, dtype=float)
    liq = dv.reindex(tickers).replace([np.inf, -np.inf], np.nan)

    el = history.get('eligible')
    if el is not None and isinstance(el, pd.DataFrame) and len(el) > 0:
        el = el.reindex(columns=tickers)
        ref_date = dates[ref_idx]
        if ref_date in el.index:
            e = el.loc[ref_date]
        else:
            sub = el.loc[:ref_date]
            e = sub.iloc[-1] if len(sub) > 0 else pd.Series(True, index=tickers)
    else:
        e = pd.Series(True, index=tickers)
    e = e.reindex(tickers).fillna(False).astype(bool)

    last_price = close_ref.iloc[-1].reindex(tickers)
    last_price = last_price.where(last_price.notna(), last)

    valid = e & last.notna() & (last > 0) & last_price.notna() & (last_price > 2.0)
    valid = valid & np.isfinite(mom6) & np.isfinite(rev5) & np.isfinite(rev21)
    valid = valid & np.isfinite(vol) & (vol > 1e-8) & np.isfinite(liq) & (liq > 0)
    valid = valid.fillna(False).astype(bool)
    if int(valid.sum()) < 6:
        return zero

    idx = valid.index[valid]
    sig = pd.DataFrame(index=idx)
    sig['mom12'] = mom12.reindex(idx)
    sig['mom6'] = mom6.reindex(idx)
    sig['high52'] = high52.reindex(idx)
    sig['rev5'] = rev5.reindex(idx)
    sig['rev10'] = rev10.reindex(idx)
    sig['rev21'] = rev21.reindex(idx)
    sig['vol'] = vol.reindex(idx)
    sig['liq'] = liq.reindex(idx)

    def _rk(s):
        s = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)
        return s.rank(pct=True, method='average') - 0.5

    r_mom12 = _rk(sig['mom12'] / sig['vol'])
    r_mom6 = _rk(sig['mom6'] / sig['vol'])
    r_high52 = _rk(sig['high52'])
    r_rev5 = _rk(sig['rev5'])
    r_rev10 = _rk(sig['rev10'])
    r_rev21 = _rk(sig['rev21'])
    r_vol = _rk(sig['vol'])
    r_liq = _rk(sig['liq'])

    score = (0.35 * r_mom12.fillna(0.0)
             + 0.25 * r_mom6.fillna(0.0)
             + 0.15 * r_high52.fillna(0.0)
             - 0.18 * r_rev5.fillna(0.0)
             - 0.12 * r_rev10.fillna(0.0)
             - 0.10 * r_rev21.fillna(0.0)
             - 0.10 * r_vol.fillna(0.0)
             + 0.05 * r_liq.fillna(0.0))
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 2:
        return zero

    n_side = min(25, len(score) // 2)
    if n_side < 1:
        return zero

    order = score.sort_values(ascending=False)
    longs = order.index[:n_side]
    shorts = order.index[-n_side:]

    w = min(0.02, 0.5 / float(n_side))
    if 2.0 * n_side * w > 1.0:
        w = 0.5 / float(n_side) * (1.0 - 1e-12)

    weights = pd.Series(0.0, index=tickers, dtype=float)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights