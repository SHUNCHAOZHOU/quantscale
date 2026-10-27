import numpy as np
import pandas as pd

def _rank(s):
    return s.rank(method='average', pct=True).fillna(0.5)

def generate_weights(history, universe):
    zero = pd.Series(0.0, index=universe, dtype=float)
    try:
        close = history.get('adjusted_close')
        if close is None:
            close = history.get('close')
        volume = history.get('volume')
        raw_close = history.get('close')
        eligible = history.get('eligible')
    except Exception:
        return zero
    if close is None or not hasattr(close, 'columns') or close.shape[0] < 30:
        return zero
    tickers = [t for t in universe if t in close.columns]
    if len(tickers) < 4:
        return zero
    idx = close.index
    last_date = idx[-1]
    try:
        prev_month_end = last_date.replace(day=1) - pd.Timedelta(days=1)
        pos = idx.searchsorted(prev_month_end, side='right') - 1
        if pos >= 0:
            signal_date = idx[pos]
        else:
            signal_date = last_date
    except Exception:
        signal_date = last_date
    p = close.loc[:signal_date, tickers]
    if p.shape[0] < 30:
        return zero
    if volume is not None:
        v = volume.loc[:signal_date, tickers].reindex(index=p.index, columns=tickers)
    else:
        v = pd.DataFrame(np.nan, index=p.index, columns=tickers)
    if raw_close is not None:
        c = raw_close.loc[:signal_date, tickers].reindex(index=p.index, columns=tickers)
    else:
        c = p
    if eligible is not None:
        el = eligible.loc[:signal_date, tickers].reindex(index=p.index, columns=tickers)
        last_el = el.iloc[-1].fillna(False).astype(bool)
    else:
        last_el = pd.Series(True, index=tickers)
    last_p = p.iloc[-1]
    nrows = p.shape[0]
    if nrows > 252:
        mom12 = p.shift(21).iloc[-1] / p.shift(252).iloc[-1] - 1.0
    else:
        mom12 = pd.Series(np.nan, index=tickers)
    if nrows > 126:
        mom6 = p.shift(21).iloc[-1] / p.shift(126).iloc[-1] - 1.0
    else:
        mom6 = pd.Series(np.nan, index=tickers)
    mom = mom12.where(mom12.notna(), mom6)
    if nrows > 21:
        rev = -(p.iloc[-1] / p.shift(21).iloc[-1] - 1.0)
    else:
        rev = pd.Series(np.nan, index=tickers)
    rets = p.pct_change()
    if nrows >= 40:
        cnt = rets.iloc[-60:].notna().sum()
        vol = rets.iloc[-60:].std()
        vol = vol.where(cnt >= 40)
    else:
        vol = pd.Series(np.nan, index=tickers)
    negvol = -vol
    if nrows >= 5:
        dv = (c.iloc[-20:] * v.iloc[-20:]).mean()
        liq = np.log1p(dv.clip(lower=0))
    else:
        liq = pd.Series(np.nan, index=tickers)
    valid = last_p.notna() & (last_p > 0) & last_el & mom.notna() & liq.notna() & (liq > 0)
    if nrows > 0:
        valid = valid & (v.iloc[-1].fillna(0) > 0)
    valid_tickers = list(valid[valid].index)
    n_valid = len(valid_tickers)
    if n_valid < 6:
        return zero
    mom_v = mom.loc[valid_tickers]
    rev_v = rev.loc[valid_tickers]
    negvol_v = negvol.loc[valid_tickers]
    liq_v = liq.loc[valid_tickers]
    score = (0.45 * _rank(mom_v) + 0.25 * _rank(rev_v) + 0.15 * _rank(negvol_v) + 0.15 * _rank(liq_v))
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0.5)
    n_side = int(round(0.10 * n_valid))
    n_side = max(n_side, min(25, n_valid // 2))
    n_side = min(n_side, 60, n_valid // 2)
    n_side = max(n_side, 1)
    order = score.sort_values(ascending=False)
    longs = list(order.index[:n_side])
    shorts = list(order.index[-n_side:])
    w = min(0.5 / n_side, 0.02)
    weights = pd.Series(0.0, index=universe, dtype=float)
    for t in longs:
        if t in weights.index:
            weights.loc[t] = w
    for t in shorts:
        if t in weights.index:
            weights.loc[t] = -w
    return weights