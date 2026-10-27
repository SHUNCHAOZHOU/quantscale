import numpy as np
import pandas as pd

def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(np.zeros(n, dtype=float), index=tickers)
    ac = history.get('adjusted_close')
    if ac is None or len(ac.index) < 5:
        return zero
    try:
        ac = ac.reindex(columns=tickers)
    except Exception:
        return zero
    dates = ac.index
    last_pos = len(dates) - 1
    signal_pos = last_pos
    if isinstance(dates, pd.DatetimeIndex) and last_pos >= 0:
        wd = dates.weekday.values
        mons = np.where(wd[:last_pos + 1] == 0)[0]
        if len(mons) > 0:
            signal_pos = mons[-1]
    L = 21
    if signal_pos < L + 1:
        return zero
    close_all = ac.values
    cur_close = close_all[last_pos]
    valid = np.isfinite(cur_close) & (cur_close > 1.0)
    elig = history.get('eligible')
    if elig is not None:
        try:
            e = elig.reindex(index=dates, columns=tickers)
            le = e.iloc[last_pos].values
            if le.dtype == bool:
                valid_e = le
            else:
                valid_e = np.array([(x is True) or (x == 1) for x in le], dtype=bool)
            if np.any(pd.notna(le)):
                valid &= valid_e
        except Exception:
            pass
    close_sig = close_all[:signal_pos + 1]
    ret = close_sig[-1] / close_sig[-1 - L] - 1.0
    valid &= np.isfinite(ret)
    vol = history.get('volume')
    if vol is not None:
        try:
            v = vol.reindex(index=dates, columns=tickers).values[:signal_pos + 1]
            if v.shape[0] >= 20:
                dv = np.nanmean(close_sig[-20:] * v[-20:], axis=0)
                valid &= np.isfinite(dv) & (dv > 0)
                if valid.sum() >= 20:
                    thresh = np.nanpercentile(dv[valid], 20)
                    valid &= dv >= thresh
        except Exception:
            pass
    idx = np.where(valid)[0]
    if len(idx) < 6:
        return zero
    raw_score = -ret[idx]
    scores = raw_score.copy()
    try:
        if close_sig.shape[0] >= 22:
            daily = close_sig[-21:] / close_sig[-22:-1] - 1.0
            vol20 = np.nanstd(daily, axis=0)
            vol20 = np.where(np.isfinite(vol20) & (vol20 > 1e-8), vol20, np.nan)
            adj = raw_score / vol20[idx]
            ok = np.isfinite(adj)
            scores = np.where(ok, adj, raw_score)
    except Exception:
        scores = raw_score
    order = np.argsort(scores)
    m = min(25, len(idx) // 2)
    if m < 1:
        return zero
    long_sel = idx[order[-m:]]
    short_sel = idx[order[:m]]
    wt = min(0.02, 0.5 / m)
    w = np.zeros(n, dtype=float)
    w[long_sel] = wt
    w[short_sel] = -wt
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, -0.02, 0.02)
    return pd.Series(w, index=tickers)