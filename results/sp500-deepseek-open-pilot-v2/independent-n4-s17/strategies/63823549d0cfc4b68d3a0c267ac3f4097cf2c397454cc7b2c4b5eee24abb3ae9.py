import numpy as np
import pandas as pd

def generate_weights(history, universe):
    tickers = pd.Index(universe)
    zeros = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return zeros
    ac = history.get('adjusted_close', None)
    if ac is None or not isinstance(ac, pd.DataFrame) or len(ac.index) == 0:
        return zeros
    dates = ac.index
    last_date = dates[-1]
    wd = int(last_date.weekday())
    if wd == 4:
        target = last_date
    elif wd > 4:
        target = last_date - pd.Timedelta(days=wd - 4)
    else:
        target = last_date - pd.Timedelta(days=wd + 3)
    pos = int(dates.searchsorted(target, side='right')) - 1
    if pos < 0:
        return zeros
    cutoff = dates[pos]
    px_all = ac.loc[:cutoff]
    if len(px_all.index) < 27:
        return zeros
    window = px_all.tail(32)
    if len(window.index) < 27:
        return zeros
    rets = window.pct_change()
    rets = rets.replace([np.inf, -np.inf], np.nan)
    vol = rets.tail(21).std(axis=0, ddof=1)
    vol_count = rets.tail(21).notna().sum(axis=0)
    vol = vol.where(vol_count >= 12)
    p_end = window.iloc[-1]
    p_start = window.iloc[-6]
    ret5 = p_end / p_start - 1.0
    ret5 = ret5.replace([np.inf, -np.inf], np.nan)
    mean_ret = ret5.mean(skipna=True)
    if not np.isfinite(mean_ret):
        mean_ret = 0.0
    score = (ret5 - mean_ret) / (vol * np.sqrt(5.0))
    score = score.replace([np.inf, -np.inf], np.nan)
    score_u = score.reindex(tickers)
    vol_u = vol.reindex(tickers)
    p_end_u = p_end.reindex(tickers)
    p_start_u = p_start.reindex(tickers)
    valid = score_u.notna() & np.isfinite(score_u) & vol_u.notna() & (vol_u > 0.0)
    valid = valid & p_end_u.notna() & p_start_u.notna() & (p_end_u > 0.0) & (p_start_u > 0.0)
    elig = history.get('eligible', None)
    if isinstance(elig, pd.DataFrame) and len(elig.index) > 0:
        try:
            e_sub = elig.loc[:cutoff]
            if len(e_sub.index) > 0:
                e_last = e_sub.iloc[-1].reindex(tickers)
                valid = valid & e_last.fillna(False).astype(bool)
        except Exception:
            pass
    close = history.get('close', None)
    volume = history.get('volume', None)
    if isinstance(close, pd.DataFrame) and isinstance(volume, pd.DataFrame):
        try:
            c_sub = close.loc[:cutoff].tail(20)
            v_sub = volume.loc[:cutoff].tail(20)
            dollar = c_sub * v_sub
            cnt = dollar.notna().sum(axis=0)
            adv = dollar.mean(axis=0)
            adv = adv.where(cnt >= 10)
            adv_u = adv.reindex(tickers)
            valid = valid & adv_u.notna() & (adv_u > 0.0)
            if valid.sum() > 0:
                adv_valid = adv_u[valid]
                if len(adv_valid) > 10:
                    q = adv_valid.quantile(0.20)
                    if np.isfinite(q):
                        valid = valid & (adv_u >= q)
        except Exception:
            pass
    if valid.sum() > 0:
        vol_valid = vol_u[valid]
        if len(vol_valid) > 10:
            qv = vol_valid.quantile(0.90)
            if np.isfinite(qv) and qv > 0.0:
                valid = valid & (vol_u <= qv)
    valid_scores = score_u[valid].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(valid_scores)
    if n < 4:
        return zeros
    n_side = min(50, n // 2)
    if n_side < 1:
        return zeros
    ordered = valid_scores.sort_values(kind='mergesort')
    idx = ordered.index
    longs = idx[:n_side]
    shorts = idx[-n_side:]
    per = min(0.02, 0.5 / float(n_side))
    w = zeros.copy()
    if len(longs) > 0:
        w.loc[longs] = per
    if len(shorts) > 0:
        w.loc[shorts] = -per
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
    w = w.clip(lower=-0.02, upper=0.02)
    return w