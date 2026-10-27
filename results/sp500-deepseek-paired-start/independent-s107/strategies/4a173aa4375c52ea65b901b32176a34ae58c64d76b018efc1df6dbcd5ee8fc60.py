import numpy as np
import pandas as pd

_STATE = {'weights': None, 'month_key': None}

def _rank_score(s):
    s = s.astype(float)
    r = s.rank(pct=True, na_option='keep')
    return r - 0.5

def generate_weights(history, universe):
    tickers = pd.Index(universe)
    zero = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return zero
    close = history.get('adjusted_close')
    if not isinstance(close, pd.DataFrame):
        close = history.get('close')
    if not isinstance(close, pd.DataFrame) or close.shape[0] == 0:
        return zero
    cols = [t for t in tickers if t in close.columns]
    if len(cols) == 0:
        return zero
    latest_date = close.index[-1]
    month_key = (latest_date.year, latest_date.month)
    if _STATE.get('month_key') == month_key and _STATE.get('weights') is not None:
        w = _STATE['weights'].reindex(tickers).fillna(0.0).astype(float)
        w = w.clip(-0.02, 0.02)
        net = float(w.sum())
        if abs(net) > 0.008:
            nz = w != 0
            if nz.any():
                w.loc[nz] = w.loc[nz] - net / float(nz.sum())
                w = w.clip(-0.02, 0.02)
        return w.fillna(0.0)
    px = close.loc[:, cols].astype(float)
    latest_px = px.iloc[-1]
    valid = latest_px.notna() & (latest_px > 0)
    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame):
        try:
            elig_latest = elig.reindex(index=px.index, columns=cols).iloc[-1]
            valid = valid & elig_latest.fillna(False).astype(bool)
        except Exception:
            pass
    rets = px / px.shift(1) - 1.0
    rets = rets.replace([np.inf, -np.inf], np.nan)
    n = len(px)
    mom = pd.Series(np.nan, index=cols, dtype=float)
    if n >= 252:
        mom = px.iloc[-22] / px.iloc[-252] - 1.0
    elif n >= 22:
        mom = px.iloc[-22] / px.iloc[0] - 1.0
    mom = mom.replace([np.inf, -np.inf], np.nan)
    st = pd.Series(np.nan, index=cols, dtype=float)
    if n >= 6:
        st = px.iloc[-1] / px.iloc[-6] - 1.0
    st = st.replace([np.inf, -np.inf], np.nan)
    vol = rets.tail(126).std(skipna=True)
    vol = vol.replace([np.inf, -np.inf], np.nan)
    dv = None
    vol_df = history.get('volume')
    if isinstance(vol_df, pd.DataFrame):
        vol_c = vol_df.reindex(index=px.index, columns=cols)
        raw_close = history.get('close')
        if isinstance(raw_close, pd.DataFrame):
            px_raw = raw_close.reindex(index=px.index, columns=cols)
            px_raw = px_raw.where(px_raw.notna(), px)
        else:
            px_raw = px
        dv = (px_raw * vol_c).tail(60).median(skipna=True)
    if dv is not None:
        dv_valid = dv.reindex(cols)
        dv_rank = dv_valid.rank(pct=True, na_option='keep')
        liq_ok = dv_valid.notna() & (dv_rank >= 0.20)
        valid = valid & liq_ok.reindex(cols).fillna(False)
    score = pd.Series(0.0, index=cols, dtype=float)
    denom = 0.0
    if mom.notna().sum() >= 10:
        score = score + 1.0 * _rank_score(mom)
        denom += 1.0
    if st.notna().sum() >= 10:
        score = score - 0.35 * _rank_score(st)
        denom += 0.35
    if vol.notna().sum() >= 10:
        score = score - 0.25 * _rank_score(vol)
        denom += 0.25
    if denom > 0.0:
        score = score / denom
    factor_count = mom.notna().astype(int) + st.notna().astype(int) + vol.notna().astype(int)
    valid = valid & (factor_count > 0)
    cand_mask = valid.reindex(cols).fillna(False) & score.notna()
    cand = list(score.index[cand_mask])
    if len(cand) < 10:
        cand_mask2 = (latest_px > 0) & latest_px.notna() & score.notna() & (factor_count > 0)
        cand = list(score.index[cand_mask2])
    if len(cand) < 4:
        return zero
    s = score.reindex(cand).sort_values(ascending=False, kind='mergesort')
    n_side = min(25, len(s) // 2)
    if n_side < 1:
        return zero
    longs = list(s.index[:n_side])
    shorts = list(s.index[-n_side:])
    w_per = min(0.02, 0.9 / (2.0 * n_side))
    w = pd.Series(0.0, index=tickers, dtype=float)
    for t in longs:
        if t in w.index:
            w.loc[t] = w_per
    for t in shorts:
        if t in w.index:
            w.loc[t] = -w_per
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    w = w.clip(-0.02, 0.02)
    net = float(w.sum())
    if abs(net) > 0.008:
        nz = w != 0
        if nz.any():
            w.loc[nz] = w.loc[nz] - net / float(nz.sum())
            w = w.clip(-0.02, 0.02)
    _STATE['weights'] = w.copy()
    _STATE['month_key'] = month_key
    return w