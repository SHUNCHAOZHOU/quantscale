import numpy as np
import pandas as pd


def _zscore(x, mask):
    v = np.where(mask, x, np.nan)
    m = np.isfinite(v)
    out = np.zeros(v.shape[0], dtype=float)
    if int(m.sum()) < 5:
        return out
    mu = np.nanmean(v)
    sd = np.nanstd(v)
    if (not np.isfinite(sd)) or sd <= 1e-12:
        return out
    z = (v - mu) / sd
    z = np.clip(z, -3.0, 3.0)
    z = np.where(m, z, 0.0)
    return z


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    empty = pd.Series(0.0, index=tickers, dtype=float)

    key = 'adjusted_close' if 'adjusted_close' in history else 'close'
    try:
        px_df = history[key].reindex(columns=tickers)
    except Exception:
        return empty
    px_df = px_df.iloc[-320:]
    if px_df.shape[0] < 30:
        return empty

    px = px_df.astype(float)
    px = px.where(px > 0.0)

    try:
        vv = history['volume'].reindex(columns=tickers).reindex(index=px.index).astype(float)
    except Exception:
        vv = pd.DataFrame(np.nan, index=px.index, columns=tickers)

    last_px = px.iloc[-1]

    ret1 = px / px.shift(1) - 1.0
    vol60 = ret1.shift(1).rolling(60, min_periods=30).std()
    vol_l = vol60.iloc[-1]

    ret5 = (px.shift(1) / px.shift(6) - 1.0).iloc[-1]
    ret21 = (px.shift(1) / px.shift(22) - 1.0).iloc[-1]
    mom = (px.shift(22) / px.shift(253) - 1.0).iloc[-1]
    dv = (px * vv).iloc[-60:].median()

    vola = vol_l.values
    safe_vol = np.where(np.isfinite(vola) & (vola > 1e-6), vola, np.nan)

    r5 = ret5.values / (safe_vol * np.sqrt(5.0))
    r21 = ret21.values / (safe_vol * np.sqrt(21.0))
    rm = mom.values / (safe_vol * np.sqrt(252.0))
    lv = safe_vol

    p_last = last_px.values.astype(float)
    dv_arr = dv.values.astype(float)

    base = (np.isfinite(p_last) & (p_last > 1.0)
            & np.isfinite(safe_vol)
            & np.isfinite(r5) & np.isfinite(r21))

    liquid = base & np.isfinite(dv_arr) & (dv_arr >= 1e6)
    mask = liquid if int(liquid.sum()) >= 30 else base

    if 'eligible' in history:
        try:
            el = history['eligible'].reindex(columns=tickers).reindex(index=px.index).iloc[-1]
            el_bool = np.nan_to_num(el.values.astype(float), nan=0.0) > 0.5
            m2 = mask & el_bool
            if int(m2.sum()) >= 20:
                mask = m2
        except Exception:
            pass

    if int(mask.sum()) < 2:
        return empty

    z5 = _zscore(-r5, mask)
    z21 = _zscore(-r21, mask)
    zm = _zscore(rm, mask)
    zlv = _zscore(-lv, mask)

    score = 0.35 * z5 + 0.20 * z21 + 0.30 * zm + 0.15 * zlv
    score = np.where(mask, score, np.nan)

    idx = np.where(mask)[0]
    s_sel = score[idx]
    order = idx[np.argsort(-s_sel, kind='mergesort')]

    k = int(min(40, idx.size // 2))
    if k < 1:
        return empty

    w_side = float(min(0.49 / k, 0.02))
    w = np.zeros(n, dtype=float)
    w[order[:k]] = w_side
    w[order[-k:]] = -w_side

    w = np.clip(w, -0.02, 0.02)
    w = np.where(np.isfinite(w), w, 0.0)
    return pd.Series(w, index=tickers, dtype=float)
