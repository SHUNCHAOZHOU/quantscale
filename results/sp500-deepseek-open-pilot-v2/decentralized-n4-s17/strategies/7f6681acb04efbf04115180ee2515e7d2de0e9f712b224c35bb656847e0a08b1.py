import numpy as np
import pandas as pd


def _zscore(x, mask):
    v = np.where(mask, x, np.nan)
    m = np.isfinite(v)
    out = np.zeros(v.shape[0], dtype=float)
    if int(m.sum()) < 10:
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

    open_key = None
    for k in ('adjusted_open', 'adjusted_close', 'close'):
        if k in history:
            open_key = k
            break
    close_key = None
    for k in ('adjusted_close', 'close'):
        if k in history:
            close_key = k
            break
    if open_key is None or close_key is None:
        return empty

    try:
        op_full = history[open_key].reindex(columns=tickers)
        cl_full = history[close_key].reindex(columns=tickers)
    except Exception:
        return empty

    if op_full.shape[0] < 80:
        return empty
    op = op_full.iloc[-320:].astype(float)
    cl = cl_full.reindex(index=op.index).astype(float)
    op = op.where(op > 0.0)
    cl = cl.where(cl > 0.0)
    if op.shape[0] < 80:
        return empty

    try:
        vol_df = history['volume'].reindex(columns=tickers).reindex(index=op.index).astype(float)
    except Exception:
        vol_df = pd.DataFrame(np.nan, index=op.index, columns=tickers)

    if op.shape[0] >= 260:
        mom_series = op.shift(21) / op.shift(252) - 1.0
    elif op.shape[0] >= 140:
        mom_series = op.shift(21) / op.shift(126) - 1.0
    else:
        mom_series = op.shift(5) / op.shift(63) - 1.0
    mom = mom_series.iloc[-5:].mean()

    if op.shape[0] >= 140:
        mom6_series = op.shift(21) / op.shift(126) - 1.0
    else:
        mom6_series = op.shift(5) / op.shift(63) - 1.0
    mom6 = mom6_series.iloc[-5:].mean()

    ret = op / op.shift(1) - 1.0
    vol60 = ret.rolling(60, min_periods=40).std().iloc[-1]

    last_px = cl.iloc[-1]
    dv = (cl * vol_df).iloc[-60:].median()

    p_last = last_px.values.astype(float)
    volv = vol60.values.astype(float)
    momv = mom.values.astype(float)
    mom6v = mom6.values.astype(float)
    dvv = dv.values.astype(float)

    base = (np.isfinite(p_last) & (p_last > 1.0)
            & np.isfinite(volv) & (volv > 1e-6)
            & np.isfinite(momv))
    liquid = base & np.isfinite(dvv) & (dvv >= 1e6)
    mask = liquid if int(liquid.sum()) >= 40 else base

    if 'eligible' in history:
        try:
            el = history['eligible'].reindex(columns=tickers).reindex(index=op.index).iloc[-1]
            el_bool = np.nan_to_num(el.values.astype(float), nan=0.0) > 0.5
            m2 = mask & el_bool
            if int(m2.sum()) >= 20:
                mask = m2
        except Exception:
            pass

    if int(mask.sum()) < 4:
        return empty

    z_mom = _zscore(momv, mask)
    z_mom6 = _zscore(mom6v, mask)
    z_lowvol = _zscore(-volv, mask)
    score = 0.65 * z_mom + 0.15 * z_mom6 + 0.20 * z_lowvol
    score = np.where(mask, score, np.nan)

    idx = np.where(mask)[0]
    if idx.size < 2:
        return empty
    s_sel = score[idx]
    order = idx[np.argsort(-s_sel, kind='mergesort')]

    k = int(min(50, idx.size // 2))
    if k < 1:
        return empty

    w_side = float(min(0.02, 0.98 / (2.0 * k)))
    w = np.zeros(n, dtype=float)
    w[order[:k]] = w_side
    w[order[-k:]] = -w_side

    w = np.clip(w, -0.02, 0.02)
    w = np.where(np.isfinite(w), w, 0.0)
    return pd.Series(w, index=tickers, dtype=float)
