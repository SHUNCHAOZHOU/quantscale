import numpy as np
import pandas as pd


def _z(x, mask):
    out = np.zeros(x.shape[0], dtype=float)
    m = np.asarray(mask, dtype=bool) & np.isfinite(x)
    if int(m.sum()) < 10:
        return out
    vals = x[m]
    mu = float(np.nanmean(vals))
    sd = float(np.nanstd(vals))
    if (not np.isfinite(sd)) or sd < 1e-12:
        return out
    z = (x - mu) / sd
    z = np.clip(z, -3.0, 3.0)
    out[m] = z[m]
    return out


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    n = len(tickers)
    empty = pd.Series(0.0, index=tickers, dtype=float)
    if n == 0:
        return empty

    okey = 'adjusted_open' if 'adjusted_open' in history else 'open'
    ckey = 'adjusted_close' if 'adjusted_close' in history else 'close'
    if okey not in history or ckey not in history:
        return empty

    try:
        op = history[okey].reindex(columns=tickers).astype(float)
        cl = history[ckey].reindex(columns=tickers).astype(float)
    except Exception:
        return empty

    if op.shape[0] < 90:
        return empty

    op = op.iloc[-300:]
    cl = cl.reindex(index=op.index)
    op = op.where(op > 0.0)
    cl = cl.where(cl > 0.0)

    try:
        vol = history['volume'].reindex(columns=tickers).reindex(index=op.index).astype(float)
    except Exception:
        vol = pd.DataFrame(np.nan, index=op.index, columns=tickers)

    ret = op / op.shift(1) - 1.0
    ret = ret.where(np.abs(ret) < 0.7)
    vol90 = ret.rolling(90, min_periods=50).std().iloc[-1]

    r21 = (op.shift(3) / op.shift(24) - 1.0).iloc[-3:].mean()
    r63 = (op.shift(5) / op.shift(68) - 1.0).iloc[-3:].mean()

    if op.shape[0] >= 252:
        mom = (op.shift(21) / op.shift(252) - 1.0).iloc[-5:].mean()
    else:
        mom = pd.Series(np.nan, index=tickers, dtype=float)

    last_px = cl.iloc[-1]
    dv = (cl * vol).rolling(60, min_periods=40).median().iloc[-1]

    p = last_px.values.astype(float)
    v = vol90.values.astype(float)
    r2 = r21.values.astype(float)
    r6 = r63.values.astype(float)
    mm = mom.reindex(tickers).values.astype(float)
    d = dv.values.astype(float)

    mask = (np.isfinite(p) & (p > 1.0)
            & np.isfinite(v) & (v > 1e-5)
            & np.isfinite(r2) & np.isfinite(r6))

    liq = mask & np.isfinite(d) & (d >= 1e6)
    if int(liq.sum()) >= 30:
        mask = liq

    if 'eligible' in history:
        try:
            el = history['eligible'].reindex(columns=tickers).reindex(index=op.index).iloc[-1]
            el = np.nan_to_num(el.values.astype(float), nan=0.0) > 0.5
            m2 = mask & el
            if int(m2.sum()) >= 20:
                mask = m2
        except Exception:
            pass

    if int(mask.sum()) < 8:
        return empty

    z_r2 = _z(-r2, mask)
    z_r6 = _z(-r6, mask)
    z_lv = _z(-v, mask)
    z_mom = _z(mm, mask & np.isfinite(mm))

    score = 0.55 * z_r2 + 0.25 * z_r6 + 0.20 * z_lv + 0.05 * z_mom
    score = np.where(mask, score, -np.inf)

    idx = np.flatnonzero(mask)
    if idx.size < 4:
        return empty

    k = int(min(50, max(10, idx.size // 3)))
    if k < 2:
        return empty

    order = idx[np.argsort(-score[idx], kind='mergesort')]
    wside = float(min(0.0195, 0.48 / k))

    w = np.zeros(n, dtype=float)
    w[order[:k]] = wside
    w[order[-k:]] = -wside
    w = np.where(np.isfinite(w), w, 0.0)

    return pd.Series(w, index=tickers, dtype=float)
