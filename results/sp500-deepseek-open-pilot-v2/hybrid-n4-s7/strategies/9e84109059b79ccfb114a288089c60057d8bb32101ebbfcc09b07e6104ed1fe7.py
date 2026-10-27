import numpy as np
import pandas as pd


def _panel(history, key, universe, index=None):
    df = history.get(key)
    if not isinstance(df, pd.DataFrame) or df.shape[0] == 0:
        return None
    try:
        if index is not None:
            df = df.reindex(index=index, columns=universe)
        else:
            df = df.reindex(columns=universe)
        return df
    except Exception:
        return None


def generate_weights(history, universe):
    u = list(universe)
    out = pd.Series(0.0, index=u, dtype=float)
    if len(u) == 0 or not isinstance(history, dict):
        return out

    px = _panel(history, 'adjusted_close', u)
    if px is None or px.shape[0] == 0:
        px = _panel(history, 'close', u)
    if px is None or px.shape[0] < 30:
        return out

    n, m = px.shape
    P = np.array(px.to_numpy(dtype=float), copy=True)
    P[~np.isfinite(P)] = np.nan

    R = np.full_like(P, np.nan)
    R[1:, :] = P[1:, :] / P[:-1, :] - 1.0
    R[~np.isfinite(R)] = np.nan

    raw_close = _panel(history, 'close', u, px.index)
    vol = _panel(history, 'volume', u, px.index)
    C = None
    V = None
    if raw_close is not None:
        C = np.array(raw_close.to_numpy(dtype=float), copy=True)
        C[~np.isfinite(C)] = np.nan
    if vol is not None:
        V = np.array(vol.to_numpy(dtype=float), copy=True)
        V[~np.isfinite(V)] = np.nan

    e = _panel(history, 'eligible', u, px.index)
    elig_last = None
    if e is not None and e.shape[0] > 0:
        try:
            elig_last = e.iloc[-1].fillna(False).astype(bool).to_numpy()
        except Exception:
            elig_last = None

    K = 5
    volwin = 20
    smooth_days = 3
    if n < K + volwin + smooth_days + 1:
        return out

    scores_list = []
    for d in range(smooth_days):
        idx = n - 1 - d
        if idx < K or idx - volwin < 0:
            continue
        rK = P[idx, :] / P[idx - K, :] - 1.0
        r_slice = R[idx - volwin:idx, :]
        cnt = np.sum(np.isfinite(r_slice), axis=0).astype(float)
        sum_r = np.nansum(r_slice, axis=0)
        sum_r2 = np.nansum(r_slice * r_slice, axis=0)
        mean_r = np.divide(sum_r, cnt, out=np.full(m, np.nan), where=cnt > 0)
        var = np.divide(sum_r2 - cnt * mean_r * mean_r, cnt - 1.0,
                        out=np.full(m, np.nan), where=cnt > 1)
        var = np.maximum(var, 0.0)
        vol_stock = np.sqrt(var)
        vol_stock[cnt < 10] = np.nan
        vol_stock = np.where(np.isfinite(vol_stock), np.maximum(vol_stock, 0.002), np.nan)

        valid = np.isfinite(rK) & np.isfinite(vol_stock) & np.isfinite(P[idx, :]) & (P[idx, :] > 0)
        if np.sum(valid) < 5:
            continue
        med = np.nanmedian(rK[valid])
        norm = np.full(m, np.nan)
        norm[valid] = (rK[valid] - med) / vol_stock[valid]
        qlo, qhi = np.nanpercentile(norm[valid], [1.0, 99.0])
        if np.isfinite(qlo) and np.isfinite(qhi) and qhi > qlo:
            norm = np.clip(norm, qlo, qhi)
        scores_list.append(-norm)

    if not scores_list:
        return out
    S = np.vstack(scores_list)
    cnt_s = np.sum(np.isfinite(S), axis=0)
    score = np.divide(np.nansum(S, axis=0), cnt_s, out=np.full(m, np.nan), where=cnt_s > 0)

    last = n - 1
    current_valid = np.isfinite(score) & np.isfinite(P[last, :]) & (P[last, :] > 0)
    if elig_last is not None and len(elig_last) == m:
        current_valid &= elig_last

    price = C[last, :] if C is not None else P[last, :]
    current_valid &= np.isfinite(price) & (price > 2.0)

    if C is not None and V is not None and n >= 20:
        dv = V[-20:, :] * C[-20:, :]
        med_dv = np.nanmedian(dv, axis=0)
        current_valid &= np.isfinite(med_dv) & (med_dv > 1e6)

    if V is not None and n >= 25:
        recent = np.nanmean(V[-5:, :], axis=0)
        base = np.nanmean(V[-25:-5, :], axis=0)
        ratio = recent / np.maximum(base, 1.0)
        current_valid &= ~(np.isfinite(ratio) & (base > 0) & (ratio > 8.0))

    valid_idx = np.flatnonzero(current_valid)
    if valid_idx.size < 4:
        return out

    s = score[valid_idx]
    order = np.argsort(s, kind='stable')
    n_side = min(40, valid_idx.size // 2)
    if n_side < 2:
        return out

    short_sel = valid_idx[order[:n_side]]
    long_sel = valid_idx[order[-n_side:]]
    w = min(0.02, 0.5 / float(n_side))
    out.iloc[long_sel] = w
    out.iloc[short_sel] = -w

    vals = out.to_numpy(dtype=float)
    vals[~np.isfinite(vals)] = 0.0
    return pd.Series(vals, index=u, dtype=float)