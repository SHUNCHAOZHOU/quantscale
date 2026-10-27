import numpy as np
import pandas as pd

def generate_weights(history, universe):
    universe = pd.Index(universe)
    n = len(universe)
    if n == 0:
        return pd.Series(dtype=float)

    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None:
        return pd.Series(0.0, index=universe)

    ac = ac.reindex(columns=universe)
    vals = ac.to_numpy(dtype=float, copy=False)
    if vals.ndim != 2 or vals.shape[0] < 5:
        return pd.Series(0.0, index=universe)

    lookback = 100
    start = max(0, vals.shape[0] - lookback)
    sub = vals[start:, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        rets = sub[1:, :] / sub[:-1, :] - 1.0
    rets[~np.isfinite(rets)] = np.nan
    rets = np.clip(rets, -0.5, 0.5)
    T = rets.shape[0]
    if T < 5:
        return pd.Series(0.0, index=universe)

    mkt = np.nanmean(rets, axis=1)
    mkt = np.where(np.isfinite(mkt), mkt, 0.0)

    win_beta = min(60, T)
    rb = rets[-win_beta:, :]
    mb = mkt[-win_beta:]
    mb_mean = np.nanmean(mb)
    if not np.isfinite(mb_mean):
        mb_mean = 0.0
    mb_dev = mb - mb_mean
    den = np.nanmean(mb_dev ** 2)
    if not np.isfinite(den) or den <= 0:
        beta = np.ones(n)
    else:
        rmean = np.nanmean(rb, axis=0)
        num = np.nanmean((rb - rmean) * mb_dev[:, None], axis=0)
        beta = num / den
        beta = np.where(np.isfinite(beta), beta, 1.0)
        beta = np.clip(beta, 0.2, 3.0)

    win_vol = min(20, T)
    rv = rets[-win_vol:, :]
    mv = mkt[-win_vol:]
    resid = rv - beta[None, :] * mv[:, None]
    idio = np.nanstd(resid, axis=0, ddof=1)
    med_idio = np.nanmedian(idio)
    if not np.isfinite(med_idio) or med_idio <= 0:
        med_idio = 0.02
    idio = np.where(np.isfinite(idio) & (idio > 0), idio, med_idio)
    idio = np.clip(idio, 0.005, 0.20)

    F = min(10, T)
    rf = rets[-F:, :]
    mf = mkt[-F:]
    resid_cum = np.nansum(rf - beta[None, :] * mf[:, None], axis=0)
    valid_count = np.sum(np.isfinite(rf), axis=0)
    score = -resid_cum / idio
    last_price = sub[-1, :]
    valid = (valid_count >= max(3, F - 2)) & np.isfinite(score) & np.isfinite(last_price) & (last_price > 0)

    elig = history.get('eligible')
    if elig is not None:
        e = elig.reindex(columns=universe).iloc[-1].to_numpy(dtype=float)
        valid &= np.isfinite(e) & (e > 0.5)

    n_valid = int(valid.sum())
    if n_valid < 10:
        return pd.Series(0.0, index=universe)

    idx = np.where(valid)[0]
    order = idx[np.argsort(score[idx])]
    n_each = min(50, max(10, n_valid // 5))
    n_each = min(n_each, n_valid // 2)
    if n_valid >= 50:
        n_each = max(n_each, 25)
        n_each = min(n_each, 50, n_valid // 2)
    if n_each < 1:
        return pd.Series(0.0, index=universe)

    short_idx = order[:n_each]
    long_idx = order[-n_each:]

    vwin = min(20, T)
    mkt_vol_ann = np.nanstd(mkt[-vwin:]) * np.sqrt(252.0)
    if not np.isfinite(mkt_vol_ann) or mkt_vol_ann <= 0:
        gross = 1.0
    else:
        gross = min(1.0, 0.12 / mkt_vol_ann)
        gross = max(0.3, gross)
    side_gross = 0.5 * gross
    max_w = 0.02

    def side_weights(indices):
        if len(indices) == 0:
            return np.array([])
        inv = 1.0 / idio[indices]
        s_inv = inv.sum()
        if s_inv <= 0 or not np.isfinite(s_inv):
            w = np.ones(len(indices)) / len(indices) * side_gross
        else:
            w = inv / s_inv * side_gross
        w = np.minimum(w, max_w)
        s = w.sum()
        if s > side_gross and s > 0:
            w = w * (side_gross / s)
        return w

    wl = side_weights(long_idx)
    ws = side_weights(short_idx)
    weights = np.zeros(n)
    weights[long_idx] = wl
    weights[short_idx] = -ws

    long_sum = weights[long_idx].sum()
    short_sum = -weights[short_idx].sum()
    target_side = min(long_sum, short_sum)
    if target_side <= 0:
        return pd.Series(0.0, index=universe)
    if long_sum > 0:
        weights[long_idx] *= target_side / long_sum
    else:
        weights[long_idx] = 0.0
    if short_sum > 0:
        weights[short_idx] *= target_side / short_sum
    else:
        weights[short_idx] = 0.0

    weights = np.clip(weights, -max_w, max_w)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    gross_sum = np.abs(weights).sum()
    if gross_sum > 1.0:
        weights = weights * (1.0 / gross_sum)
    return pd.Series(weights, index=universe)