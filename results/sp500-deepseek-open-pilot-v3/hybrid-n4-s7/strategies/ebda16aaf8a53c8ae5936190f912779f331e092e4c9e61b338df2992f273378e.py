import numpy as np
import pandas as pd


def generate_weights(history, universe):
    u = pd.Index(universe)
    zeros = pd.Series(0.0, index=u, dtype=float)
    if len(u) == 0:
        return zeros

    ac = history.get('adjusted_close')
    if ac is None or not isinstance(ac, pd.DataFrame) or ac.empty:
        ac = history.get('close')
    if ac is None or not isinstance(ac, pd.DataFrame) or ac.empty:
        return zeros

    ac = ac.reindex(columns=u)
    dates = pd.DatetimeIndex(ac.index)
    if len(dates) < 126:
        return zeros

    cur = dates[-1]
    ym = dates.year.to_numpy() * 12 + dates.month.to_numpy()
    cur_ym = cur.year * 12 + cur.month
    mask = ym < cur_ym
    if not mask.any():
        return zeros

    sig_pos = int(np.where(mask)[0][-1])
    start = max(0, sig_pos + 1 - 300)
    sub = ac.iloc[start:sig_pos + 1]
    if sub.shape[0] < 60:
        return zeros

    prices = sub.to_numpy(dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        rets = prices[1:] / prices[:-1] - 1.0
    rets = np.clip(rets, -0.5, 0.5)
    rets[~np.isfinite(rets)] = np.nan
    T = rets.shape[0]
    if T < 126:
        return zeros

    mkt = np.nanmean(rets, axis=1)
    mkt = np.where(np.isfinite(mkt), mkt, 0.0)

    bwin = min(252, T)
    rb = rets[-bwin:]
    mb = mkt[-bwin:]
    mb_mean = np.nanmean(mb)
    if not np.isfinite(mb_mean):
        mb_mean = 0.0
    mb_dev = mb - mb_mean
    den = np.nanmean(mb_dev ** 2)
    if not np.isfinite(den) or den <= 0:
        beta = np.ones(prices.shape[1])
    else:
        rmean = np.nanmean(rb, axis=0)
        num = np.nanmean((rb - rmean) * mb_dev[:, None], axis=0)
        beta = num / den
        beta = np.where(np.isfinite(beta), beta, 1.0)
        beta = np.clip(beta, 0.2, 3.0)

    resid = rets - beta[None, :] * mkt[:, None]

    if T >= 252:
        rwin = resid[-252:-21]
    else:
        rwin = resid[-126:-21]
    if rwin.shape[0] < 20:
        return zeros

    score_raw = np.nansum(rwin, axis=0)
    vol = np.nanstd(rwin, axis=0, ddof=1)
    obs = np.sum(np.isfinite(rwin), axis=0)
    med_vol = np.nanmedian(vol)
    if not np.isfinite(med_vol) or med_vol <= 0:
        med_vol = 0.02
    vol = np.where(np.isfinite(vol) & (vol > 0), vol, med_vol)
    vol = np.clip(vol, 0.005, 0.20)
    score = score_raw / vol

    last = prices[-1]

    dollar = None
    vdf = history.get('volume')
    if vdf is not None and isinstance(vdf, pd.DataFrame) and not vdf.empty:
        vdf = vdf.reindex(columns=u)
        vsub = vdf.iloc[start:sig_pos + 1].tail(60)
        psub = sub.tail(vsub.shape[0])
        if vsub.shape[0] == psub.shape[0] and vsub.shape[0] > 0:
            dv = psub.to_numpy(dtype=float) * vsub.to_numpy(dtype=float)
            with np.errstate(all='ignore'):
                dollar = np.nanmedian(dv, axis=0)

    valid = np.isfinite(score) & np.isfinite(last) & (last > 5.0) & (obs >= max(20, rwin.shape[0] * 0.8))

    edf = history.get('eligible')
    if edf is not None and isinstance(edf, pd.DataFrame) and len(edf) > 0:
        e = edf.reindex(columns=u).iloc[-1].to_numpy(dtype=float)
        valid &= np.isfinite(e) & (e > 0.5)

    if dollar is not None:
        valid &= np.isfinite(dollar) & (dollar > 0)
        if valid.sum() >= 20:
            thr = np.nanpercentile(dollar[valid], 20)
            valid &= (dollar >= thr)

    n_valid = int(valid.sum())
    if n_valid < 20:
        return zeros

    idx = np.where(valid)[0]
    order = idx[np.argsort(score[idx], kind='mergesort')]
    n_side = min(50, n_valid // 2)
    if n_side < 5:
        return zeros

    longs = order[-n_side:]
    shorts = order[:n_side]
    side_gross = min(0.5, 0.02 * n_side)
    w = side_gross / n_side

    weights = np.zeros(len(u), dtype=float)
    weights[longs] = w
    weights[shorts] = -w
    weights = np.clip(weights, -0.02, 0.02)
    g = np.abs(weights).sum()
    if g > 1.0 and g > 0:
        weights *= 1.0 / g
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.Series(weights, index=u)