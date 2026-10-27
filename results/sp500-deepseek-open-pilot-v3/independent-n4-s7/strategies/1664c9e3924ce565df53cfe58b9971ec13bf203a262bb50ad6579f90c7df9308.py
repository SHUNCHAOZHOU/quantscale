import numpy as np
import pandas as pd


def generate_weights(history, universe):
    tickers = pd.Index(universe)
    n_assets = len(tickers)
    if n_assets == 0:
        return pd.Series(dtype=float)

    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None:
        return pd.Series(0.0, index=tickers, dtype=float)

    ac = ac.reindex(columns=tickers)
    dates = ac.index
    if len(dates) == 0:
        return pd.Series(0.0, index=tickers, dtype=float)

    current = dates[-1]
    first_current = pd.Timestamp(current.year, current.month, 1)
    prior_dates = dates[dates < first_current]
    asof = prior_dates[-1] if len(prior_dates) > 0 else current

    ac_hist = ac.loc[:asof].tail(320)
    if ac_hist.shape[0] < 30:
        return pd.Series(0.0, index=tickers, dtype=float)

    hist_index = ac_hist.index
    C = ac_hist.to_numpy(dtype=float)
    n = C.shape[0]

    vol = history.get('volume')
    if vol is not None:
        vol_hist = vol.reindex(columns=tickers).reindex(index=hist_index)
        V = vol_hist.to_numpy(dtype=float)
    else:
        V = None

    close_raw = history.get('close')
    if close_raw is not None:
        close_hist = close_raw.reindex(columns=tickers).reindex(index=hist_index)
        CR = close_hist.to_numpy(dtype=float)
    else:
        CR = C.copy()

    elig = history.get('eligible')
    if elig is not None:
        elig_hist = elig.reindex(columns=tickers).loc[:asof]
        if len(elig_hist) > 0:
            elig_last = elig_hist.iloc[-1].fillna(False).astype(bool)
        else:
            elig_last = pd.Series(True, index=tickers)
    else:
        elig_last = pd.Series(True, index=tickers)

    last_close = C[-1, :]
    valid = np.isfinite(last_close) & (last_close > 0)
    valid &= elig_last.to_numpy(dtype=bool)

    if V is not None and V.shape == C.shape:
        with np.errstate(all='ignore'):
            DV = CR * V
            med_dv = np.nanmedian(DV[-63:, :], axis=0)
        valid &= np.isfinite(med_dv) & (med_dv >= 5e5)

    with np.errstate(all='ignore'):
        valid &= np.isfinite(CR[-1, :]) & (CR[-1, :] >= 3.0)

    score = np.full(n_assets, np.nan, dtype=float)

    if n >= 260:
        with np.errstate(all='ignore'):
            mom = C[-22, :] / C[-252, :] - 1.0
            rev = C[-1, :] / C[-22, :] - 1.0
            sub = C[-127:, :]
            rets = sub[1:, :] / sub[:-1, :] - 1.0
            vola = np.nanstd(rets, axis=0)
            score = (mom - 0.30 * rev) / (vola + 0.02)
    elif n >= 130:
        with np.errstate(all='ignore'):
            mom = C[-22, :] / C[-126, :] - 1.0
            rev = C[-1, :] / C[-22, :] - 1.0
            sub = C[-64:, :]
            rets = sub[1:, :] / sub[:-1, :] - 1.0
            vola = np.nanstd(rets, axis=0)
            score = (mom - 0.30 * rev) / (vola + 0.02)
    else:
        with np.errstate(all='ignore'):
            rev = C[-1, :] / C[-22, :] - 1.0
            score = -rev

    score = np.where(valid & np.isfinite(score), score, np.nan)
    finite = np.isfinite(score)
    if finite.sum() < 2:
        return pd.Series(0.0, index=tickers, dtype=float)

    idx = np.where(finite)[0]
    scores = score[idx]
    order = np.argsort(scores)
    sorted_idx = idx[order]
    M = len(sorted_idx)
    N = min(50, M // 2)
    if N == 0:
        return pd.Series(0.0, index=tickers, dtype=float)

    longs = sorted_idx[-N:]
    shorts = sorted_idx[:N]
    w = min(0.02, 0.5 / float(N))

    weights = np.zeros(n_assets, dtype=float)
    weights[longs] = w
    weights[shorts] = -w

    return pd.Series(weights, index=tickers, dtype=float)
