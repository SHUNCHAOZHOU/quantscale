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
    wd = dates.weekday
    fri_idx = np.where(wd == 4)[0]
    if len(fri_idx) > 0 and dates[fri_idx[-1]] <= current:
        asof = dates[fri_idx[-1]]
    else:
        asof = current

    ac_hist = ac.loc[:asof].tail(300)
    if ac_hist.shape[0] < 30:
        return pd.Series(0.0, index=tickers, dtype=float)

    hist_index = ac_hist.index
    C = ac_hist.to_numpy(dtype=float)
    n = C.shape[0]

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

    vol = history.get('volume')
    close_raw = history.get('close')
    if vol is not None:
        vol_hist = vol.reindex(columns=tickers).reindex(index=hist_index)
        V = vol_hist.to_numpy(dtype=float)
        if close_raw is not None:
            CR = close_raw.reindex(columns=tickers).reindex(index=hist_index).to_numpy(dtype=float)
        else:
            CR = C.copy()
        with np.errstate(all='ignore'):
            DV = CR * V
            med_dv = np.nanmedian(DV[-63:, :], axis=0)
        valid &= np.isfinite(med_dv) & (med_dv >= 1e6)

    if close_raw is not None:
        CR_last = close_raw.reindex(columns=tickers).loc[:asof].iloc[-1].to_numpy(dtype=float)
        valid &= np.isfinite(CR_last) & (CR_last >= 3.0)
    else:
        valid &= np.isfinite(last_close) & (last_close >= 3.0)

    score = np.full(n_assets, np.nan, dtype=float)
    if n >= 22:
        with np.errstate(all='ignore'):
            ret5 = C[-1, :] / C[-6, :] - 1.0
            ret21 = C[-1, :] / C[-22, :] - 1.0
            sub = C[-22:, :]
            rets = sub[1:, :] / sub[:-1, :] - 1.0
            vola = np.nanstd(rets, axis=0)
            score = -(0.7 * ret5 + 0.3 * ret21) / (vola + 0.01)
    elif n >= 6:
        with np.errstate(all='ignore'):
            ret5 = C[-1, :] / C[-6, :] - 1.0
            sub = C[-6:, :]
            rets = sub[1:, :] / sub[:-1, :] - 1.0
            vola = np.nanstd(rets, axis=0)
            score = -ret5 / (vola + 0.01)
    else:
        return pd.Series(0.0, index=tickers, dtype=float)

    score = np.where(valid & np.isfinite(score), score, np.nan)
    finite = np.isfinite(score)
    if finite.sum() < 2:
        return pd.Series(0.0, index=tickers, dtype=float)

    idx = np.where(finite)[0]
    order = np.argsort(score[idx])
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
