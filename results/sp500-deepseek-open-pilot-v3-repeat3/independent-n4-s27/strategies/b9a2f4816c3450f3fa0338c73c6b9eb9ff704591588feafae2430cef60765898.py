import numpy as np
import pandas as pd

_CACHE = {}


def _rank(x):
    s = pd.Series(x, dtype=float)
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return np.full(len(x), 0.5)
    return s.rank(pct=True).fillna(0.5).to_numpy()


def generate_weights(history, universe):
    tickers = pd.Index(universe)
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=tickers)

    adj_df = history.get('adjusted_close')
    if adj_df is None:
        adj_df = history.get('close')
    if adj_df is None:
        return zero
    adj_df = adj_df.reindex(columns=tickers)
    if adj_df.shape[0] == 0:
        return zero

    idx = adj_df.index
    dates = idx

    close_df = history.get('close')
    if close_df is None:
        close_df = adj_df
    else:
        close_df = close_df.reindex(index=idx, columns=tickers)

    vol_df = history.get('volume')
    if vol_df is None:
        vol_df = pd.DataFrame(1.0, index=idx, columns=tickers)
    else:
        vol_df = vol_df.reindex(index=idx, columns=tickers)

    elig_df = history.get('eligible')
    if elig_df is not None:
        elig_df = elig_df.reindex(index=idx, columns=tickers)

    flags = np.zeros(len(dates), dtype=bool)
    if len(dates) > 1:
        flags[:-1] = (dates[1:].year != dates[:-1].year) | (dates[1:].month != dates[:-1].month)
    if len(dates) > 0:
        try:
            nxt = dates[-1] + pd.offsets.BDay(1)
            flags[-1] = (nxt.year != dates[-1].year) or (nxt.month != dates[-1].month)
        except Exception:
            flags[-1] = False
    completed = np.where(flags)[0]
    if len(completed) > 0:
        pos = int(completed[-1])
    elif len(dates) >= 60:
        pos = len(dates) - 1
    else:
        return zero

    if pos < 60:
        return zero

    key = (dates[pos], tuple(tickers.tolist()))
    cached = _CACHE.get(key)
    if cached is not None:
        return pd.Series(cached.copy(), index=tickers)

    start = max(0, pos + 1 - 300)
    adj = adj_df.iloc[start:pos + 1].to_numpy(dtype=float)
    close = close_df.iloc[start:pos + 1].to_numpy(dtype=float)
    volume = vol_df.iloc[start:pos + 1].to_numpy(dtype=float)
    L = adj.shape[0]
    if L < 6:
        return zero

    if elig_df is not None:
        e_last = elig_df.iloc[pos].to_numpy()
        eligible = pd.Series(e_last).fillna(False).astype(bool).to_numpy()
    else:
        eligible = np.ones(n, dtype=bool)

    last_adj = adj[-1]
    valid_price = np.isfinite(last_adj) & (last_adj > 0)
    if not np.any(valid_price):
        return zero

    with np.errstate(divide='ignore', invalid='ignore'):
        if L >= 132:
            p1 = adj[-6]
            p0 = adj[-132]
        else:
            p1 = adj[-6]
            p0 = adj[0]
        mom = p1 / p0 - 1.0
        rev = adj[-1] / adj[-6] - 1.0
        ret = adj[1:] / adj[:-1] - 1.0

    mom = np.where(np.isfinite(mom), mom, np.nan)
    rev = np.where(np.isfinite(rev), rev, np.nan)
    ret = np.where(np.isfinite(ret), ret, np.nan)

    if ret.shape[0] >= 2:
        vol_tail = ret[-min(60, ret.shape[0]):]
        vol = np.nanstd(vol_tail, axis=0)
    else:
        vol = np.full(n, np.nan)
    vol = np.where(np.isfinite(vol), vol, np.nan)

    close_arr = np.where(np.isfinite(close), close, adj)
    dollar = close_arr * volume
    liq_tail = dollar[-min(60, L):]
    if liq_tail.shape[0] > 0:
        liq = np.nanmedian(liq_tail, axis=0)
    else:
        liq = np.zeros(n)
    liq = np.where(np.isfinite(liq), liq, 0.0)

    mask = eligible & valid_price & np.isfinite(mom) & np.isfinite(rev) & np.isfinite(vol) & (vol > 1e-8) & (liq >= 5e5)
    if np.sum(mask) < 4:
        mask = eligible & valid_price & np.isfinite(mom)
    if np.sum(mask) < 2:
        return zero

    cand_idx = np.where(mask)[0]
    n_cand = len(cand_idx)
    n_side = min(50, n_cand // 2)
    if n_side < 1:
        return zero

    r_mom = _rank(mom[cand_idx])
    r_rev = _rank(rev[cand_idx])
    r_vol = _rank(vol[cand_idx])
    r_liq = _rank(liq[cand_idx])

    score = 0.55 * r_mom + 0.15 * r_liq - 0.20 * r_rev - 0.20 * r_vol
    order = np.argsort(-score, kind='mergesort')
    long_sel = cand_idx[order[:n_side]]
    short_sel = cand_idx[order[-n_side:]]

    w = min(0.02, 0.5 / n_side)
    weights = np.zeros(n, dtype=float)
    weights[long_sel] = w
    weights[short_sel] = -w
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(weights, -0.02, 0.02)

    if len(_CACHE) >= 24:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = weights.copy()

    return pd.Series(weights, index=tickers)