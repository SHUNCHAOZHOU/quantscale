import numpy as np
import pandas as pd

_CACHE = {}


def _rank(x):
    s = pd.Series(np.asarray(x, dtype=float))
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return np.full(len(s), 0.5)
    return s.rank(pct=True).fillna(0.5).to_numpy()


def generate_weights(history, universe):
    tickers = pd.Index(universe)
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=tickers)

    adj_full = history.get('adjusted_close')
    if adj_full is None:
        adj_full = history.get('close')
    if adj_full is None or adj_full.shape[0] < 2:
        return zero

    dates = adj_full.index
    y = dates.year.to_numpy()
    m = dates.month.to_numpy()
    change = (y[1:] != y[:-1]) | (m[1:] != m[:-1])
    completed = np.where(change)[0]
    if len(completed) == 0:
        return zero
    pos = int(completed[-1])
    if pos < 126:
        return zero

    key = (dates[pos], tuple(tickers))
    cached = _CACHE.get(key)
    if cached is not None:
        return pd.Series(cached.copy(), index=tickers)

    start = max(0, pos + 1 - 300)
    adj_df = adj_full.iloc[start:pos + 1].reindex(columns=tickers)
    idx = adj_df.index

    close_full = history.get('close')
    if close_full is None:
        close_df = adj_df
    else:
        close_df = close_full.reindex(index=idx, columns=tickers)

    vol_full = history.get('volume')
    if vol_full is None:
        vol_df = pd.DataFrame(1.0, index=idx, columns=tickers)
    else:
        vol_df = vol_full.reindex(index=idx, columns=tickers)

    elig_full = history.get('eligible')
    if elig_full is not None:
        elig_last = elig_full.reindex(index=idx, columns=tickers).iloc[-1]
        eligible = elig_last.fillna(False).astype(bool).to_numpy()
    else:
        eligible = np.ones(n, dtype=bool)

    adj = adj_df.to_numpy(dtype=float)
    close = close_df.to_numpy(dtype=float)
    volume = vol_df.to_numpy(dtype=float)
    L = adj.shape[0]
    if L < 126:
        return zero

    last_adj = adj[-1]
    valid_price = np.isfinite(last_adj) & (last_adj > 0)
    if not np.any(valid_price):
        return zero

    with np.errstate(divide='ignore', invalid='ignore'):
        p1 = adj[-6]
        p0 = adj[-126] if L >= 126 else adj[0]
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
    if np.sum(mask) < 10:
        mask = eligible & valid_price & np.isfinite(mom) & np.isfinite(rev)
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

    if len(_CACHE) >= 8:
        first_key = None
        for k in _CACHE:
            first_key = k
            break
        if first_key is not None:
            _CACHE.pop(first_key, None)
    _CACHE[key] = weights.copy()

    return pd.Series(weights, index=tickers)