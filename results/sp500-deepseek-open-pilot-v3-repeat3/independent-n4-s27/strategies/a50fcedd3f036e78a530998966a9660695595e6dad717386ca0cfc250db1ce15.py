import numpy as np
import pandas as pd


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

    dates = idx
    y = dates.year.to_numpy()
    m = dates.month.to_numpy()
    if len(dates) > 1:
        change = (y[1:] != y[:-1]) | (m[1:] != m[:-1])
        cand = np.where(change)[0]
    else:
        cand = np.array([], dtype=int)

    if len(cand) > 0:
        pos = int(cand[-1])
    else:
        pos = len(dates) - 1

    if pos < 60:
        if len(dates) - 1 >= 60:
            pos = len(dates) - 1
        else:
            return zero

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

    vol_tail = ret[-min(60, ret.shape[0]):] if ret.shape[0] > 0 else ret
    if vol_tail.shape[0] >= 2:
        vol = np.nanstd(vol_tail, axis=0)
    else:
        vol = np.full(n, np.nan)
    vol = np.where(np.isfinite(vol), vol, np.nan)

    dollar = close * volume
    liq_tail = dollar[-min(60, L):]
    if liq_tail.shape[0] > 0:
        liq = np.nanmedian(liq_tail, axis=0)
    else:
        liq = np.zeros(n)
    liq = np.where(np.isfinite(liq), liq, 0.0)

    mask = eligible & valid_price & np.isfinite(mom) & np.isfinite(rev) & np.isfinite(vol) & (vol > 1e-8) & (liq >= 5e5)
    if np.sum(mask) < 10:
        mask = eligible & valid_price & np.isfinite(mom)
    if np.sum(mask) == 0:
        mask = valid_price
    if np.sum(mask) == 0:
        return zero

    cand_idx = np.where(mask)[0]
    if len(cand_idx) == 0:
        return zero

    def _rank(x):
        s = pd.Series(x)
        if s.notna().sum() == 0:
            return np.full(len(x), 0.5)
        return s.rank(pct=True).fillna(0.5).to_numpy()

    r_mom = _rank(mom[cand_idx])
    r_rev = _rank(rev[cand_idx])
    r_vol = _rank(vol[cand_idx])
    r_liq = _rank(liq[cand_idx])

    score = 0.55 * r_mom + 0.15 * r_liq - 0.20 * r_rev - 0.20 * r_vol

    n_select = min(60, len(cand_idx))
    order = np.argsort(-score, kind='mergesort')
    sel = cand_idx[order[:n_select]]
    w = min(1.0 / n_select, 0.02)

    weights = np.zeros(n, dtype=float)
    weights[sel] = w
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(weights, -0.02, 0.02)

    return pd.Series(weights, index=tickers)