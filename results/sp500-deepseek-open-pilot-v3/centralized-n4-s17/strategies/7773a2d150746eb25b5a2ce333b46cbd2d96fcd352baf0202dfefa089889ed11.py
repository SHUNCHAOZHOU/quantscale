import numpy as np
import pandas as pd


def _pick(history, keys, universe):
    if isinstance(history, dict):
        for k in keys:
            if k in history and isinstance(history[k], pd.DataFrame):
                try:
                    return history[k].reindex(columns=universe)
                except Exception:
                    return None
    return None


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zeros

    adj = _pick(history, ['adjusted_close', 'close'], universe)
    if adj is None:
        return zeros
    adj = adj.sort_index().tail(320).astype(float)
    if len(adj) < 132:
        return zeros

    A = adj.where(adj > 0)

    elig = _pick(history, ['eligible'], universe)
    if elig is not None and len(elig) > 0:
        try:
            last = elig.sort_index().iloc[-1].reindex(universe)
            eligible = last.fillna(False).astype(bool)
        except Exception:
            eligible = pd.Series(True, index=universe, dtype=bool)
    else:
        eligible = pd.Series(True, index=universe, dtype=bool)

    n = len(A)
    skip = 5
    end = n - 1 - skip
    if end < 126:
        return zeros

    p_end = A.iloc[end]
    r120 = p_end / A.iloc[end - 120] - 1.0
    r60 = p_end / A.iloc[end - 60] - 1.0

    rets = A / A.shift(1) - 1.0
    v120 = rets.iloc[max(0, end - 119):end + 1].std(ddof=1)
    v60 = rets.iloc[max(0, end - 59):end + 1].std(ddof=1)

    high = A.iloc[max(0, end - 251):end + 1].max()
    prox = p_end / high

    df = pd.DataFrame({
        'price': p_end,
        'r120': r120,
        'r60': r60,
        'v120': v120,
        'v60': v60,
        'prox': prox,
    }, index=universe)
    df = df.replace([np.inf, -np.inf], np.nan)

    good = eligible.reindex(universe).fillna(False)
    df = df.loc[good & df.notna().all(axis=1) & (df['price'] > 1.0) & (df['v120'] > 0.0)]
    if len(df) < 6:
        return zeros

    eps = 1e-12
    mom120 = df['r120'] / (df['v120'] * np.sqrt(120.0) + eps)
    mom60 = df['r60'] / (df['v60'].fillna(df['v120']) * np.sqrt(60.0) + eps)

    win = rets.iloc[max(0, end - 119):end + 1]
    X = win.to_numpy(dtype=float)
    mkt = win.mean(axis=1, skipna=True)
    m = mkt.to_numpy(dtype=float)
    if X.ndim == 2 and X.shape[0] > 2 and X.shape[1] > 0:
        mask = np.isfinite(X) & np.isfinite(m)[:, None]
        count = mask.sum(axis=0)
        safe_count = np.maximum(count, 1)
        x_mean = np.nansum(np.where(mask, X, np.nan), axis=0) / safe_count
        m_mean = np.nansum(np.where(mask, m[:, None], np.nan), axis=0) / safe_count
        cov = np.nansum(mask * (X - x_mean) * (m[:, None] - m_mean), axis=0) / np.maximum(count - 1, 1)
        var = np.nansum(mask * (m[:, None] - m_mean) ** 2, axis=0) / np.maximum(count - 1, 1)
        beta = np.where(var > 1e-12, cov / np.maximum(var, 1e-12), np.nan)
        resid_cum = np.nansum(mask * (X - beta * m[:, None]), axis=0)
        resid_cum = np.where((count >= 40) & np.isfinite(beta), resid_cum, np.nan)
        resid = pd.Series(resid_cum, index=win.columns).reindex(df.index)
    else:
        resid = pd.Series(np.nan, index=df.index)
    resid = resid.replace([np.inf, -np.inf], np.nan)
    resid_mom = resid / (df['v120'] * np.sqrt(120.0) + eps)

    def pct_rank(s):
        return pd.Series(s, index=df.index, dtype=float).rank(pct=True).fillna(0.5)

    score = -(
        0.45 * pct_rank(mom120)
        + 0.20 * pct_rank(mom60)
        + 0.20 * pct_rank(df['prox'])
        + 0.15 * pct_rank(resid_mom)
    )
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 4:
        return zeros

    mscore = len(score)
    k = min(25, mscore // 2)
    if k < 1:
        return zeros

    longs = score.nlargest(k).index
    shorts = score.nsmallest(k).index
    w = min(0.02, 1.0 / (2.0 * float(k)))

    weights = zeros.copy()
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights