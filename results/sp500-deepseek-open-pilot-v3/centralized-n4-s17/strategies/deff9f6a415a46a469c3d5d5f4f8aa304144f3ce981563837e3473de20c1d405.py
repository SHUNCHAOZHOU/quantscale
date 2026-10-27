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

    adj = adj.sort_index().tail(300).astype(float)
    if len(adj) < 132:
        return zeros

    A = adj.where(adj > 0)

    elig = _pick(history, ['eligible'], universe)
    eligible = pd.Series(True, index=universe, dtype=bool)
    if elig is not None and len(elig) > 0:
        try:
            last = elig.sort_index().iloc[-1].reindex(universe)
            eligible = last.fillna(False).astype(bool)
        except Exception:
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

    score = (
        0.55 * mom120.rank(pct=True)
        + 0.25 * mom60.rank(pct=True)
        + 0.20 * df['prox'].rank(pct=True)
    )
    score = score.replace([np.inf, -np.inf], np.nan).dropna()

    m = len(score)
    if m < 4:
        return zeros

    k = min(25, m // 2)
    if k < 1:
        return zeros

    longs = score.nlargest(k).index
    shorts = score.nsmallest(k).index
    w = min(0.02, 0.5 / float(k))

    weights = zeros.copy()
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    return weights.fillna(0.0).replace([np.inf, -np.inf], 0.0)
