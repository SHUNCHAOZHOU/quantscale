import numpy as np
import pandas as pd


def _get_field(history, candidates, universe):
    if isinstance(history, dict):
        for key in candidates:
            if key in history:
                obj = history[key]
                if isinstance(obj, pd.DataFrame):
                    try:
                        return obj.reindex(columns=universe)
                    except Exception:
                        return None
    return None


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zeros

    adj = _get_field(history, ['adjusted_close'], universe)
    close = _get_field(history, ['close'], universe)
    if adj is None:
        adj = close
    elif close is not None:
        adj = adj.combine_first(close.reindex_like(adj))
    if adj is None or adj.shape[1] == 0:
        return zeros

    adj = adj.sort_index().tail(300)
    if len(adj) < 60:
        return zeros

    eligible = pd.Series(True, index=universe, dtype=bool)
    elig_df = _get_field(history, ['eligible'], universe)
    if elig_df is not None and len(elig_df) > 0:
        last = elig_df.sort_index().iloc[-1]
        last = last.reindex(universe).fillna(True)
        try:
            eligible = last.astype(bool)
        except Exception:
            eligible = pd.Series(True, index=universe, dtype=bool)

    skip = 5
    rets_win = None
    market = None
    if len(adj) >= 126:
        rec = adj.tail(130)
        rets = rec.pct_change()
        if len(rets) >= 126:
            rets_win = rets.iloc[-125:-5]
            market = rets_win.mean(axis=1, skipna=True)

    rows = []
    for t in universe:
        if not bool(eligible.get(t, True)):
            continue
        s = adj[t].dropna()
        if len(s) < 66:
            continue
        vals = s.values.astype(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 66 or vals[-1] <= 0:
            continue

        def ret_vol(lookback):
            end = len(vals) - skip
            if end <= lookback:
                return np.nan, np.nan
            start = end - (lookback + 1)
            if start < 0:
                return np.nan, np.nan
            w = vals[start:end]
            if len(w) < lookback + 1 or w[0] <= 0 or w[-1] <= 0:
                return np.nan, np.nan
            ret = w[-1] / w[0] - 1.0
            lr = np.diff(np.log(w))
            vol = float(np.std(lr, ddof=1)) if len(lr) > 1 else np.nan
            return ret, vol

        r120, v120 = ret_vol(120)
        r60, v60 = ret_vol(60)
        if not np.isfinite(r120) or not np.isfinite(v120) or v120 <= 0:
            continue
        if not np.isfinite(r60) or not np.isfinite(v60) or v60 <= 0:
            r60, v60 = r120, v120

        end = len(vals) - skip
        if end < 60:
            continue
        start = max(0, end - 252)
        high = np.nanmax(vals[start:end])
        if not np.isfinite(high) or high <= 0:
            continue
        prox = vals[end - 1] / high
        if not np.isfinite(prox):
            continue

        resid = np.nan
        if rets_win is not None and t in rets_win.columns:
            r = rets_win[t].values.astype(float)
            m = market.values.astype(float)
            mask = np.isfinite(r) & np.isfinite(m)
            if mask.sum() >= 40:
                rm = r[mask]
                mm = m[mask]
                var = np.var(mm, ddof=1)
                if var > 1e-12:
                    beta = np.cov(rm, mm, ddof=1)[0, 1] / var
                    resid = float(np.sum(rm - beta * mm))

        rows.append({'ticker': t, 'r120': r120, 'r60': r60,
                     'v120': v120, 'v60': v60, 'prox': prox, 'resid': resid})

    if len(rows) < 4:
        return zeros

    df = pd.DataFrame(rows).set_index('ticker')
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['r120', 'v120', 'prox'])
    if len(df) < 4:
        return zeros

    eps = 1e-12
    mom120 = df['r120'] / (df['v120'] * np.sqrt(120.0) + eps)
    mom60 = df['r60'].fillna(df['r120']) / (df['v60'].fillna(df['v120']) * np.sqrt(60.0) + eps)
    resid_mom = df['resid'].fillna(0.0) / (df['v120'] * np.sqrt(120.0) + eps)

    def rp(s, ascending=True):
        return pd.Series(s, dtype=float).rank(pct=True, ascending=ascending).fillna(0.5)

    score = (0.40 * rp(mom120) +
             0.20 * rp(mom60) +
             0.15 * rp(resid_mom) +
             0.15 * rp(df['prox']) -
             0.10 * rp(df['v120'], ascending=True))
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 4:
        return zeros

    n = len(score)
    k = min(25, n // 2)
    if k < 2:
        return zeros

    longs = score.nlargest(k).index
    shorts = score.nsmallest(k).index
    w = min(0.02, 1.0 / (2.0 * k))

    weights = zeros.copy()
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights
