import numpy as np
import pandas as pd


def _zscore(s):
    s = s.replace([np.inf, -np.inf], np.nan)
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=s.index)
    z = (s - mu) / sd
    return z.clip(-3.0, 3.0)


def generate_weights(history, universe):
    u = pd.Index(universe)
    if len(u) == 0:
        return pd.Series(dtype=float, index=u)

    adj = history.get('adjusted_close')
    if adj is None or not isinstance(adj, pd.DataFrame):
        adj = history.get('close')
    if adj is None or not isinstance(adj, pd.DataFrame):
        return pd.Series(0.0, index=u)

    adj = adj.reindex(columns=u)
    if len(adj) < 130:
        return pd.Series(0.0, index=u)

    idx = adj.index
    last_pos = len(idx) - 1
    mondays = np.where(idx.dayofweek == 0)[0]
    if len(mondays) > 0:
        base_pos = int(mondays[-1])
        if base_pos < 129:
            base_pos = last_pos
    else:
        base_pos = last_pos

    start = max(0, base_pos - 259)
    adj_w = adj.iloc[start:base_pos + 1]
    if len(adj_w) < 130:
        base_pos = last_pos
        start = max(0, base_pos - 259)
        adj_w = adj.iloc[start:base_pos + 1]
    if len(adj_w) < 130:
        return pd.Series(0.0, index=u)

    valid = pd.Series(True, index=u)
    last_adj = adj_w.iloc[-1]
    valid &= np.isfinite(last_adj.values) & (last_adj.values > 0)
    valid &= (adj_w.notna().sum().values >= min(130, len(adj_w)))

    elig = history.get('eligible')
    if elig is not None and isinstance(elig, pd.DataFrame):
        elig2 = elig.reindex(columns=u)
        if len(elig2) > 0:
            if base_pos < len(elig2):
                e = elig2.iloc[base_pos]
            else:
                e = elig2.iloc[-1]
            valid &= e.fillna(False).astype(bool).values

    volume = history.get('volume')
    if volume is not None and isinstance(volume, pd.DataFrame):
        vol2 = volume.reindex(columns=u).iloc[start:base_pos + 1]
        vtail = vol2.iloc[-20:]
        close = history.get('close')
        if close is not None and isinstance(close, pd.DataFrame):
            px = close.reindex(columns=u).iloc[start:base_pos + 1].iloc[-20:]
        else:
            px = adj_w.iloc[-20:]
        dollar = (vtail * px).median()
        valid &= np.isfinite(dollar.values) & (dollar.values > 1e6)

    arr_last = adj_w.iloc[-1]
    r5 = arr_last / adj_w.iloc[-6] - 1.0

    if len(adj_w) >= 127:
        mom = adj_w.iloc[-22] / adj_w.iloc[-127] - 1.0
    else:
        mom = pd.Series(np.nan, index=u)

    rets = adj_w.pct_change()
    vol = rets.iloc[-126:].std(ddof=0)
    vol = vol.replace(0.0, np.nan)

    rev_score = -r5 / (vol * np.sqrt(5.0))
    mom_score = mom / (vol * np.sqrt(105.0))
    rev_score = rev_score.replace([np.inf, -np.inf], np.nan)
    mom_score = mom_score.replace([np.inf, -np.inf], np.nan)

    signal = 0.7 * _zscore(mom_score) + 0.3 * _zscore(rev_score)
    signal = signal.replace([np.inf, -np.inf], np.nan)

    cand_mask = valid.values & np.isfinite(signal.values)
    cand = list(u[cand_mask])
    m = len(cand)
    if m < 10:
        return pd.Series(0.0, index=u)

    n = min(30, m // 2)
    if n < 1:
        return pd.Series(0.0, index=u)

    ordered = signal.loc[cand].sort_values(ascending=False, kind='mergesort')
    longs = ordered.index[:n]
    shorts = ordered.index[-n:]

    side_weight = 0.5 / n
    w = min(side_weight, 0.02)

    weights = pd.Series(0.0, index=u)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    return weights.astype(float)
