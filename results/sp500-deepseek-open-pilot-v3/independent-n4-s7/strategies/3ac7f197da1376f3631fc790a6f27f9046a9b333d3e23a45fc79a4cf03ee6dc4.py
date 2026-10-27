import numpy as np
import pandas as pd

def _get(history, keys):
    for k in keys:
        if k in history:
            df = history[k]
            if isinstance(df, pd.DataFrame) and df.shape[0] > 0:
                return df
    return None

def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)
    zeros = pd.Series(0.0, index=universe)

    px = _get(history, ('adjusted_close', 'adj_close', 'close'))
    if px is None:
        return zeros
    cols = [c for c in universe if c in px.columns]
    if len(cols) == 0:
        return zeros

    p = px[cols].copy()
    if len(p) < 40:
        return zeros
    tail_n = min(len(p), 260)
    pt = p.iloc[-tail_n:]
    arr = pt.to_numpy(dtype=float)
    n = arr.shape[0]

    edf = _get(history, ('eligible', 'tradable'))
    if edf is not None:
        try:
            last = edf.reindex(columns=cols).iloc[-1]
            elig = last.fillna(False).astype(bool)
        except Exception:
            elig = pd.Series(True, index=cols)
    else:
        elig = pd.Series(True, index=cols)

    vdf = _get(history, ('volume', 'vol'))
    if vdf is not None:
        try:
            v = vdf.reindex(index=pt.index, columns=cols)
            c = p.reindex(index=pt.index, columns=cols)
            dv = v * c
            adv = dv.tail(20).mean()
            liq = adv.notna() & (adv > 1e6)
        except Exception:
            liq = pd.Series(True, index=cols)
    else:
        liq = pd.Series(True, index=cols)

    look = 21
    signals = {}
    for j, t in enumerate(cols):
        s = arr[:, j]
        if not np.isfinite(s[-1]) or s[-1] <= 0.0:
            continue
        if n <= look:
            continue
        p1 = s[-1]
        p0 = s[-1 - look]
        if not (np.isfinite(p1) and np.isfinite(p0) and p0 > 0.0):
            continue
        ret = p1 / p0 - 1.0

        rets = s[1:] / s[:-1] - 1.0
        rets = rets[np.isfinite(rets)]
        if len(rets) > 63:
            rets = rets[-63:]
        if len(rets) >= 15:
            vol = float(np.std(rets, ddof=1)) * np.sqrt(252.0)
        else:
            vol = np.nan
        if not np.isfinite(vol) or vol <= 0.0:
            vol = 0.25
        vol = max(vol, 0.05)

        signals[t] = -ret / vol

    if len(signals) < 4:
        return zeros
    sig = pd.Series(signals)

    keep = []
    for t in sig.index:
        if t in elig.index and not bool(elig.loc[t]):
            continue
        if t in liq.index and not bool(liq.loc[t]):
            continue
        keep.append(t)
    sig = sig.loc[keep]
    if len(sig) < 4:
        return zeros

    n_side = min(25, len(sig) // 2)
    if n_side < 1:
        return zeros

    ranked = sig.sort_values(ascending=False)
    longs = list(ranked.index[:n_side])
    shorts = list(ranked.index[-n_side:])

    short_set = set(shorts)
    longs = [x for x in longs if x not in short_set]
    long_set = set(longs)
    shorts = [x for x in shorts if x not in long_set]

    n_side = min(len(longs), len(shorts))
    if n_side < 1:
        return zeros
    longs = longs[:n_side]
    shorts = shorts[:n_side]

    target_gross = 0.98
    w = min(0.02, target_gross / (2.0 * n_side))

    weights = pd.Series(0.0, index=universe)
    for t in longs:
        weights.loc[t] = w
    for t in shorts:
        weights.loc[t] = -w

    weights = weights.clip(-0.02, 0.02)
    return weights.fillna(0.0)
