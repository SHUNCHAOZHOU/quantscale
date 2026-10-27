import numpy as np
import pandas as pd

def _field(history, names):
    for k in names:
        if k in history:
            df = history[k]
            if isinstance(df, pd.DataFrame) and df.shape[0] > 0:
                return df
    return None

def _zeros(universe):
    return pd.Series(0.0, index=universe)

def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)
    px = _field(history, ('adjusted_close', 'adj_close', 'close'))
    if px is None:
        return _zeros(universe)
    cols = [c for c in universe if c in px.columns]
    if len(cols) == 0:
        return _zeros(universe)
    p = px[cols].copy()
    if len(p) < 30:
        return _zeros(universe)
    tail_n = min(len(p), 320)
    pt = p.iloc[-tail_n:]
    arr = pt.to_numpy(dtype=float)
    n = arr.shape[0]

    edf = _field(history, ('eligible', 'tradable'))
    if edf is not None:
        try:
            last_elig = edf.reindex(columns=cols).iloc[-1]
            elig = last_elig.fillna(False).astype(bool)
        except Exception:
            elig = pd.Series(True, index=cols)
    else:
        elig = pd.Series(True, index=cols)

    vdf = _field(history, ('volume', 'vol'))
    if vdf is not None:
        try:
            v = vdf.reindex(index=pt.index, columns=cols)
            dv = v * pt
            adv = dv.tail(20).mean() if len(dv) >= 20 else dv.mean()
            liq = adv.notna() & (adv > 0)
        except Exception:
            liq = pd.Series(True, index=cols)
    else:
        liq = pd.Series(True, index=cols)

    sig = {}
    for j, ticker in enumerate(cols):
        s = arr[:, j]
        if not np.isfinite(s[-1]) or s[-1] <= 0.0:
            continue

        def lag_ret(look, skip):
            i1 = n - 1 - skip
            i0 = i1 - look
            if i0 < 0:
                return np.nan
            p1 = s[i1]
            p0 = s[i0]
            if np.isfinite(p1) and np.isfinite(p0) and p0 > 0.0 and p1 > 0.0:
                return p1 / p0 - 1.0
            return np.nan

        mom = lag_ret(252, 21)
        if not np.isfinite(mom):
            mom = lag_ret(126, 21)
        if not np.isfinite(mom):
            mom = lag_ret(63, 21)
        if not np.isfinite(mom):
            continue

        rets = s[1:] / s[:-1] - 1.0
        if len(rets) > 126:
            rets = rets[-126:]
        rets = rets[np.isfinite(rets)]
        if len(rets) >= 20:
            vol = float(np.std(rets, ddof=1)) * np.sqrt(252.0)
        else:
            vol = np.nan
        if not np.isfinite(vol) or vol <= 0.0:
            vol = 0.25
        vol = max(vol, 0.05)

        sig[ticker] = mom / vol

    if len(sig) < 4:
        return _zeros(universe)
    sigs = pd.Series(sig)

    keep = []
    for t in sigs.index:
        if t in elig.index and not bool(elig.loc[t]):
            continue
        if t in liq.index and not bool(liq.loc[t]):
            continue
        keep.append(t)
    sigs = sigs.loc[keep]
    if len(sigs) < 4:
        return _zeros(universe)

    n_side = int(min(50, max(1, len(sigs) // 10)))
    n_side = min(n_side, len(sigs) // 2)
    if n_side < 1:
        return _zeros(universe)

    ranked = sigs.sort_values(ascending=False)
    longs = list(ranked.index[:n_side])
    shorts = list(ranked.index[-n_side:])

    target_gross = 0.98
    w = min(0.02, target_gross / (2.0 * n_side))
    weights = pd.Series(0.0, index=universe)
    for t in longs:
        weights.loc[t] = w
    for t in shorts:
        weights.loc[t] = -w

    weights = weights.clip(-0.02, 0.02)
    return weights.fillna(0.0)
