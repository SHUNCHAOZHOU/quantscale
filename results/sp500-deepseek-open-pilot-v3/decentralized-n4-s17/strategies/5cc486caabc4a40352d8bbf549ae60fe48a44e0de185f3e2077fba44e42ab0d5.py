import numpy as np
import pandas as pd

def _zeros(universe):
    return pd.Series(0.0, index=universe, dtype=float)

def generate_weights(history, universe):
    tickers = list(universe)
    if len(tickers) == 0:
        return pd.Series(dtype=float)

    n = len(tickers)
    ret5 = np.full(n, np.nan, dtype=float)
    vol20 = np.full(n, np.nan, dtype=float)
    liq = np.full(n, np.nan, dtype=float)

    for i, t in enumerate(tickers):
        df = history.get(t)
        if df is None:
            continue
        try:
            m = len(df)
        except Exception:
            continue
        if m == 0:
            continue

        sub = df.iloc[-min(m, 30):]
        if 'adjusted_close' not in sub.columns:
            continue

        ac = sub['adjusted_close'].to_numpy(dtype=float)
        if len(ac) < 8:
            continue
        if (not np.isfinite(ac[-1])) or ac[-1] <= 0.0:
            continue

        if 'eligible' in sub.columns:
            e = sub['eligible'].iloc[-1]
            if not pd.notna(e) or not bool(e):
                continue

        if len(ac) >= 6:
            p0 = ac[-6]
            if np.isfinite(p0) and p0 > 0.0:
                ret5[i] = ac[-1] / p0 - 1.0

        if len(ac) >= 21:
            vals = ac[-21:]
            ok = np.isfinite(vals) & (vals > 0.0)
            if ok.sum() >= 15:
                logs = np.log(vals)
                lr = np.diff(logs)
                lr = lr[np.isfinite(lr)]
                if len(lr) >= 10:
                    vol20[i] = float(np.std(lr, ddof=1)) * np.sqrt(252.0)

        if 'volume' in sub.columns and len(ac) >= 5:
            v = sub['volume'].to_numpy(dtype=float)[-20:]
            p = ac[-20:]
            k = min(len(v), len(p))
            if k >= 5:
                v = v[-k:]
                p = p[-k:]
                good = np.isfinite(v) & np.isfinite(p) & (v >= 0.0) & (p > 0.0)
                if good.sum() >= 5:
                    dv = v[good] * p[good]
                    liq[i] = float(np.median(dv))

    valid = (
        np.isfinite(ret5)
        & np.isfinite(vol20)
        & (vol20 > 0.0)
        & np.isfinite(liq)
        & (liq > 0.0)
    )
    if valid.sum() < 4:
        return _zeros(universe)

    liq_valid = liq[valid]
    liq_threshold = float(np.median(liq_valid))
    liquid = valid & (liq >= liq_threshold)
    if liquid.sum() < 4:
        liquid = valid

    score = np.full(n, np.nan, dtype=float)
    r = ret5[liquid].copy()
    if len(r) >= 4:
        lo = float(np.percentile(r, 2.5))
        hi = float(np.percentile(r, 97.5))
        r = np.clip(r, lo, hi)
    score_liquid = -r / np.maximum(vol20[liquid], 0.10)
    score[liquid] = score_liquid

    candidates = np.where(np.isfinite(score))[0]
    if len(candidates) < 2:
        return _zeros(universe)

    order = candidates[np.argsort(-score[candidates], kind='mergesort')]
    side = min(25, len(order) // 2)
    if side < 1:
        return _zeros(universe)

    long_idx = order[:side]
    short_idx = order[-side:]
    per = min(0.019999, 0.5 / float(side))

    w = pd.Series(0.0, index=universe, dtype=float)
    for idx in long_idx:
        w.iloc[int(idx)] = per
    for idx in short_idx:
        w.iloc[int(idx)] = -per

    return w