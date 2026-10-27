import numpy as np
import pandas as pd


def _get_df(history, names):
    for name in names:
        obj = history.get(name)
        if isinstance(obj, pd.DataFrame) and not obj.empty:
            return obj
    return None


def _zero(universe):
    return pd.Series(0.0, index=universe, dtype=float)


def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)

    price_df = _get_df(history, ['adjusted_close', 'close'])
    if price_df is None:
        return _zero(universe)

    price_df = price_df.reindex(columns=universe).astype(float)
    n_rows = len(price_df)
    if n_rows < 100:
        return _zero(universe)

    lookback = min(n_rows, 700)
    price = price_df.iloc[-lookback:].copy()
    if len(price) < 100:
        return _zero(universe)

    vol_df = _get_df(history, ['volume'])
    if vol_df is not None:
        vol_df = vol_df.reindex(columns=universe).astype(float).iloc[-lookback:]
        volmed = vol_df.rolling(20, min_periods=5).median()
        vol_vals = volmed.values
    else:
        vol_vals = None

    elig_df = _get_df(history, ['eligible'])
    if elig_df is not None:
        elig = elig_df.reindex(columns=universe).iloc[-lookback:].fillna(True).astype(bool)
        elig_vals = elig.values
    else:
        elig_vals = None

    price_vals = price.values
    n_assets = price_vals.shape[1]
    ret63 = np.full((price_vals.shape[0], n_assets), np.nan, dtype=float)
    if price_vals.shape[0] > 63:
        ret63[63:] = price_vals[63:] / price_vals[:-63] - 1.0

    max_days = 500
    start = max(63, len(price) - max_days)
    prev = np.zeros(n_assets, dtype=float)
    max_turnover = 0.02
    alpha = 0.05
    n_side_cap = 50
    min_side = 5
    min_volume = 100000.0
    min_price = 5.0

    for pos in range(start, len(price)):
        p = price_vals[pos]
        r = ret63[pos]
        valid = np.isfinite(p) & (p > min_price) & np.isfinite(r) & (r > -0.95) & (r < 5.0)
        if elig_vals is not None:
            valid &= elig_vals[pos]
        if vol_vals is not None:
            vm = vol_vals[pos]
            valid &= np.isfinite(vm) & (vm > min_volume)

        idx = np.where(valid)[0]
        if len(idx) >= 2 * min_side:
            scores = -r[idx]
            order = np.argsort(scores)[::-1]
            n_side = min(n_side_cap, len(idx) // 2)
            if n_side >= min_side:
                long_idx = idx[order[:n_side]]
                short_idx = idx[order[-n_side:]]
                w = min(0.5 / float(n_side), 0.02)
                target = np.zeros(n_assets, dtype=float)
                target[long_idx] = w
                target[short_idx] = -w
            else:
                target = np.zeros(n_assets, dtype=float)
        else:
            target = np.zeros(n_assets, dtype=float)

        delta = target - prev
        l1 = np.abs(delta).sum()
        if l1 > 1e-12:
            step = min(alpha, max_turnover / l1)
            prev = prev + step * delta

        prev = np.clip(prev, -0.02, 0.02)
        gross = np.abs(prev).sum()
        if gross > 1.0:
            prev = prev / gross
        net = prev.sum()
        if abs(net) > 0.01:
            prev = prev - net / float(n_assets)
            prev = np.clip(prev, -0.02, 0.02)
            gross = np.abs(prev).sum()
            if gross > 1.0:
                prev = prev / gross

    return pd.Series(prev, index=universe, dtype=float).fillna(0.0)