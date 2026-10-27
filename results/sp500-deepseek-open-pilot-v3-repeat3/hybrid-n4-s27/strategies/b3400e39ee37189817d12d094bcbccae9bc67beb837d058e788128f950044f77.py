import math
import numpy as np
import pandas as pd


def _price_frame(history):
    for name in ('adjusted_close', 'close'):
        df = history.get(name)
        if isinstance(df, pd.DataFrame) and df.shape[0] > 0 and df.shape[1] > 0:
            return df
    return None


def generate_weights(history, universe):
    universe = pd.Index(list(universe))
    weights = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return weights

    raw_prices = _price_frame(history)
    if raw_prices is None:
        return weights

    prices = raw_prices.reindex(columns=universe).astype(float)
    if prices.shape[0] < 2:
        return weights

    prices = prices.ffill()
    last = prices.iloc[-1]

    elig_df = history.get('eligible')
    if isinstance(elig_df, pd.DataFrame):
        elig = elig_df.reindex(columns=universe).ffill().iloc[-1]
        elig = elig.fillna(False).astype(bool)
    else:
        elig = pd.Series(True, index=universe)

    base_valid = last.notna() & np.isfinite(last) & (last > 0.0) & elig
    if not base_valid.any():
        return weights

    max_lookback = 252
    smooth_days = 5
    need = max_lookback + 80
    px = prices.tail(need)

    if px.shape[0] < 25:
        L = min(20, max(1, px.shape[0] - 1))
        if L < 1:
            return weights
        signal = px.iloc[-1] / px.iloc[-1 - L] - 1.0
        signal = signal.replace([np.inf, -np.inf], np.nan)
    else:
        ret = px.pct_change()
        ret = ret.replace([np.inf, -np.inf], np.nan)
        vol = ret.rolling(63, min_periods=20).std(ddof=0)
        vol = vol.where(vol > 1e-8)

        lookbacks = [21, 63, 126, 252]
        lb_weights = [0.15, 0.30, 0.30, 0.25]
        comps = []
        for L, wt in zip(lookbacks, lb_weights):
            if px.shape[0] <= L:
                continue
            mom = px / px.shift(L) - 1.0
            mom = mom.replace([np.inf, -np.inf], np.nan)
            comp = mom / (vol * math.sqrt(float(L)))
            comp = comp.replace([np.inf, -np.inf], np.nan)
            mu = comp.mean(axis=1, skipna=True)
            sd = comp.std(axis=1, ddof=0, skipna=True)
            sd = sd.where(sd > 1e-12)
            z = comp.sub(mu, axis=0).div(sd, axis=0)
            z = z.replace([np.inf, -np.inf], np.nan)
            z_s = z.tail(smooth_days).mean(axis=0, skipna=True)
            comps.append(z_s * wt)

        if comps:
            signal = pd.concat(comps, axis=1).sum(axis=1, min_count=1)
        else:
            L = min(20, max(1, px.shape[0] - 1))
            signal = px.iloc[-1] / px.iloc[-1 - L] - 1.0
            signal = signal.replace([np.inf, -np.inf], np.nan)

    signal = signal.reindex(universe)
    valid = base_valid & signal.notna() & np.isfinite(signal)
    candidates = signal[valid]
    n = int(candidates.shape[0])
    if n < 2:
        return weights

    N = min(50, n // 2)
    if N < 1:
        return weights

    ordered = candidates.sort_values(ascending=False, kind='mergesort')
    longs = ordered.index[:N]
    shorts = ordered.index[-N:]

    per_name = min(0.02, 1.0 / (2.0 * float(N)))
    weights.loc[longs] = per_name
    weights.loc[shorts] = -per_name

    weights = weights.clip(-0.02, 0.02)
    gross = float(weights.abs().sum())
    if gross > 1.0 + 1e-12:
        weights = weights * (1.0 / gross)

    return weights.fillna(0.0).astype(float)
