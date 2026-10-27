import pandas as pd
import numpy as np


def _panel(history, universe, names):
    """Return a wide DataFrame (rows=dates, cols=tickers) for the first
    available field name, reindexed to universe. Handles field-keyed dict."""
    for n in names:
        try:
            obj = history[n]
        except Exception:
            obj = None
        if isinstance(obj, pd.DataFrame) and obj.shape[1] > 0:
            return obj.reindex(columns=universe)
    return None


def _size_side(names, inv_metric, target_gross, cap):
    """Proportional-to-metric sizing with per-name cap and gross target."""
    if len(names) == 0:
        return pd.Series(dtype=float)
    iv = inv_metric.reindex(names)
    iv = iv.replace([np.inf, -np.inf], np.nan).dropna()
    iv = iv[iv > 0]
    if len(iv) == 0:
        return pd.Series(dtype=float)
    w = target_gross * iv / iv.sum()
    for _ in range(6):
        w = w.clip(upper=cap)
        s = w.sum()
        if s <= 0:
            break
        if abs(s - target_gross) < 1e-10:
            break
        if s > target_gross:
            w = w * (target_gross / s)
        else:
            free = w < cap - 1e-12
            if free.sum() == 0:
                break
            deficit = target_gross - s
            base = w[free]
            if base.sum() > 0:
                w[free] = base * (base.sum() + deficit) / base.sum()
    w = w.clip(upper=cap)
    return w


def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=universe)

    # Wide field panels reindexed to universe
    close = _panel(history, universe, ['adjusted_close', 'close', 'adjusted_open', 'open'])
    if close is None or close.empty:
        return zero
    close = close.sort_index()

    vol_panel = _panel(history, universe, ['volume'])
    elig_panel = _panel(history, universe, ['eligible'])

    if elig_panel is not None and not elig_panel.empty:
        last_elig = elig_panel.iloc[-1].reindex(universe).fillna(False).astype(bool)
    else:
        last_elig = pd.Series(True, index=universe)

    if len(close) < 130:
        return zero

    rets = close.pct_change()
    # 6-month momentum, skip most recent 5 days
    mom = close.shift(5) / close.shift(126) - 1.0
    vol = rets.rolling(20).std()

    last_close = close.iloc[-1]
    last_mom = mom.iloc[-1]
    last_vol = vol.iloc[-1]

    if vol_panel is not None and not vol_panel.empty:
        vp = vol_panel.reindex(index=close.index, columns=universe).sort_index()
        liq = (close * vp).rolling(20).median().iloc[-1]
        liq_series = liq.reindex(universe)
        have_liq = liq_series.notna().any()
    else:
        liq_series = pd.Series(np.inf, index=universe)
        have_liq = False

    valid = (
        last_close.notna() & (last_close > 0)
        & last_mom.notna() & np.isfinite(last_mom)
        & last_vol.notna() & (last_vol > 1e-6)
        & last_elig.reindex(universe).fillna(False)
    )
    if have_liq:
        valid = valid & (liq_series.fillna(0) > 3e5)

    candidates = universe[valid]
    if len(candidates) < 20:
        return zero

    m = last_mom.reindex(candidates)
    v = last_vol.reindex(candidates)
    score = m.rank(pct=True) - 0.3 * v.rank(pct=True)

    n_side = min(50, len(candidates) // 2)
    if n_side < 10:
        return zero

    ranked = score.sort_values(ascending=False)
    longs = list(ranked.head(n_side).index)
    shorts = list(ranked.tail(n_side).index)

    cap = 0.02
    target_side = 0.49  # total gross ~0.98 under 1.0 limit, net ~0
    inv_metric = (1.0 / last_vol.clip(lower=1e-6)) ** 0.5

    lw = _size_side(longs, inv_metric, target_side, cap)
    sw = _size_side(shorts, inv_metric, target_side, cap)

    weights = pd.Series(0.0, index=universe)
    if len(lw) > 0:
        weights.loc[lw.index] += lw.values
    if len(sw) > 0:
        weights.loc[sw.index] -= sw.values

    weights = weights.clip(lower=-cap, upper=cap).fillna(0.0)

    gross = weights.abs().sum()
    if gross > 1.0:
        weights = weights * (1.0 / gross)

    return weights
