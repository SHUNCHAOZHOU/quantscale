import numpy as np
import pandas as pd


def _get_field(history, names, universe):
    for name in names:
        df = history.get(name)
        if isinstance(df, pd.DataFrame):
            return df.reindex(columns=universe).astype(float)
    return None


def generate_weights(history, universe):
    universe = pd.Index(list(universe))
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    close = _get_field(history, ['adjusted_close', 'close'], universe)
    openp = _get_field(history, ['adjusted_open', 'open'], universe)
    volume = _get_field(history, ['volume'], universe)
    if close is None or openp is None or volume is None:
        return zero

    idx = close.index
    if len(idx) < 30:
        return zero

    openp = openp.reindex(index=idx)
    volume = volume.reindex(index=idx)

    elig_df = history.get('eligible')
    if isinstance(elig_df, pd.DataFrame):
        elig = elig_df.reindex(index=idx, columns=universe)
    else:
        elig = None

    n = len(idx)
    last_pos = n - 1
    base_pos = last_pos - (last_pos % 5)

    max_rows = 180
    start = max(0, n - max_rows)
    if base_pos < start + 25:
        base_pos = last_pos
    if base_pos < 25:
        return zero

    sl = slice(start, base_pos + 1)
    c = close.iloc[sl]
    o = openp.iloc[sl]
    v = volume.iloc[sl]
    if len(c) < 25:
        return zero

    if elig is not None:
        e = elig.iloc[sl].fillna(False).astype(bool)
    else:
        e = pd.DataFrame(True, index=c.index, columns=universe)

    prev_close = c.shift(1)
    overnight = o / prev_close - 1.0
    intraday = c / o - 1.0
    overnight = overnight.replace([np.inf, -np.inf], np.nan)
    intraday = intraday.replace([np.inf, -np.inf], np.nan)

    log_o = np.log1p(overnight)
    log_i = np.log1p(intraday)

    valid_px = (c > 0.0) & (o > 0.0) & (prev_close > 0.0)
    base_valid = e & valid_px
    log_o = log_o.where(base_valid)
    log_i = log_i.where(base_valid)

    v_ma = v.rolling(20, min_periods=10).mean()
    v_base = v.rolling(60, min_periods=20).median()
    ratio = v_ma / v_base
    ratio = ratio.replace([np.inf, -np.inf], np.nan).clip(lower=0.2, upper=5.0)
    log_shock = np.log(ratio)
    rs = log_shock.rank(axis=1, pct=True)
    factor = 1.0 + 0.5 * rs.fillna(0.0)

    components = []
    for lookback in (10, 20):
        cum_o = np.expm1(log_o.rolling(lookback, min_periods=lookback).sum())
        cum_i = np.expm1(log_i.rolling(lookback, min_periods=lookback).sum())
        cum_o = cum_o.replace([np.inf, -np.inf], np.nan).clip(lower=-0.9, upper=5.0)
        cum_i = cum_i.replace([np.inf, -np.inf], np.nan).clip(lower=-0.9, upper=5.0)
        rank_o = cum_o.rank(axis=1, pct=True)
        rank_i = cum_i.rank(axis=1, pct=True)
        components.append(rank_o - rank_i * factor)

    score_daily = (components[0] + components[1]) / 2.0
    recent = score_daily.tail(5)
    score = recent.mean(axis=0, skipna=True)
    score_count = recent.count(axis=0)
    score = score.where(score_count >= 3)

    last_elig = e.iloc[-1]
    last_close = c.iloc[-1]
    last_open = o.iloc[-1]
    last_volume = v.iloc[-1]
    dollar_vol = (c * v).tail(20).median(axis=0)

    valid = (
        last_elig &
        last_close.notna() & last_open.notna() & last_volume.notna() &
        (last_close > 0.0) & (last_open > 0.0) & (last_volume > 0.0) &
        dollar_vol.notna() & (dollar_vol > 0.0) &
        score.notna() & np.isfinite(score)
    )

    if int(valid.sum()) < 4:
        return zero

    candidates = score[valid].sort_values(ascending=False, kind='mergesort')
    m = len(candidates)
    n_side = min(30, m // 2)
    if n_side < 1:
        return zero

    longs = candidates.index[:n_side]
    shorts = candidates.index[-n_side:]
    per_name = min(0.02, 0.45 / float(n_side))

    weights = zero.copy()
    weights.loc[longs] = per_name
    weights.loc[shorts] = -per_name
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    weights = weights.clip(lower=-0.02, upper=0.02)

    gross = float(weights.abs().sum())
    if gross > 1.0:
        weights = weights / gross

    return weights.astype(float)