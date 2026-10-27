import numpy as np
import pandas as pd


def _get_field(history, name, fallback, universe):
    df = history.get(name)
    if df is None and fallback is not None:
        df = history.get(fallback)
    if df is None or not isinstance(df, pd.DataFrame):
        return None
    return df.reindex(columns=universe).astype(float)


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    close = _get_field(history, 'adjusted_close', 'close', universe)
    openp = _get_field(history, 'adjusted_open', 'open', universe)
    volume = _get_field(history, 'volume', None, universe)
    if close is None or openp is None or volume is None:
        return zero

    n = min(len(close), 60)
    if n < 10:
        return zero
    close = close.iloc[-n:]
    openp = openp.iloc[-n:]
    volume = volume.iloc[-n:]

    prev_close = close.shift(1)
    overnight = openp / prev_close - 1.0
    intraday = close / openp - 1.0

    if len(close) < 6:
        return zero

    o5 = (1.0 + overnight.iloc[-5:]).prod(min_count=5) - 1.0
    i5 = (1.0 + intraday.iloc[-5:]).prod(min_count=5) - 1.0

    recent_vol = volume.iloc[-5:].mean()
    if len(volume) >= 25:
        base = volume.iloc[-25:-5].median()
    elif len(volume) >= 10:
        base = volume.iloc[:-5].median()
    else:
        base = volume.median()
    base = base.where(base > 0, np.nan)
    ratio = (recent_vol / base).replace([np.inf, -np.inf], np.nan)
    ratio = ratio.clip(lower=1e-2, upper=1e2)
    log_shock = np.log(ratio)

    elig_df = history.get('eligible')
    if isinstance(elig_df, pd.DataFrame):
        elig = elig_df.reindex(columns=universe).iloc[-1]
        elig = elig.fillna(False).astype(bool)
    else:
        elig = pd.Series(True, index=universe, dtype=bool)

    last_close = close.iloc[-1]
    last_open = openp.iloc[-1]
    last_vol = volume.iloc[-1]

    o5 = o5.replace([np.inf, -np.inf], np.nan)
    i5 = i5.replace([np.inf, -np.inf], np.nan)

    valid = (
        elig &
        last_close.notna() & last_open.notna() & last_vol.notna() &
        (last_close > 0) & (last_open > 0) & (last_vol > 0) &
        o5.notna() & i5.notna() & log_shock.notna()
    )

    if valid.sum() < 2:
        return zero

    ro = o5.rank(pct=True) - 0.5
    ri = i5.rank(pct=True) - 0.5
    rs = log_shock.rank(pct=True)
    score = ro - ri * (1.0 + rs)
    score = score.replace([np.inf, -np.inf], np.nan)
    valid = valid & score.notna()

    cand = score[valid].sort_values(ascending=False)
    m = len(cand)
    if m < 2:
        return zero

    N = min(50, m // 2)
    if N < 1:
        return zero

    longs = cand.index[:N]
    shorts = cand.index[-N:]
    weight = min(0.5 / N, 0.02)

    w = zero.copy()
    w.loc[longs] = weight
    w.loc[shorts] = -weight
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    w = w.clip(lower=-0.02, upper=0.02)

    gross = w.abs().sum()
    if gross > 1.0:
        w = w / gross

    return w.astype(float)