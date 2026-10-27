import numpy as np
import pandas as pd


def _safe_frame(history, key):
    obj = history.get(key, None)
    if isinstance(obj, pd.DataFrame) and not obj.empty:
        return obj
    return None


def _last_row_shift(df, n):
    if n <= 0:
        return df.iloc[-1]
    if n >= len(df):
        return pd.Series(np.nan, index=df.columns, dtype=float)
    return df.iloc[-1 - n]


def _cs_rank(s):
    return s.rank(method='average', pct=True) - 0.5


def generate_weights(history, universe):
    universe = pd.Index(universe)
    weights = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return weights

    close = _safe_frame(history, 'adjusted_close')
    close_raw = _safe_frame(history, 'close')
    if close is None:
        close = close_raw
    else:
        close = close.copy()
        if close_raw is not None:
            close_raw = close_raw.reindex(index=close.index, columns=close.columns)
            close = close.where(close.notna(), close_raw)
    if close is None or close.empty:
        return weights

    close = close.reindex(columns=universe)
    if close.empty:
        return weights
    close = close.apply(pd.to_numeric, errors='coerce')
    L = len(close)
    if L < 2:
        return weights

    if L >= 253:
        skip, lb12, lb6 = 21, 252, 126
    elif L >= 127:
        skip, lb12, lb6 = 21, 126, 63
    else:
        if L >= 4:
            skip = max(1, min(21, L // 5))
            lb12 = max(skip + 1, min(L - 1, L // 2))
            lb6 = max(skip + 1, min(L - 1, L // 3))
        else:
            skip, lb12, lb6 = 1, 1, 1

    last = close.iloc[-1]
    p_skip = _last_row_shift(close, skip)
    p_12 = _last_row_shift(close, lb12)
    p_6 = _last_row_shift(close, lb6)

    mom12 = (p_skip / p_12 - 1.0).replace([np.inf, -np.inf], np.nan)
    mom6 = (p_skip / p_6 - 1.0).replace([np.inf, -np.inf], np.nan)
    rev = (last / p_skip - 1.0).replace([np.inf, -np.inf], np.nan)

    ret = close.pct_change()
    vol_window = min(126, max(20, L // 4))
    vol_last = ret.tail(vol_window).std().replace([np.inf, -np.inf], np.nan)

    r12 = _cs_rank(mom12)
    r6 = _cs_rank(mom6)
    rrev = _cs_rank(rev)
    rvol = _cs_rank(vol_last)

    signal = 0.55 * r12 + 0.30 * r6 - 0.10 * rrev - 0.05 * rvol
    signal = signal.replace([np.inf, -np.inf], np.nan)

    eligible = pd.Series(True, index=universe, dtype=bool)
    elig_df = _safe_frame(history, 'eligible')
    if elig_df is not None:
        e = elig_df.reindex(columns=universe)
        if not e.empty:
            e_last = e.iloc[-1]
            try:
                eligible = e_last.fillna(False).astype(bool)
            except Exception:
                eligible = e_last.notna()
    eligible = eligible.reindex(universe).fillna(False).astype(bool)

    liquid = pd.Series(True, index=universe, dtype=bool)
    vol_df = _safe_frame(history, 'volume')
    if vol_df is not None:
        v = vol_df.reindex(columns=universe)
        if not v.empty:
            v_med = v.tail(min(20, len(v))).median()
            liquid = v_med.notna() & (v_med > 0)
    liquid = liquid.reindex(universe).fillna(False).astype(bool)

    valid = (
        eligible
        & liquid
        & last.notna()
        & (last > 0)
        & signal.notna()
    )
    valid = valid.fillna(False).astype(bool)

    ranked = signal[valid].sort_values(ascending=False)
    n = len(ranked)
    if n < 2:
        return weights

    n_side = min(50, n // 2)
    if n_side < 1:
        return weights

    longs = ranked.index[:n_side]
    shorts = ranked.index[-n_side:]
    w = min(0.02, 0.5 / n_side, 1.0 / (2.0 * n_side))
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    return weights