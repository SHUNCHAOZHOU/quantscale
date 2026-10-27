import numpy as np
import pandas as pd


def _frame(history, key):
    obj = history.get(key, None)
    if isinstance(obj, pd.DataFrame) and not obj.empty:
        return obj
    return None


def _rank(s):
    return s.rank(method='average', pct=True) - 0.5


def generate_weights(history, universe):
    universe = pd.Index(universe)
    weights = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return weights

    close = _frame(history, 'adjusted_close')
    raw_close = _frame(history, 'close')
    if close is None:
        close = raw_close
    elif raw_close is not None:
        close = close.copy()
        raw_close = raw_close.reindex(index=close.index, columns=close.columns)
        close = close.where(close.notna(), raw_close)
    if close is None or close.empty:
        return weights

    close = close.reindex(columns=universe)
    if close.empty:
        return weights
    close = close.apply(pd.to_numeric, errors='coerce')
    L = len(close)
    if L < 25:
        return weights

    last = close.iloc[-1]
    if L >= 22:
        r21 = last / close.iloc[-22] - 1.0
    else:
        r21 = last / close.iloc[0] - 1.0
    if L >= 6:
        r5 = last / close.iloc[-6] - 1.0
    else:
        r5 = last / close.iloc[0] - 1.0
    if L >= 64:
        r63 = last / close.iloc[-64] - 1.0
    else:
        r63 = last / close.iloc[0] - 1.0

    ret = close.pct_change()
    vol_window = min(63, max(20, L - 1))
    vol = ret.tail(vol_window).std().replace([np.inf, -np.inf], np.nan)

    volume = _frame(history, 'volume')
    liq_ok = pd.Series(True, index=universe, dtype=bool)
    liq_rank = pd.Series(0.0, index=universe, dtype=float)
    if volume is not None:
        v = volume.reindex(columns=universe)
        v = v.apply(pd.to_numeric, errors='coerce')
        if not v.empty:
            v_tail = v.tail(min(21, len(v)))
            v_med = v_tail.median()
            liq_ok = (v_med.notna() & (v_med > 0)).reindex(universe).fillna(False).astype(bool)
            dv = (close * v).tail(min(21, L)).median()
            liq_rank = _rank(dv.replace([np.inf, -np.inf], np.nan)).reindex(universe).fillna(0.0)

    eligible = pd.Series(True, index=universe, dtype=bool)
    elig = _frame(history, 'eligible')
    if elig is not None:
        e = elig.reindex(columns=universe)
        if not e.empty:
            e_last = e.iloc[-1]
            try:
                eligible = e_last.fillna(False).astype(bool)
            except Exception:
                eligible = e_last.notna()
    eligible = eligible.reindex(universe).fillna(False).astype(bool)

    signal = (
        -0.60 * _rank(r21)
        - 0.20 * _rank(r5)
        - 0.10 * _rank(r63)
        - 0.10 * _rank(vol)
        + 0.05 * liq_rank
    )
    signal = signal.replace([np.inf, -np.inf], np.nan)

    valid = eligible & liq_ok & last.notna() & (last > 0) & signal.notna()
    valid = valid.fillna(False).astype(bool)
    ranked = signal[valid].sort_values(ascending=False)
    n = len(ranked)
    if n < 4:
        return weights

    n_side = min(50, n // 2)
    if n_side < 2:
        return weights
    longs = ranked.index[:n_side]
    shorts = ranked.index[-n_side:]
    side_gross = min(0.45, n_side * 0.02)
    w = side_gross / float(n_side)
    if w > 0.02:
        w = 0.02
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    return weights.reindex(universe).fillna(0.0).astype(float)