import pandas as pd
import numpy as np


def _field(history, name, universe):
    universe = pd.Index(universe)
    if isinstance(history, dict):
        obj = history.get(name)
        if isinstance(obj, pd.DataFrame):
            return obj.reindex(columns=universe).sort_index()
        cols = {}
        for t in universe:
            df = history.get(t)
            if isinstance(df, pd.DataFrame) and name in df.columns:
                s = df[name]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                cols[t] = s
        if cols:
            return pd.DataFrame(cols).reindex(columns=universe).sort_index()
    return None


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zeros

    close = _field(history, 'close', universe)
    adj = _field(history, 'adjusted_close', universe)
    if adj is None:
        adj = close
    if adj is None or len(adj) < 30:
        return zeros

    C = adj.copy().where(np.isfinite(adj))
    last_close = C.iloc[-1]

    # Short-horizon reversal signal: prior 5-session return.
    r5 = C / C.shift(5) - 1.0
    r5 = r5.where(np.isfinite(r5))
    last_r5 = r5.iloc[-1]

    rets = C.pct_change()
    vol20 = rets.rolling(20, min_periods=10).std()
    last_vol = vol20.iloc[-1]

    volume = _field(history, 'volume', universe)
    if volume is not None:
        price = close if close is not None else C
        dollar_volume = price * volume
        liq = dollar_volume.rolling(20, min_periods=10).median()
        last_liq = liq.iloc[-1]
    else:
        last_liq = pd.Series(np.inf, index=universe)

    elig_df = _field(history, 'eligible', universe)
    if elig_df is not None and len(elig_df) > 0:
        elig = elig_df.iloc[-1].reindex(universe).fillna(False).astype(bool)
    else:
        elig = last_close.notna()

    valid = (
        last_close.notna()
        & last_r5.notna()
        & last_vol.notna()
        & (last_vol > 0)
        & (last_liq > 1e6)
        & elig
    )
    candidates = universe[valid]
    if len(candidates) < 20:
        return zeros

    # Long prior losers, short prior winners.
    score = (-last_r5[candidates]).rank(pct=True)
    n_side = min(25, len(candidates) // 2)
    if n_side < 5:
        return zeros

    ranked = score.sort_values(ascending=False)
    longs = ranked.head(n_side).index
    shorts = ranked.tail(n_side).index

    w = min(0.02, 0.5 / n_side)
    weights = pd.Series(0.0, index=universe, dtype=float)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    return weights.fillna(0.0)
