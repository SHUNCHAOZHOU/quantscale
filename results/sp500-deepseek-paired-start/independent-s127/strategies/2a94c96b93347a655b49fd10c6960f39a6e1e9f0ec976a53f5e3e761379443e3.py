import pandas as pd
import numpy as np

def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)

    adj_close = {}
    volume = {}
    eligible_last = {}
    for t in universe:
        df = history.get(t)
        if df is None or len(df) == 0:
            eligible_last[t] = False
            continue
        if 'adjusted_close' in df.columns:
            adj_close[t] = df['adjusted_close']
        elif 'close' in df.columns:
            adj_close[t] = df['close']
        else:
            eligible_last[t] = False
            continue
        if 'volume' in df.columns:
            volume[t] = df['volume']
        if 'eligible' in df.columns and len(df) > 0:
            eligible_last[t] = bool(df['eligible'].iloc[-1])
        else:
            eligible_last[t] = True

    if not adj_close:
        return pd.Series(0.0, index=universe)

    close_df = pd.DataFrame(adj_close).reindex(columns=universe).sort_index()
    if len(close_df) < 150:
        return pd.Series(0.0, index=universe)

    rets = close_df.pct_change()
    mom = close_df.shift(5) / close_df.shift(126) - 1.0
    vol = rets.rolling(20).std()

    if volume:
        vol_df = pd.DataFrame(volume).reindex(columns=universe).sort_index()
        dollar_vol = close_df * vol_df
        liq = dollar_vol.rolling(20).median()
    else:
        liq = pd.DataFrame(np.nan, index=close_df.index, columns=universe)

    last_mom = mom.iloc[-1]
    last_vol = vol.iloc[-1]
    last_liq = liq.iloc[-1] if len(liq) > 0 else pd.Series(np.nan, index=universe)
    last_close = close_df.iloc[-1]
    elig = pd.Series(eligible_last).reindex(universe).fillna(False).astype(bool)

    valid = (
        last_close.notna()
        & last_mom.notna()
        & last_vol.notna()
        & (last_vol > 0)
        & (last_vol < 0.15)
        & (last_liq > 1e6)
        & elig
    )
    candidates = universe[valid]
    if len(candidates) < 20:
        return pd.Series(0.0, index=universe)

    m = last_mom[candidates]
    v = last_vol[candidates]
    mom_rank = m.rank(pct=True)
    vol_rank = v.rank(pct=True)
    score = mom_rank - 0.3 * vol_rank

    n_side = min(50, len(candidates) // 2)
    if n_side < 10:
        return pd.Series(0.0, index=universe)

    ranked = score.sort_values(ascending=False)
    longs = ranked.head(n_side).index
    shorts = ranked.tail(n_side).index

    w = min(0.5 / n_side, 0.02)
    weights = pd.Series(0.0, index=universe)
    weights[longs] = w
    weights[shorts] = -w
    return weights.fillna(0.0)
