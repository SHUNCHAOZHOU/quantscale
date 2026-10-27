import pandas as pd
import numpy as np

def _field(history, name, universe):
    universe = pd.Index(universe)
    if isinstance(history, dict):
        obj = history.get(name)
        if isinstance(obj, pd.DataFrame):
            if len(universe) == 0 or len(pd.Index(obj.columns).intersection(universe)) > 0:
                return obj.reindex(columns=universe).sort_index()
    cols = {}
    if isinstance(history, dict):
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
    zeros = pd.Series(0.0, index=universe)
    if len(universe) == 0:
        return zeros
    close = _field(history, 'close', universe)
    adj = _field(history, 'adjusted_close', universe)
    if adj is None:
        adj = close
    if adj is None or adj.shape[0] < 140:
        return zeros
    C = adj.copy()
    rets = C / C.shift(1) - 1.0
    rets = rets.where(np.isfinite(rets))
    mom6 = C.shift(5) / C.shift(126) - 1.0
    mom12 = C.shift(21) / C.shift(252) - 1.0
    mom6 = mom6.where(np.isfinite(mom6))
    mom12 = mom12.where(np.isfinite(mom12))
    vol = rets.rolling(20, min_periods=15).std()
    last_close = C.iloc[-1]
    last_m6 = mom6.iloc[-1]
    last_m12 = mom12.iloc[-1]
    last_vol = vol.iloc[-1]
    volume = _field(history, 'volume', universe)
    if volume is not None:
        price_liq = close if close is not None else C
        liq = (price_liq * volume).rolling(20, min_periods=10).median()
        last_liq = liq.iloc[-1]
    else:
        last_liq = pd.Series(np.inf, index=universe)
    elig_df = _field(history, 'eligible', universe)
    if elig_df is not None and len(elig_df) > 0:
        elig = elig_df.iloc[-1].reindex(universe).fillna(False).astype(bool)
    else:
        elig = last_close.notna()
    valid = (last_close.notna() & last_m6.notna() & last_vol.notna() & (last_vol > 0) & (last_vol < 0.10) & (last_liq > 1e6) & elig)
    candidates = universe[valid]
    if len(candidates) < 20:
        return zeros
    r6 = last_m6[candidates].rank(pct=True)
    r12 = last_m12.reindex(candidates).rank(pct=True)
    mom_score = 0.5 * r6 + 0.5 * r12
    mom_score = mom_score.fillna(r6)
    vol_rank = last_vol[candidates].rank(pct=True)
    score = mom_score - 0.3 * vol_rank
    n_side = min(50, len(candidates) // 2)
    if n_side < 10:
        return zeros
    ranked = score.sort_values(ascending=False)
    longs = ranked.head(n_side).index
    shorts = ranked.tail(n_side).index
    w = min(0.5 / n_side, 0.02)
    weights = pd.Series(0.0, index=universe)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    return weights.fillna(0.0)
