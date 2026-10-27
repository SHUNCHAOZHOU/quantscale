import numpy as np
import pandas as pd

def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)

    adj = history.get('adjusted_close')
    if adj is None:
        adj = history.get('close')
    if adj is None or not isinstance(adj, pd.DataFrame):
        return pd.Series(0.0, index=universe)

    cols = [c for c in universe if c in adj.columns]
    if len(cols) == 0:
        return pd.Series(0.0, index=universe)

    if not adj.index.is_monotonic_increasing:
        adj = adj.sort_index()

    px = adj[cols].tail(300).copy()
    n = len(px)
    if n < 260:
        return pd.Series(0.0, index=universe)

    # 12-1 return: price at t-21 / price at t-252 - 1
    ret_12_1 = px.shift(21) / px.shift(252) - 1.0
    # 6-1 return: price at t-21 / price at t-126 - 1
    ret_6_1 = px.shift(21) / px.shift(126) - 1.0

    r12 = ret_12_1.iloc[-1]
    r6 = ret_6_1.iloc[-1]

    # Composite past return, emphasis on slower 12-1 signal
    past_ret = 0.6 * r12 + 0.4 * r6
    # Invert: long past losers, short past winners
    signal = -past_ret

    last_price = px.iloc[-1]
    valid = last_price.notna() & (last_price > 0)
    valid = valid & (px.notna().sum() >= 260)
    valid = valid & signal.notna() & r12.notna() & r6.notna()

    if 'eligible' in history:
        edf = history['eligible']
        if isinstance(edf, pd.DataFrame):
            if not edf.index.is_monotonic_increasing:
                edf = edf.sort_index()
            ecols = [c for c in cols if c in edf.columns]
            if ecols:
                e_last = edf[ecols].tail(1)
                if len(e_last) > 0:
                    e = e_last.iloc[-1].reindex(cols).fillna(False).astype(bool)
                    valid = valid & e

    idx_valid = valid[valid].index
    if len(idx_valid) < 20:
        return pd.Series(0.0, index=universe)

    sig = signal.loc[idx_valid]
    ranks = sig.rank(method='average', pct=True)
    w = ranks - 0.5
    w = w - w.mean()

    gross = w.abs().sum()
    if gross > 1e-12:
        w = w / gross

    w = w.clip(-0.02, 0.02)
    w = w - w.mean()
    w = w.clip(-0.02, 0.02)

    gross = w.abs().sum()
    if gross > 1.0 + 1e-12:
        w = w / gross

    weights = pd.Series(0.0, index=universe)
    weights.loc[idx_valid] = w
    return weights.reindex(universe).fillna(0.0)
