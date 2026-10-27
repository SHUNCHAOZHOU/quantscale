import numpy as np
import pandas as pd

def generate_weights(history, universe):
    if universe is None or len(universe) == 0:
        return pd.Series(dtype=float)
    ac = history.get('adjusted_close')
    if ac is None or ac.empty:
        return pd.Series(0.0, index=universe)
    n = len(ac)
    if n < 2:
        return pd.Series(0.0, index=universe)

    # Momentum signal: 12-1 if enough history, else shorter windows
    if n >= 253:
        px_skip = ac.iloc[-22]
        px_past = ac.iloc[-253]
        mom = px_skip / px_past - 1.0
    elif n >= 127:
        mom = ac.iloc[-1] / ac.iloc[-127] - 1.0
    elif n >= 64:
        mom = ac.iloc[-1] / ac.iloc[-64] - 1.0
    elif n >= 22:
        mom = ac.iloc[-1] / ac.iloc[-22] - 1.0
    else:
        mom = ac.iloc[-1] / ac.iloc[0] - 1.0
    mom = mom.replace([np.inf, -np.inf], np.nan)

    # Volume filter
    vol = history.get('volume')
    if vol is not None and not vol.empty:
        m = min(20, len(vol))
        avg_vol = vol.iloc[-m:].mean()
    else:
        avg_vol = pd.Series(np.inf, index=ac.columns)

    # Eligibility filter
    if 'eligible' in history:
        e = history['eligible']
        if not e.empty:
            elig = e.iloc[-1]
        else:
            elig = pd.Series(True, index=ac.columns)
    else:
        elig = pd.Series(True, index=ac.columns)

    df = pd.DataFrame({
        'mom': mom,
        'avg_vol': avg_vol,
        'px': ac.iloc[-1],
        'eligible': elig
    })
    df = df.reindex(universe)

    valid = df['mom'].notna() & np.isfinite(df['mom'])
    valid &= (df['px'] > 1.0)
    valid &= (df['avg_vol'] > 100000)
    valid &= df['eligible'].fillna(False).astype(bool)

    if valid.sum() < 20:
        valid = df['mom'].notna() & np.isfinite(df['mom']) & (df['px'] > 0.5)
    if valid.sum() < 20:
        return pd.Series(0.0, index=universe)

    sig = df.loc[valid, 'mom'].sort_values(ascending=False)
    n_select = min(50, len(sig) // 2)
    if n_select < 1:
        return pd.Series(0.0, index=universe)

    weight = min(0.02, 0.5 / n_select)
    longs = sig.iloc[:n_select].index
    shorts = sig.iloc[-n_select:].index

    w = pd.Series(0.0, index=universe)
    w.loc[longs] = weight
    w.loc[shorts] = -weight
    return w