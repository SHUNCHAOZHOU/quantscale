import numpy as np
import pandas as pd


def _as_numeric(df):
    try:
        return df.apply(pd.to_numeric, errors='coerce')
    except Exception:
        return df.astype(float)


def _ret(px, start_lag, end_lag):
    if px.shape[0] < start_lag or px.shape[0] < end_lag:
        return pd.Series(np.nan, index=px.columns, dtype=float)
    p0 = px.iloc[-start_lag]
    p1 = px.iloc[-end_lag]
    return p1 / p0 - 1.0


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zeros

    close = history.get('adjusted_close')
    if close is None:
        close = history.get('close')
    if close is None:
        return zeros

    close = close.reindex(columns=universe)
    if close.shape[1] == 0:
        return zeros

    close = _as_numeric(close)
    px = close.ffill()
    n = px.shape[0]
    if n < 22:
        return zeros

    if n >= 252:
        mom = _ret(px, 252, 21)
    elif n >= 126:
        mom = _ret(px, 126, 21)
    elif n >= 63:
        mom = _ret(px, 63, 5)
    else:
        mom = pd.Series(np.nan, index=universe, dtype=float)

    recent = _ret(px, 21, 1)

    if n >= 61:
        vol = px.iloc[-61:].pct_change().std(ddof=0)
    else:
        vol = pd.Series(np.nan, index=universe, dtype=float)

    valid = mom.notna() & recent.notna() & np.isfinite(mom) & np.isfinite(recent)

    eligible = history.get('eligible')
    if eligible is not None:
        try:
            elig = eligible.reindex(columns=universe)
            if elig.shape[0] > 0:
                le = elig.iloc[-1].reindex(universe)
                if le.notna().any():
                    le = le.fillna(False).astype(bool)
                    if bool(le.sum()):
                        valid = valid & le
        except Exception:
            pass

    symbols = valid[valid].index
    if len(symbols) < 4:
        return zeros

    mom_v = mom.reindex(symbols)
    rec_v = recent.reindex(symbols)
    vol_v = vol.reindex(symbols)

    r_mom = mom_v.rank(pct=True, method='average')
    r_rec = rec_v.rank(pct=True, method='average')

    if vol_v.notna().sum() >= 3:
        r_vol = vol_v.rank(pct=True, method='average').fillna(0.5)
    else:
        r_vol = pd.Series(0.5, index=symbols, dtype=float)

    score = 0.60 * r_mom - 0.30 * r_rec - 0.20 * r_vol
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 4:
        return zeros

    k = min(25, len(score) // 2)
    if k < 1:
        return zeros

    ordered = score.sort_values(ascending=False, kind='mergesort')
    longs = ordered.index[:k]
    shorts = ordered.index[-k:]

    w = min(0.02, 1.0 / (2.0 * k))
    weights = pd.Series(0.0, index=universe, dtype=float)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights