import numpy as np
import pandas as pd


def _zscore(s):
    s = s.replace([np.inf, -np.inf], np.nan)
    med = s.median()
    mad = (s - med).abs().median()
    if pd.notna(mad) and mad > 1e-12:
        z = (s - med) / (1.4826 * mad)
    else:
        std = s.std()
        if pd.notna(std) and std > 1e-12:
            z = (s - s.mean()) / std
        else:
            z = s * 0.0
    return z.clip(-5.0, 5.0)


def generate_weights(history, universe):
    u = pd.Index(universe)
    if len(u) == 0:
        return pd.Series(dtype=float)
    zeros = pd.Series(0.0, index=u)
    if not isinstance(history, dict):
        return zeros

    px_df = history.get('adjusted_close')
    if px_df is None or px_df.empty:
        px_df = history.get('close')
    if px_df is None or px_df.empty:
        return zeros

    cols = [t for t in u if t in px_df.columns]
    if len(cols) < 2:
        return zeros

    px = px_df.loc[:, cols].astype(float)
    if len(px) < 30:
        return zeros

    rets = px / px.shift(1) - 1.0
    mom = px.shift(5) / px.shift(126) - 1.0
    rev = -(px / px.shift(5) - 1.0)
    vol = rets.rolling(60, min_periods=20).std()

    m = mom.iloc[-1]
    r = rev.iloc[-1]
    v = vol.iloc[-1]
    valid = m.notna() & r.notna() & v.notna() & np.isfinite(v) & (v > 0.0)

    if valid.sum() < 20:
        rev20 = -(px / px.shift(20) - 1.0)
        r = rev20.iloc[-1]
        v = vol.iloc[-1]
        valid = r.notna() & v.notna() & np.isfinite(v) & (v > 0.0)
        if valid.sum() < 10:
            return zeros
        score = _zscore(r.where(valid)) - 0.25 * _zscore(v.where(valid))
    else:
        score = (
            _zscore(m.where(valid))
            + 0.5 * _zscore(r.where(valid))
            - 0.25 * _zscore(v.where(valid))
        )

    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(score)
    if n < 10:
        return zeros

    k = min(25, n // 2)
    if k < 1:
        return zeros

    ranked = score.sort_values()
    short_names = ranked.index[:k]
    long_names = ranked.index[-k:]

    w = pd.Series(0.0, index=u)
    weight = 0.02
    w.loc[long_names] = weight
    w.loc[short_names] = -weight
    return w
