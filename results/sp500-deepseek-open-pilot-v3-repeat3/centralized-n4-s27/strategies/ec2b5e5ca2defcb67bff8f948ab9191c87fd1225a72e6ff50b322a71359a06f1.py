import numpy as np
import pandas as pd


def _get_df(history, names):
    for name in names:
        df = history.get(name, None)
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            return df
    return None


def generate_weights(history, universe):
    universe = pd.Index(list(universe))
    zeros = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zeros
    try:
        px = _get_df(history, ['adjusted_open', 'open'])
        if px is None:
            return zeros
        px = px.reindex(columns=universe)
        if px.shape[0] < 2:
            return zeros
        px = px.replace([np.inf, -np.inf], np.nan)
        last = px.iloc[-1].astype(float)
        W = 5
        L = 60
        w = min(W, px.shape[0] - 1)
        l = min(L, px.shape[0] - 1)
        if w < 1:
            return zeros
        p_w = px.iloc[-1 - w]
        p_l = px.iloc[-1 - l]
        cum_w = (last / p_w - 1.0).replace([np.inf, -np.inf], np.nan)
        cum_l = (last / p_l - 1.0).replace([np.inf, -np.inf], np.nan)
        elig = pd.Series(True, index=universe)
        edf = _get_df(history, ['eligible'])
        if edf is not None:
            ed = edf.reindex(columns=universe)
            if len(ed) > 0:
                le = ed.iloc[-1]
                if le.notna().any():
                    elig = le.fillna(False).astype(bool)
        mask = elig & np.isfinite(last) & (last > 0) & np.isfinite(cum_w) & np.isfinite(cum_l)
        vol = _get_df(history, ['volume'])
        if vol is not None:
            v = vol.reindex(columns=universe).tail(20).replace([np.inf, -np.inf], np.nan)
            vol_med = v.median(axis=0, skipna=True)
            has_vol = vol_med.notna() & (vol_med > 0)
            base_count = int(mask.sum())
            if base_count >= 50 and int((mask & has_vol).sum()) >= 40:
                vals = vol_med[mask & has_vol].astype(float).values
                if vals.size >= 20:
                    thr = float(np.nanpercentile(vals, 25.0))
                    mask = mask & has_vol & (vol_med >= thr)
        if int(mask.sum()) < 2:
            return zeros
        idx = universe[mask.values]
        y = cum_w.loc[idx].astype(float).values
        x = cum_l.loc[idx].astype(float).values
        y = y - np.nanmean(y)
        x = x - np.nanmean(x)
        var_x = float(np.nanmean(x * x))
        if var_x > 1e-12:
            beta = float(np.nanmean(x * y) / var_x)
            if not np.isfinite(beta):
                beta = 0.0
            beta = float(np.clip(beta, -0.5, 0.5))
        else:
            beta = 0.0
        resid = y - beta * x
        signal = pd.Series(-resid, index=idx, dtype=float)
        signal = signal.replace([np.inf, -np.inf], np.nan).dropna()
        n = min(25, len(signal) // 2)
        if n < 1:
            return zeros
        vals = signal.values
        order = np.argsort(-vals, kind='mergesort')
        longs = signal.index[order[:n]]
        shorts = signal.index[order[-n:]]
        wgt = 0.02
        out = pd.Series(0.0, index=universe, dtype=float)
        out.loc[longs] = wgt
        out.loc[shorts] = -wgt
        out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        out = out.clip(-0.02, 0.02)
        gross = float(out.abs().sum())
        if gross > 1.0 + 1e-12:
            out = out * (1.0 / gross)
        return out
    except Exception:
        return zeros
