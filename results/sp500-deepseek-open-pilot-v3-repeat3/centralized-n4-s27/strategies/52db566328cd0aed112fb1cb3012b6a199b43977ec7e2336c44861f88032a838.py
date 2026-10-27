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
        px = _get_df(history, ['adjusted_close', 'close'])
        if px is None:
            return zeros
        px = px.reindex(columns=universe)
        if px.shape[0] < 60:
            return zeros
        px = px.replace([np.inf, -np.inf], np.nan)
        T = px.shape[0]
        P = 21
        r = T - 1 - ((T - 1) % P)
        if r < 60:
            return zeros
        L = 252
        S = 21
        start = max(0, r - L)
        end = max(0, r - S)
        if end <= start:
            return zeros
        p_start = px.iloc[start]
        p_end = px.iloc[end]
        mom = (p_end / p_start - 1.0).replace([np.inf, -np.inf], np.nan)
        elig = pd.Series(True, index=universe)
        edf = _get_df(history, ['eligible'])
        if edf is not None:
            ed = edf.reindex(index=px.index, columns=universe)
            if len(ed) > 0:
                le = ed.iloc[-1]
                if le.notna().any():
                    elig = le.fillna(False).astype(bool)
        vol_ok = pd.Series(True, index=universe)
        vol = _get_df(history, ['volume'])
        if vol is not None:
            v = vol.reindex(index=px.index, columns=universe)
            if len(v) > 0:
                rv = min(r, len(v) - 1)
                vv = v.iloc[max(0, rv - 19):rv + 1].replace([np.inf, -np.inf], np.nan)
                vmed = vv.median(axis=0, skipna=True)
                vol_ok = vmed.notna() & (vmed > 0)
                if int(vol_ok.sum()) >= 50:
                    vals = vmed[vol_ok].astype(float).values
                    if vals.size >= 20:
                        thr = float(np.nanpercentile(vals, 20.0))
                        vol_ok = vol_ok & (vmed >= thr)
        mask = elig & np.isfinite(p_start) & np.isfinite(p_end) & (p_start > 0) & (p_end > 0) & np.isfinite(mom) & vol_ok
        if int(mask.sum()) < 10:
            return zeros
        idx = universe[mask.values]
        sig = mom.loc[idx].astype(float)
        sig = sig.replace([np.inf, -np.inf], np.nan).dropna()
        if len(sig) < 4:
            return zeros
        n_side = min(30, len(sig) // 2)
        if n_side < 2:
            return zeros
        order = np.argsort(-sig.values, kind='mergesort')
        longs = sig.index[order[:n_side]]
        shorts = sig.index[order[-n_side:]]
        nL = len(longs)
        nS = len(shorts)
        if nL == 0 or nS == 0:
            return zeros
        target_net = 0.009
        gL = min(0.5 + target_net / 2.0, 0.02 * nL)
        gS = min(0.5 - target_net / 2.0, 0.02 * nS)
        if gL - gS > target_net:
            gL = gS + target_net
        elif gL - gS < target_net:
            if gL - target_net >= 0:
                gS = gL - target_net
            else:
                gL = gS + target_net
                if gL > 0.02 * nL:
                    gL = 0.02 * nL
                    gS = max(0.0, gL - target_net)
        gross = gL + gS
        if gross > 1.0:
            scale = 1.0 / gross
            gL *= scale
            gS *= scale
        wL = min(0.02, gL / nL)
        wS = min(0.02, gS / nS)
        out = pd.Series(0.0, index=universe, dtype=float)
        out.loc[longs] = wL
        out.loc[shorts] = -wS
        out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        out = out.clip(-0.02, 0.02)
        return out
    except Exception:
        return zeros
