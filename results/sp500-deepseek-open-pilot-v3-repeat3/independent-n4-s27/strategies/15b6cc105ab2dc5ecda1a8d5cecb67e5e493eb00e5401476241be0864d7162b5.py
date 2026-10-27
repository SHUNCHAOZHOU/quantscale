import numpy as np
import pandas as pd


def _pct_rank(s):
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=s.index)
    return s.rank(method='average', pct=True)


def generate_weights(history, universe):
    u = pd.Index(universe)
    zero = pd.Series(0.0, index=u, dtype=float)
    try:
        adj = history.get('adjusted_close')
        if adj is None:
            adj = history.get('close')
        if adj is None or len(adj) < 6:
            return zero

        adj = adj.reindex(columns=u).astype(float)
        last = adj.iloc[-1]
        valid = last.notna() & np.isfinite(last)
        valid &= adj.notna().sum() >= 20

        elig = history.get('eligible')
        if elig is not None:
            elig = elig.reindex(columns=u)
            if len(elig) > 0:
                e = elig.iloc[-1]
                try:
                    if e.dtype == bool:
                        valid &= e.fillna(False).astype(bool)
                    else:
                        valid &= (e.fillna(0).astype(float) > 0)
                except Exception:
                    pass

        n = len(adj)
        if n >= 260:
            p_end = adj.iloc[-22]
            mom12 = p_end / adj.iloc[-253] - 1.0
            mom6 = p_end / adj.iloc[-127] - 1.0
            rev1 = adj.iloc[-1] / adj.iloc[-22] - 1.0
        elif n >= 130:
            p_end = adj.iloc[-22] if n > 22 else adj.iloc[-1]
            mom6 = p_end / adj.iloc[-127] - 1.0
            mom12 = mom6
            rev1 = adj.iloc[-1] / adj.iloc[-22] - 1.0 if n > 22 else pd.Series(0.0, index=u)
        else:
            mom12 = adj.iloc[-1] / adj.iloc[0] - 1.0
            mom6 = mom12
            rev1 = adj.iloc[-1] / adj.iloc[-6] - 1.0 if n >= 6 else pd.Series(0.0, index=u)

        rets = adj / adj.shift(1) - 1.0
        rets = rets.replace([np.inf, -np.inf], np.nan)
        w = min(120, max(20, n // 4))
        vol = rets.iloc[-w:].std()

        vol_df = history.get('volume')
        if vol_df is not None:
            vol_df = vol_df.reindex(columns=u).astype(float)
            dv = adj * vol_df
            liq = dv.iloc[-min(20, n):].median()
        else:
            liq = pd.Series(0.0, index=u)

        def prep(x):
            x = x.reindex(u).astype(float)
            x = x.where(valid, np.nan)
            x = x.replace([np.inf, -np.inf], np.nan)
            if x.notna().sum() < 2:
                return pd.Series(0.5, index=u)
            med = x.median()
            x = x.fillna(med)
            return _pct_rank(x)

        r12 = prep(mom12)
        r6 = prep(mom6)
        rrev = prep(rev1)
        rvol = prep(vol)
        rliq = prep(liq)

        score = r12 + 0.50 * r6 - 0.05 * rrev - 0.30 * rvol + 0.05 * rliq
        score = score.replace([np.inf, -np.inf], np.nan).fillna(-1e9)
        cand = score.where(valid, np.nan).dropna()
        if len(cand) < 2:
            return zero

        k = min(25, len(cand) // 2)
        if k < 1:
            return zero

        ordered = cand.sort_values(ascending=False)
        long_ids = ordered.index[:k]
        short_ids = ordered.index[-k:]

        out = zero.copy()
        out.loc[long_ids] = 0.02
        out.loc[short_ids] = -0.02
        out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        out = out.clip(-0.02, 0.02)
        return out
    except Exception:
        return zero
