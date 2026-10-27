import numpy as np
import pandas as pd


def generate_weights(history, universe):
    u = pd.Index(universe)
    zero = pd.Series(0.0, index=u, dtype=float)
    try:
        adj_full = history.get('adjusted_close')
        if adj_full is None:
            adj_full = history.get('close')
        if adj_full is None or len(adj_full) < 30:
            return zero

        adj = adj_full.iloc[-200:].reindex(columns=u).astype(float)
        idx = adj.index
        if len(idx) < 30:
            return zero

        current = idx[-1]
        cur_y, cur_m = current.year, current.month
        mask = (idx.year < cur_y) | ((idx.year == cur_y) & (idx.month < cur_m))
        if not mask.any():
            return zero

        sig_date = idx[mask][-1]
        sig_loc = int(idx.get_indexer([sig_date])[0])
        if sig_loc < 20:
            return zero

        adj_s = adj.iloc[:sig_loc + 1]
        if len(adj_s) < 25:
            return zero

        last = adj_s.iloc[-1]
        valid = last.notna() & np.isfinite(last)
        valid &= adj_s.notna().sum() >= 20

        elig_full = history.get('eligible')
        if elig_full is not None:
            try:
                elig = elig_full.reindex(index=adj.index, columns=u)
                e = elig.iloc[sig_loc]
                if e.dtype == bool:
                    valid &= e.fillna(False).astype(bool)
                else:
                    valid &= (e.fillna(0).astype(float) > 0)
            except Exception:
                pass

        if valid.sum() < 6:
            return zero

        n = len(adj_s)
        look21 = min(21, n - 1)
        look5 = min(5, n - 1)

        ret21 = adj_s.iloc[-1] / adj_s.iloc[-1 - look21] - 1.0
        ret5 = adj_s.iloc[-1] / adj_s.iloc[-1 - look5] - 1.0

        rets = adj_s / adj_s.shift(1) - 1.0
        rets = rets.replace([np.inf, -np.inf], np.nan)
        wvol = min(60, max(10, n // 4))
        vol = rets.iloc[-wvol:].std()

        vol_full = history.get('volume')
        if vol_full is not None:
            try:
                vdf = vol_full.reindex(index=adj.index, columns=u).astype(float)
                dv = adj_s * vdf.iloc[:sig_loc + 1]
                liq = dv.iloc[-min(20, n):].median()
            except Exception:
                liq = pd.Series(0.0, index=u)
        else:
            liq = pd.Series(0.0, index=u)

        def prep(x):
            x = x.reindex(u).astype(float)
            x = x.where(valid, np.nan).replace([np.inf, -np.inf], np.nan)
            if x.notna().sum() < 2:
                return pd.Series(0.5, index=u)
            med = x.median()
            x = x.fillna(med)
            return x.rank(method='average', pct=True)

        r21 = prep(ret21)
        r5 = prep(ret5)
        rv = prep(vol)
        rl = prep(liq)

        score = -0.65 * r21 - 0.35 * r5 - 0.10 * rv + 0.05 * rl
        score = score.where(valid, np.nan).replace([np.inf, -np.inf], np.nan)
        cand = score.dropna()
        if len(cand) < 6:
            return zero

        k = min(40, len(cand) // 2)
        if k < 1:
            return zero

        wgt = min(0.02, 0.5 / k)
        ordered = cand.sort_values(ascending=False)
        longs = ordered.index[:k]
        shorts = ordered.index[-k:]

        out = zero.copy()
        out.loc[longs] = wgt
        out.loc[shorts] = -wgt
        out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
        return out
    except Exception:
        return zero
