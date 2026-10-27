import numpy as np
import pandas as pd


def _pct_rank(x, mask):
    x = x.reindex(mask.index).astype(float)
    x = x.where(mask, np.nan)
    if x.notna().sum() < 2:
        return pd.Series(0.5, index=mask.index)
    med = x.median()
    x = x.fillna(med)
    return x.rank(method='average', pct=True)


def generate_weights(history, universe):
    u = pd.Index(universe)
    zero = pd.Series(0.0, index=u, dtype=float)
    try:
        adj_full = history.get('adjusted_close')
        if adj_full is None:
            adj_full = history.get('close')
        if adj_full is None or adj_full.shape[0] < 50:
            return zero

        adj_full = adj_full.reindex(columns=u).astype(float)
        idx = adj_full.index
        current = idx[-1]
        prior_mask = (idx.year < current.year) | ((idx.year == current.year) & (idx.month < current.month))
        if not prior_mask.any():
            return zero

        sig_date = idx[prior_mask][-1]
        sig_loc = int(idx.get_indexer([sig_date])[0])
        if sig_loc < 40:
            return zero

        adj = adj_full.iloc[:sig_loc + 1]
        last = adj.iloc[-1]
        valid = last.notna() & np.isfinite(last) & (last > 1.0)
        valid = valid & (adj.notna().sum() >= 80)

        elig_full = history.get('eligible')
        if elig_full is not None:
            try:
                elig = elig_full.reindex(index=adj.index, columns=u).astype(float)
                e = elig.iloc[-1]
                valid = valid & (e.fillna(0.0) > 0)
            except Exception:
                pass

        if int(valid.sum()) < 30:
            return zero

        vol_full = history.get('volume')
        if vol_full is not None:
            try:
                vdf = vol_full.reindex(index=adj.index, columns=u).astype(float)
                dv = adj * vdf
                liq = dv.iloc[-min(60, len(dv)):].median()
            except Exception:
                liq = pd.Series(0.0, index=u)
        else:
            liq = pd.Series(0.0, index=u)
        liq = liq.reindex(u).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        rets = adj / adj.shift(1) - 1.0
        rets = rets.replace([np.inf, -np.inf], np.nan)

        liq_valid = liq.where(valid)
        n_valid = int(valid.sum())
        top_n = min(300, n_valid)
        if top_n < 20:
            return zero
        liq_cut = liq_valid.nlargest(top_n).min()
        pool = valid & (liq >= liq_cut)
        if int(pool.sum()) < 20:
            return zero

        look = min(120, len(rets) - 1)
        if look < 40:
            return zero

        R = rets.iloc[-look:]
        M = R.values
        m = np.nanmean(M, axis=1)
        m = np.where(np.isfinite(m), m, 0.0)
        var_m = np.nanvar(m)
        beta = np.full(M.shape[1], np.nan)
        if var_m > 0.0:
            for j in range(M.shape[1]):
                x = M[:, j]
                msk = np.isfinite(x)
                if int(msk.sum()) >= 40:
                    xv = x[msk]
                    mv = m[msk]
                    xm = xv - xv.mean()
                    mm = mv - mv.mean()
                    den = float(np.dot(mm, mm))
                    if den > 0.0:
                        beta[j] = float(np.dot(xm, mm) / den)

        beta_s = pd.Series(beta, index=u)
        beta_s = beta_s.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(-1.0, 3.0)

        market_full = rets.mean(axis=1)
        R2 = rets.iloc[-look:]
        m2 = market_full.iloc[-look:].values
        m2 = np.where(np.isfinite(m2), m2, 0.0)
        resid = R2.values - m2[:, None] * beta_s.values[None, :]
        resid_vol = np.nanstd(resid, axis=0)
        resid_vol_s = pd.Series(resid_vol, index=u)
        resid_vol_s = resid_vol_s.replace([np.inf, -np.inf], np.nan)
        resid_vol_s = resid_vol_s.where(resid_vol_s > 1e-8, np.nan)

        def _resid_sum(k):
            k = min(k, len(rets))
            rr = rets.iloc[-k:]
            mm = market_full.iloc[-k:].values
            mm = np.where(np.isfinite(mm), mm, 0.0)
            res = rr.values - mm[:, None] * beta_s.values[None, :]
            return pd.Series(np.nansum(res, axis=0), index=u)

        r21 = _resid_sum(21)
        r5 = _resid_sum(5)
        r21z = r21 / (resid_vol_s * np.sqrt(21.0))
        r5z = r5 / (resid_vol_s * np.sqrt(5.0))
        r21z = r21z.replace([np.inf, -np.inf], np.nan)
        r5z = r5z.replace([np.inf, -np.inf], np.nan)

        pool = pool & r21z.notna() & r5z.notna()
        if int(pool.sum()) < 20:
            return zero

        vol60 = rets.iloc[-min(60, len(rets)):].std()
        rk21 = _pct_rank(r21z, pool)
        rk5 = _pct_rank(r5z, pool)
        rkv = _pct_rank(vol60, pool)
        rkl = _pct_rank(liq, pool)

        score = (1.0 - rk21) + 0.5 * (1.0 - rk5) - 0.15 * rkv + 0.05 * rkl
        score = score.where(pool, np.nan)

        beta_pool = beta_s.where(pool)
        brank = beta_pool.rank(method='first')
        nb = int(pool.sum())
        q = np.ceil(brank / float(nb) * 5.0).clip(1, 5)

        long_candidates = []
        short_candidates = []
        for b in range(1, 6):
            members = score[(q == b) & pool].dropna()
            if len(members) < 2:
                continue
            n_side = min(6, len(members) // 2)
            if n_side < 1:
                continue
            ordered = members.sort_values(ascending=False)
            long_candidates.extend(ordered.index[:n_side].tolist())
            short_candidates.extend(ordered.index[-n_side:].tolist())

        long_candidates = list(dict.fromkeys(long_candidates))
        short_candidates = list(dict.fromkeys(short_candidates))
        long_set = set(long_candidates)
        short_candidates = [t for t in short_candidates if t not in long_set]

        if len(long_candidates) < 5 or len(short_candidates) < 5:
            return zero

        long_sorted = sorted(
            long_candidates,
            key=lambda t: score.loc[t] if pd.notna(score.loc[t]) else -1e9,
            reverse=True,
        )
        short_sorted = sorted(
            short_candidates,
            key=lambda t: score.loc[t] if pd.notna(score.loc[t]) else 1e9,
        )

        k = min(len(long_sorted), len(short_sorted), 30)
        if k < 5:
            return zero

        w = min(0.02, 0.5 / float(k))
        out = zero.copy()
        out.loc[long_sorted[:k]] = w
        out.loc[short_sorted[:k]] = -w
        out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.02, 0.02)
        return out
    except Exception:
        return zero
