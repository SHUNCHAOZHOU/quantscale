import numpy as np
import pandas as pd

def _prep(df, cols):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    common = [c for c in cols if c in df.columns]
    if not common:
        return None
    sub = df.loc[:, common]
    if sub.columns.duplicated().any():
        sub = sub.loc[:, ~sub.columns.duplicated()]
    return sub

def _row(sub, pos, cols):
    if sub is None or pos <= 0 or pos > len(sub):
        return None
    r = sub.iloc[pos-1]
    if isinstance(r, pd.DataFrame):
        r = r.iloc[:, 0]
    return pd.to_numeric(r, errors='coerce').reindex(cols)

def _rk(s):
    return s.rank(method='average', pct=True, na_option='keep')

def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero
    try:
        ar = history.get('adjusted_close')
        if ar is None or ar.empty:
            ar = history.get('close')
        adj = _prep(ar, universe)
        if adj is None or adj.empty:
            return zero
        adj = adj.sort_index()
        idx = adj.index
        if len(idx) < 30:
            return zero
        last = idx[-1]
        try:
            nxt = last + pd.offsets.BDay(1)
            me = (nxt.year != last.year) or (nxt.month != last.month)
        except Exception:
            me = False
        if me:
            asof = last
        else:
            per = idx.to_period('M')
            prev = last.to_period('M') - 1
            mask = per == prev
            asof = idx[mask][-1] if mask.any() else idx[-1]
        pos = int(idx.searchsorted(asof, side='right'))
        if pos <= 0:
            return zero
        px = adj.iloc[max(0, pos-300):pos]
        if len(px) < 30:
            return zero
        cols = list(universe)
        ret = px / px.shift(1) - 1.0
        ret = ret.replace([np.inf, -np.inf], np.nan)
        last_px = _row(px, len(px), cols)
        if last_px is None:
            return zero
        rev = -(px.iloc[-1] / px.shift(21).iloc[-1] - 1.0) if len(px) >= 22 else pd.Series(np.nan, index=px.columns)
        mom = px.shift(21).iloc[-1] / px.shift(252).iloc[-1] - 1.0 if len(px) >= 253 else pd.Series(np.nan, index=px.columns)
        vol = ret.tail(63).std() if len(ret) >= 20 else pd.Series(np.nan, index=px.columns)
        rev = rev.reindex(cols)
        mom = mom.reindex(cols)
        vol = vol.reindex(cols)
        score = 0.40*_rk(mom).fillna(0.5) + 0.40*_rk(rev) - 0.20*_rk(vol)
        valid = (last_px > 0) & rev.notna() & vol.notna() & score.notna()
        elig = _prep(history.get('eligible'), cols)
        er = _row(elig, pos, cols) if elig is not None else None
        if er is not None:
            valid = valid & (er.fillna(1.0).astype(float) > 0.5)
        vdf = _prep(history.get('volume'), cols)
        cdf = _prep(history.get('close'), cols)
        if vdf is not None and cdf is not None:
            st = max(0, pos-21)
            vv = vdf.iloc[st:pos]
            cc = cdf.iloc[st:pos]
            ci = vv.index.intersection(cc.index)
            if len(ci) > 0:
                adv = (vv.loc[ci] * cc.loc[ci]).mean().reindex(cols)
                valid = valid & (adv.fillna(0.0) > 5e5)
        cand = score[valid].dropna()
        if len(cand) < 6:
            return zero
        n = min(30, len(cand)//2)
        if n < 1:
            return zero
        order = cand.sort_values(ascending=False)
        longs = order.index[:n]
        shorts = order.index[-n:]
        gt = 0.6
        try:
            disp = ret.tail(21).std(axis=1).mean()
            if np.isfinite(disp) and disp < 0.012:
                gt = 0.4
            sub = ret.tail(60)
            sub = sub.loc[:, sub.notna().sum() >= 40]
            if sub.shape[1] > 60:
                sub = sub.iloc[:, :60]
            if sub.shape[1] >= 3:
                corr = sub.corr(min_periods=30).values
                iu = np.triu_indices_from(corr, k=1)
                ac = np.nanmean(corr[iu])
                if np.isfinite(ac) and ac > 0.7:
                    gt = min(gt, 0.4)
        except Exception:
            pass
        w = min(0.02, (gt/2.0)/float(n))
        weights = pd.Series(0.0, index=universe, dtype=float)
        for t in longs:
            weights.loc[t] += w
        for t in shorts:
            weights.loc[t] -= w
        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.02, 0.02)
        return weights
    except Exception:
        return zero