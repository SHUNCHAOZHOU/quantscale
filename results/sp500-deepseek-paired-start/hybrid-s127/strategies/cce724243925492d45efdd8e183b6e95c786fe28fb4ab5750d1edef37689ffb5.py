import numpy as np
import pandas as pd


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    ac = history.get('adjusted_close')
    if not isinstance(ac, pd.DataFrame):
        ac = history.get('close')
    if not isinstance(ac, pd.DataFrame) or ac.shape[0] == 0 or ac.shape[1] == 0:
        return zero

    cols = [c for c in universe if c in ac.columns]
    if len(cols) < 4:
        return zero

    px = ac.loc[:, cols].astype(float).ffill()
    n = px.shape[0]
    if n < 130:
        return zero

    R = 10
    anchor = ((n - 1) // R) * R
    if anchor < 127:
        return zero

    pw = px.iloc[:anchor + 1]
    na = pw.shape[0]
    last = pw.iloc[-1]

    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame) and elig.shape[0] > 0:
        e_last = elig.iloc[-1].reindex(cols).fillna(0).astype(bool)
    else:
        e_last = pd.Series(True, index=cols)

    last_np = last.to_numpy(dtype=float)
    valid = pd.Series(
        last.notna().to_numpy() & np.isfinite(last_np) & (last_np > 0.0),
        index=cols,
    ) & e_last

    volp = history.get('volume')
    closp = history.get('close')
    if isinstance(volp, pd.DataFrame):
        vp = volp.reindex(index=px.index, columns=cols).iloc[:anchor + 1].astype(float)
        if isinstance(closp, pd.DataFrame):
            cp = closp.reindex(index=px.index, columns=cols).iloc[:anchor + 1].astype(float)
        else:
            cp = pw
        dv = (vp * cp).iloc[-60:]
        med = dv.median()
        liq_ok = med.fillna(0.0) > 1e6
        valid = valid & liq_ok.reindex(cols).fillna(False)

    def ret_over(lb, skip):
        end = na - 1 - skip
        start = end - lb
        if start < 0:
            return pd.Series(np.nan, index=cols, dtype=float)
        return pw.iloc[end] / pw.iloc[start] - 1.0

    rev1 = -ret_over(21, 0)
    mom12 = ret_over(252, 21)
    mom6 = ret_over(126, 21)
    mom = mom12.where(mom12.notna(), mom6)
    vol60 = pw.pct_change().iloc[-60:].std()

    f = pd.DataFrame({'rev': rev1, 'mom': mom, 'vol': vol60}).reindex(cols)
    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.loc[valid]
    if len(f) < 4:
        return zero

    score = pd.Series(0.0, index=f.index, dtype=float)
    wsum = pd.Series(0.0, index=f.index, dtype=float)
    for name, wgt in (('rev', 0.55), ('mom', 0.25), ('vol', 0.20)):
        s = f[name]
        if name == 'vol':
            s = -s
        r = s.rank(pct=True)
        mask = r.notna()
        score[mask] += wgt * r[mask]
        wsum[mask] += wgt
    score = (score / wsum).where(wsum > 0)

    cand = score.dropna()
    if len(cand) < 4:
        return zero
    n_side = int(min(50, len(cand) // 2))
    if n_side < 1:
        return zero

    ranked = cand.sort_values(ascending=False)
    longs = ranked.index[:n_side]
    shorts = ranked.index[-n_side:]

    w = min(0.02, 0.5 / float(n_side))
    out = zero.copy()
    out.loc[longs] = w
    out.loc[shorts] = -w
    out = out.clip(-0.02, 0.02)
    gross = float(out.abs().sum())
    if gross > 1.0 and gross > 0.0:
        out = out * (1.0 / gross)
    return out.fillna(0.0).astype(float)
