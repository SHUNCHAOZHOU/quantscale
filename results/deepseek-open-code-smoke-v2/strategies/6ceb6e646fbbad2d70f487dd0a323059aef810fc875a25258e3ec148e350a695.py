import numpy as np
import pandas as pd


def _as_df(x):
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, dict):
        try:
            return pd.DataFrame(x)
        except Exception:
            return None
    return None


def generate_weights(history, universe):
    try:
        cols = pd.Index(universe)
    except Exception:
        cols = pd.Index(list(universe))
    zero = pd.Series(0.0, index=cols, dtype=float)
    n = len(cols)
    if n < 2:
        return zero

    px = _as_df(history.get('adjusted_close', None))
    if px is None or px.empty:
        px = _as_df(history.get('close', None))
    if px is None or px.empty:
        return zero

    vol = _as_df(history.get('volume', None))
    elig = _as_df(history.get('eligible', None))
    if px.index is None or len(px.index) == 0:
        return zero
    latest = px.index[-1]

    records = []
    for t in cols:
        if t not in px.columns:
            continue
        s = px[t].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 2:
            continue
        try:
            if (latest - s.index[-1]).days > 7:
                continue
        except Exception:
            if s.index[-1] != latest:
                continue

        last = float(s.iloc[-1])
        if not np.isfinite(last) or last <= 0:
            continue

        if elig is not None and t in elig.columns:
            e = elig[t].replace([np.inf, -np.inf], np.nan).dropna()
            if len(e) > 0:
                ev = e.iloc[-1]
                if isinstance(ev, str):
                    if ev.strip().lower() in ('false', '0', 'nan', ''):
                        continue
                else:
                    try:
                        if not bool(ev):
                            continue
                    except Exception:
                        pass

        def ret(k):
            if len(s) > k:
                prev = float(s.iloc[-1 - k])
                if np.isfinite(prev) and prev > 0:
                    return last / prev - 1.0
            return np.nan

        r1 = ret(1)
        r5 = ret(5)
        r10 = ret(10)
        r20 = ret(20)
        r60 = ret(60)

        rets = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(rets) >= 20:
            vol20 = float(rets.iloc[-20:].std(ddof=0))
        elif len(rets) >= 5:
            vol20 = float(rets.std(ddof=0))
        else:
            vol20 = np.nan
        if len(rets) >= 60:
            vol60 = float(rets.iloc[-60:].std(ddof=0))
        elif len(rets) >= 10:
            vol60 = float(rets.std(ddof=0))
        else:
            vol60 = np.nan

        vr = np.nan
        dv = np.nan
        if vol is not None and t in vol.columns:
            v = vol[t].replace([np.inf, -np.inf], np.nan).dropna()
            if len(v) >= 5:
                v5 = float(v.iloc[-5:].mean())
                if len(v) >= 20:
                    v20 = float(v.iloc[-20:].mean())
                else:
                    v20 = float(v.mean())
                if v20 > 0:
                    vr = v5 / v20
                dv = v20

        records.append({
            'ticker': t,
            'r1': r1,
            'r5': r5,
            'r10': r10,
            'r20': r20,
            'r60': r60,
            'vol20': vol20,
            'vol60': vol60,
            'vr': vr,
            'dv': dv
        })

    if len(records) < 2:
        return zero

    df = pd.DataFrame(records).set_index('ticker')
    if len(df) > 60:
        df = df.assign(_dv_rank=df['dv'].rank(pct=True, method='average').fillna(0.0))
        df = df.sort_values('_dv_rank', ascending=False, kind='mergesort').iloc[:60]

    def zrank(col):
        x = df[col]
        if x.notna().sum() == 0:
            return pd.Series(0.0, index=df.index, dtype=float)
        r = x.rank(pct=True, method='average')
        z = (r - 0.5) * 2.0
        return z.fillna(0.0)

    zr1 = zrank('r1')
    zr5 = zrank('r5')
    zr10 = zrank('r10')
    zr20 = zrank('r20')
    zr60 = zrank('r60')
    zvol = zrank('vol20')
    zvr = zrank('vr')

    score = (0.45 * zr20 + 0.15 * zr60 - 0.35 * zr10 - 0.10 * zr5
             - 0.15 * zvol - 0.05 * zvr)

    if zr20.abs().sum() == 0 and zr10.abs().sum() == 0:
        score = -0.50 * zr5 - 0.25 * zr1 - 0.15 * zvol

    score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if float(score.abs().sum()) < 1e-12:
        return zero

    df = df.assign(score=score)
    ranked = df.sort_values('score', ascending=False, kind='mergesort')
    k = min(25, len(ranked) // 2)
    if k < 1:
        return zero

    longs = ranked.index[:k].tolist()
    shorts = ranked.index[-k:].tolist()
    max_w = 0.0199

    w = pd.Series(0.0, index=cols, dtype=float)
    for t in longs:
        if t in w.index:
            w.loc[t] = max_w
    for t in shorts:
        if t in w.index:
            w.loc[t] = -max_w

    w = w.clip(-0.02, 0.02)
    gross = float(w.abs().sum())
    if gross > 1.0:
        w = w * (1.0 / gross)

    net = float(w.sum())
    if abs(net) > 0.005:
        nz = w[w != 0.0]
        if len(nz) > 0:
            w.loc[nz.index] = w.loc[nz.index] - net / len(nz)
            w = w.clip(-0.02, 0.02)

    return w.astype(float)