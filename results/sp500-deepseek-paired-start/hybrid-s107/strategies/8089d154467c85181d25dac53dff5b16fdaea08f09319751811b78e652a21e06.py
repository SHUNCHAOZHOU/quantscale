import numpy as np
import pandas as pd


def _rank_center(s):
    """Rank-based centering into [-0.5, 0.5]; robust to outliers."""
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 2:
        return s
    return s.rank(method='average', pct=True) - 0.5


def _spread_weights(raw, total, cap):
    """Allocate `total` gross across raw proportional to raw, with per-name cap."""
    if raw is None or len(raw) == 0:
        return None
    r = raw.replace([np.inf, -np.inf], np.nan).dropna()
    r = r[r > 0]
    if len(r) < 1:
        return None
    s = float(r.sum())
    if not np.isfinite(s) or s <= 0:
        return None
    w = r * (total / s)
    for _ in range(8):
        over = w > (cap + 1e-15)
        if not bool(over.any()):
            break
        w = w.clip(upper=cap)
        free = ~over
        rem = total - float(w[over].sum())
        if rem <= 1e-12 or not bool(free.any()):
            break
        sub = w[free]
        tot = float(sub.sum())
        if tot <= 1e-12:
            break
        w[free] = sub * (rem / tot)
    return w.clip(upper=cap)


def _sub_book(P, R, i, cols, elig_ok, liq_ok, wcap=0.02, k_max=45, leg=0.5):
    """One dollar-neutral, inverse-vol weighted sub-portfolio at snapshot row i."""
    p_i = P.iloc[i]
    p5 = P.iloc[i - 5]
    p21 = P.iloc[i - 21]
    p126 = P.iloc[i - 126]
    vol20 = R.iloc[i - 19:i + 1].std(ddof=1)
    vol126 = R.iloc[i - 125:i + 1].std(ddof=1)

    valid = (p_i.notna() & (p_i > 0)
             & p5.notna() & (p5 > 0)
             & p21.notna() & (p21 > 0)
             & p126.notna() & (p126 > 0)
             & vol20.notna() & (vol20 > 0)
             & vol126.notna() & (vol126 > 0))

    if elig_ok is not None:
        e = elig_ok.reindex(cols)
        if e.notna().sum() >= 12:
            valid = valid & e.fillna(False).astype(bool)
    if liq_ok is not None:
        l = liq_ok.reindex(cols)
        if l.notna().sum() >= 12:
            valid = valid & l.fillna(False).astype(bool)

    valid = valid.fillna(False).astype(bool)
    idxv = valid[valid].index
    if len(idxv) < 14:
        return None

    eps = 1e-9
    rev = -(p_i / p5 - 1.0) / (vol20 * np.sqrt(5.0) + eps)
    trd = (p21 / p126 - 1.0) / (vol126 * np.sqrt(105.0) + eps)

    rz = _rank_center(rev.reindex(idxv))
    tz = _rank_center(trd.reindex(idxv))
    vz = _rank_center(vol20.reindex(idxv) * -1.0)
    if len(rz) < 12 or len(tz) < 12:
        return None

    score = 0.60 * rz + 0.30 * tz + 0.10 * vz
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 12:
        return None

    k = int(min(k_max, len(score) // 3))
    if k < 4:
        return None

    longs = score.nlargest(k).index
    shorts = score.nsmallest(k).index

    w = pd.Series(0.0, index=list(cols), dtype=float)
    for names, sign in ((longs, 1.0), (shorts, -1.0)):
        v = vol20.reindex(names).replace([np.inf, -np.inf], np.nan).dropna()
        v = v[v > 0]
        if len(v) < 1:
            continue
        iv = 1.0 / v.clip(lower=1e-4)
        wl = _spread_weights(iv, leg, wcap)
        if wl is None or len(wl) == 0:
            continue
        w.loc[wl.index] = w.loc[wl.index] + sign * wl.values

    if float(w.abs().sum()) <= 0:
        return None
    return w


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    zero = pd.Series(0.0, index=tickers, dtype=float)
    if len(tickers) == 0:
        return zero

    px = history.get('adjusted_close')
    if not isinstance(px, pd.DataFrame):
        px = history.get('close')
    if not isinstance(px, pd.DataFrame) or px.shape[0] < 155:
        return zero
    if not px.index.is_monotonic_increasing:
        px = px.sort_index()

    cols = [c for c in tickers if c in px.columns]
    if len(cols) < 20:
        return zero

    n = int(px.shape[0])
    win = int(min(240, n))
    P = px[cols].iloc[-win:].astype(float)
    P = P.replace([np.inf, -np.inf], np.nan)
    m = int(P.shape[0])
    if m < 155:
        return zero
    R = P / P.shift(1) - 1.0
    R = R.replace([np.inf, -np.inf], np.nan)

    # Point-in-time eligibility flag (last available row only).
    elig_ok = None
    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame) and elig.shape[0] > 0:
        e = elig.reindex(columns=cols)
        if not e.index.is_monotonic_increasing:
            e = e.sort_index()
        last_e = e.iloc[-1].replace([np.inf, -np.inf], np.nan)
        if int(last_e.notna().sum()) >= 20:
            elig_ok = last_e.fillna(False).astype(bool)

    # Liquidity screen: recent median dollar volume must be in the upper ~70%.
    liq_ok = None
    vdf = history.get('volume')
    if isinstance(vdf, pd.DataFrame) and vdf.shape[0] > 0:
        v = vdf.reindex(columns=cols).iloc[-60:].astype(float)
        v = v.replace([np.inf, -np.inf], np.nan)
        dvdf = v * P.iloc[-60:]
        dv = dvdf.median(axis=0).replace([np.inf, -np.inf], np.nan)
        if int(dv.notna().sum()) >= 50:
            thr = dv.quantile(0.30)
            liq_ok = dv.notna() & (dv > 0)
            if np.isfinite(thr):
                liq_ok = liq_ok & (dv >= thr)

    # Staggered sub-books: sub-book j refreshes every K*D sessions,
    # exactly one sub-book refreshes every D sessions.
    K = 5
    D = 5
    KD = K * D
    last_abs = n - 1
    base_abs = n - m  # absolute position of relative row 0

    books = []
    for j in range(K):
        lag = int((last_abs - j * D) % KD)
        i_abs = last_abs - lag
        i = int(i_abs - base_abs)
        if i < 130 or i > m - 1:
            continue
        b = _sub_book(P, R, i, cols, elig_ok, liq_ok)
        if b is not None:
            books.append(b)

    if len(books) == 0:
        return zero

    W = pd.concat(books, axis=1).mean(axis=1).reindex(cols)
    W = W.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    W = W.clip(-0.02, 0.02)

    gross = float(W.abs().sum())
    if gross > 1.0:
        W = W * (1.0 / gross)

    # Keep the book essentially net-neutral (average of dollar-neutral books).
    net = float(W.sum())
    if abs(net) > 0.008 and gross > 0:
        W = W - net / float((W != 0).sum() if (W != 0).sum() > 0 else len(W))
        W = W.clip(-0.02, 0.02)
        gross = float(W.abs().sum())
        if gross > 1.0:
            W = W * (1.0 / gross)

    out = pd.Series(0.0, index=tickers, dtype=float)
    if len(W) > 0:
        out.loc[W.index] = W.values
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out
