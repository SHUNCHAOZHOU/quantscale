import math
import numpy as np
import pandas as pd

_STD = ('open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'eligible')


def _panel(history, universe, field, fallback=None):
    if not isinstance(history, dict):
        return pd.DataFrame()
    df = None
    if isinstance(history.get(field), pd.DataFrame):
        df = history[field]
    elif fallback is not None and isinstance(history.get(fallback), pd.DataFrame):
        df = history[fallback]
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    cols = [t for t in universe if t in df.columns]
    if not cols:
        return pd.DataFrame()
    out = df.loc[:, cols].apply(pd.to_numeric, errors='coerce')
    return out.sort_index()


def _get(history, universe, field, fallback=None):
    if not isinstance(history, dict):
        return pd.DataFrame()
    panel = any(isinstance(history.get(k), pd.DataFrame) for k in _STD)
    if panel:
        return _panel(history, universe, field, fallback)
    data = {}
    for t in universe:
        d = history.get(t)
        if not isinstance(d, pd.DataFrame):
            continue
        if field in d.columns:
            col = field
        elif fallback is not None and fallback in d.columns:
            col = fallback
        else:
            continue
        s = pd.to_numeric(d[col], errors='coerce').dropna()
        if len(s):
            data[t] = s
    if data:
        return pd.DataFrame(data).sort_index()
    return pd.DataFrame()


def _z(s):
    s = pd.to_numeric(s, errors='coerce')
    mu = s.mean()
    sd = s.std(ddof=0)
    if (not np.isfinite(sd)) or sd == 0:
        return pd.Series(0.0, index=s.index, dtype=float)
    return (s - mu) / sd


def _side_weights(names, invvol, budget, cap):
    w = pd.Series(0.0, index=pd.Index(names), dtype=float)
    active = list(names)
    b = float(budget)
    for _ in range(len(names) + 2):
        if b <= 1e-12 or not active:
            break
        iv = invvol.reindex(active).astype(float)
        s = float(iv.sum())
        if (not np.isfinite(s)) or s <= 0:
            break
        prop = b * iv / s
        over = prop[prop > cap + 1e-15]
        if over.empty:
            w.loc[active] = prop.values
            b = 0.0
            break
        idx = list(over.index)
        w.loc[idx] = cap
        b -= cap * len(idx)
        drop = set(idx)
        active = [t for t in active if t not in drop]
    return w


def generate_weights(history, universe):
    universe = pd.Index(list(universe))
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    closes = _get(history, universe, 'adjusted_close', 'close')
    if closes is None or closes.empty:
        return zero
    closes = closes.replace([np.inf, -np.inf], np.nan).dropna(how='all')
    if len(closes) < 41:
        return zero

    dates = closes.index
    last = dates[-1]
    mask = (dates.year == last.year) & (dates.month == last.month)
    where = np.flatnonzero(np.asarray(mask))
    if where.size == 0:
        return zero
    pos = int(where[0])
    if pos < 40:
        return zero

    px = closes.iloc[:pos + 1]
    n = len(px)
    if n < 130:
        return zero

    last_px = px.iloc[-1]
    valid = last_px.notna() & (last_px > 0)

    rets = px.pct_change()
    vol = rets.iloc[-126:].std(ddof=0)

    if n >= 22:
        rev1 = last_px / px.iloc[-22] - 1.0
    else:
        rev1 = pd.Series(np.nan, index=px.columns, dtype=float)

    if n >= 253:
        mom = px.iloc[-22] / px.iloc[-253] - 1.0
    else:
        mom = pd.Series(np.nan, index=px.columns, dtype=float)

    volp = _get(history, universe, 'volume')
    if volp is not None and not volp.empty:
        volp = volp.reindex(index=px.index, columns=px.columns).apply(pd.to_numeric, errors='coerce')
        dollar = px * volp
        liq = dollar.iloc[-60:].median()
    else:
        liq = pd.Series(np.nan, index=px.columns, dtype=float)
    logliq = np.log1p(liq.where(liq > 0))

    elig = _get(history, universe, 'eligible')
    if elig is not None and not elig.empty:
        elig = elig.reindex(index=px.index, columns=px.columns)
        e = elig.iloc[-1]
        if e.notna().any():
            en = pd.to_numeric(e, errors='coerce')
            valid = valid & (en.fillna(1.0) > 0.5)

    df = pd.DataFrame({
        'vol': pd.to_numeric(vol, errors='coerce'),
        'rev1': pd.to_numeric(rev1, errors='coerce'),
        'mom': pd.to_numeric(mom, errors='coerce'),
        'logliq': pd.to_numeric(logliq, errors='coerce'),
    }, index=px.columns)

    df = df.loc[valid.reindex(df.index).fillna(False)]
    keep = df['vol'].notna() & (df['vol'] > 1e-6) & df['rev1'].notna() & df['logliq'].notna()
    df = df.loc[keep]
    if len(df) < 10:
        return zero

    z_vol = _z(df['vol'])
    z_liq = _z(df['logliq'])
    z_rev = _z(df['rev1'])
    score = -0.9 * z_vol + 0.4 * z_liq - 0.5 * z_rev

    has_mom = int(df['mom'].notna().sum()) >= max(10, int(0.5 * len(df)))
    if has_mom:
        m = df['mom'].fillna(df['mom'].median())
        score = score - 0.15 * _z(m)

    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 10:
        return zero

    n_side = min(35, len(score) // 2)
    if n_side < 5:
        return zero

    ranked = score.sort_values(ascending=False)
    longs = list(ranked.index[:n_side])
    shorts = list(ranked.index[-n_side:])

    vv = df['vol'].astype(float).copy()
    q_lo = float(vv.quantile(0.05))
    q_hi = float(vv.quantile(0.95))
    if not np.isfinite(q_lo):
        q_lo = float(vv.min())
    if not np.isfinite(q_hi) or q_hi <= q_lo:
        q_hi = q_lo + 1e-6
    vv = vv.clip(q_lo, q_hi)
    invvol = 1.0 / vv

    cap = 0.02
    budget = 0.46
    wl = _side_weights(longs, invvol, budget, cap)
    ws = _side_weights(shorts, invvol, budget, cap)

    def _est_var(w, v):
        if w is None or len(w) == 0:
            return 0.0
        x = (w * v.reindex(w.index)).fillna(0.0)
        s1 = float(x.sum())
        s2 = float((x * x).sum())
        rho = 0.2
        return rho * s1 * s1 + (1.0 - rho) * s2

    var = _est_var(wl, vv) + _est_var(ws, vv)
    est = math.sqrt(var) if var > 0 else 0.0
    target = 0.09
    k = 1.0 if est <= 1e-12 else min(1.0, target / est)

    w = pd.Series(0.0, index=universe, dtype=float)
    w = w.add(k * wl.reindex(universe, fill_value=0.0), fill_value=0.0)
    w = w.add(-k * ws.reindex(universe, fill_value=0.0), fill_value=0.0)

    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-cap, cap)
    gross = float(w.abs().sum())
    if gross > 1.0:
        w = w * (1.0 / gross)
    net = float(w.sum())
    gross = float(w.abs().sum())
    if gross > 1e-12 and abs(net) > 1e-9:
        w = w - net * (w.abs() / gross)
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-cap, cap)
    return w.astype(float)