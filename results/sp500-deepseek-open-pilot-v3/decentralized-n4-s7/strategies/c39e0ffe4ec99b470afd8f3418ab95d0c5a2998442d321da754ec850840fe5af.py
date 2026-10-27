import numpy as np
import pandas as pd

_CACHE = {}


def _zscore(s):
    s = s.replace([np.inf, -np.inf], np.nan)
    m = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(0.0, index=s.index)
    return (s - m) / sd


def generate_weights(history, universe):
    u = list(universe)
    zero = pd.Series(0.0, index=u, dtype=float)
    try:
        adj = history["adjusted_close"]
    except Exception:
        return zero
    if not isinstance(adj, pd.DataFrame) or adj.shape[0] < 65:
        return zero

    idx = adj.index
    cut = len(idx) - 1
    if hasattr(idx, "year"):
        try:
            months = np.asarray(idx.year, dtype=np.int64) * 12 + np.asarray(idx.month, dtype=np.int64)
        except Exception:
            months = None
        if months is not None and len(months) > 1:
            ch = np.where(months[1:] != months[:-1])[0]
            if len(ch) > 0:
                cut = int(ch[-1])
            try:
                last_ts = idx[-1]
                nxt = last_ts + pd.tseries.offsets.BDay(1)
                if nxt.month != last_ts.month:
                    cut = len(idx) - 1
            except Exception:
                pass

    if cut < 60:
        return zero

    cut_date = idx[cut]
    key = (cut_date, len(u), u[0] if u else "", u[-1] if u else "")
    if key in _CACHE:
        return _CACHE[key].reindex(u).fillna(0.0).astype(float)

    adj_sub = adj.iloc[: cut + 1]
    n = len(adj_sub)
    cols = [c for c in u if c in adj_sub.columns]
    if len(cols) < 30:
        return zero
    px = adj_sub[cols].astype(float)

    try:
        el = history["eligible"]
        if isinstance(el, pd.DataFrame):
            el_a = el.reindex(adj_sub.index)
            el_last = el_a.iloc[-1].reindex(cols) if len(el_a) > 0 else pd.Series(True, index=cols)
        else:
            el_last = pd.Series(True, index=cols)
    except Exception:
        el_last = pd.Series(True, index=cols)
    el_last = el_last.fillna(False).astype(bool)

    try:
        vh = history["volume"]
        if isinstance(vh, pd.DataFrame):
            vh_a = vh.reindex(adj_sub.index)
            tail_v = vh_a.iloc[-61:][cols] if len(vh_a) >= 1 else None
            med_vol = tail_v.median() if tail_v is not None else pd.Series(np.inf, index=cols)
        else:
            med_vol = pd.Series(np.inf, index=cols)
    except Exception:
        med_vol = pd.Series(np.inf, index=cols)
    if not isinstance(med_vol, pd.Series):
        med_vol = pd.Series(np.inf, index=cols)

    last = px.iloc[-1]
    valid = el_last & last.notna() & (last > 1.0) & (med_vol.reindex(cols).fillna(0) > 50000)
    if int(valid.sum()) < 20:
        return zero

    tail_px = px.iloc[-61:]
    rets = tail_px.pct_change()
    vol = rets.std()
    vol = vol.replace(0, np.nan).fillna(1e-4).clip(lower=1e-5)

    skip = int(min(21, max(1, n // 12)))
    look = int(min(252, n - skip - 5))
    if look < 20:
        return zero
    end_i = n - 1 - skip
    start_i = max(0, end_i - look)
    mom = px.iloc[end_i] / px.iloc[start_i] - 1.0
    mom = mom.replace([np.inf, -np.inf], np.nan)

    rev_look = int(min(5, n - 1))
    rev = px.iloc[-1] / px.iloc[-1 - rev_look] - 1.0
    rev = rev.replace([np.inf, -np.inf], np.nan)

    mom_ra = mom / vol
    rev_ra = -rev / vol
    score = _zscore(mom_ra) + 0.5 * _zscore(rev_ra)
    score = score.where(valid).replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 20:
        return zero

    n_side = min(50, len(score) // 2)
    if n_side < 5:
        return zero
    w_side = 0.5 / n_side
    if w_side > 0.02:
        w_side = 0.02
        n_side = min(n_side, 25)

    ranked = score.sort_values(kind="mergesort")
    shorts = ranked.index[:n_side]
    longs = ranked.index[-n_side:]

    w = zero.copy()
    w.loc[longs] = w_side
    w.loc[shorts] = -w_side
    _CACHE[key] = w
    return w
