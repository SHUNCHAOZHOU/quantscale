import numpy as np
import pandas as pd

_STATE = {'month': None, 'weights': None, 'universe': None}


def _rank_df(df):
    return df.rank(axis=1, pct=True) - 0.5


def _factor_return(px, sig_df, hold=21, n_dates=12, top_frac=0.20):
    n = px.shape[0]
    if n < hold + 2:
        return 0.0
    last_j = n - 1 - hold
    first_j = max(hold, last_j - (n_dates - 1) * hold)
    vals = []
    for j in range(last_j, first_j - 1, -hold):
        s = sig_df.iloc[j]
        p0 = px.iloc[j]
        p1 = px.iloc[j + hold]
        v = s.notna() & p0.notna() & p1.notna() & (p0 > 0.0)
        if int(v.sum()) < 10:
            continue
        ss = s[v].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ss) < 10:
            continue
        k = int(len(ss) * top_frac)
        if k < 2:
            k = 2
        if k > len(ss) // 2:
            k = len(ss) // 2
        if k < 1:
            continue
        order = ss.sort_values(kind='mergesort')
        shorts = order.index[:k]
        longs = order.index[-k:]
        r = p1 / p0 - 1.0
        try:
            lr = float(r.loc[longs].mean())
            sr = float(r.loc[shorts].mean())
        except Exception:
            continue
        if np.isfinite(lr) and np.isfinite(sr):
            vals.append(lr - sr)
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _compute_target(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)

    px = history.get('adjusted_close')
    if px is None or px.shape[1] == 0:
        px = history.get('close')
    if px is None or px.shape[1] == 0:
        return zeros
    try:
        px = px.reindex(columns=universe).astype(float)
    except Exception:
        return zeros

    n = px.shape[0]
    if n < 130:
        return zeros

    last = px.iloc[-1]
    valid = last.notna() & (last > 0.0) & (px.notna().sum() >= 120)
    if int(valid.sum()) < 10:
        return zeros

    ret = px / px.shift(1) - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)

    volm = history.get('volume')
    if volm is not None:
        try:
            volm = volm.reindex(columns=universe).reindex(index=px.index).astype(float)
            dollar = (px * volm).tail(60).median()
            dollar = dollar.reindex(universe).fillna(0.0)
            liq = dollar > 5e6
            if int(liq.sum()) >= 20:
                valid = valid & liq
        except Exception:
            pass

    el = history.get('eligible')
    if el is not None:
        try:
            el = el.reindex(columns=universe).reindex(index=px.index)
            el_last = el.ffill().iloc[-1]
            flag = el_last.fillna(1.0).astype(float) > 0.5
            if int(flag.sum()) >= 10:
                valid = valid & flag
        except Exception:
            pass

    if int(valid.sum()) < 10:
        return zeros

    mom12 = px.shift(21) / px.shift(252) - 1.0
    mom6 = px.shift(5) / px.shift(126) - 1.0
    r12 = _rank_df(mom12)
    r6 = _rank_df(mom6)
    mom_sig = r12.fillna(0.0) * 0.65 + r6.fillna(0.0) * 0.35
    mom_sig = mom_sig.where(r12.notna() | r6.notna())

    rev_raw = -(px / px.shift(21) - 1.0)
    rev_sig = _rank_df(rev_raw)

    if n >= 22:
        r_mom = _factor_return(px, mom_sig)
        r_rev = _factor_return(px, rev_sig)
    else:
        r_mom = 0.0
        r_rev = 0.0

    w_mom = float(np.tanh(r_mom / 0.01)) if np.isfinite(r_mom) else 0.0
    w_rev = float(np.tanh(r_rev / 0.01)) if np.isfinite(r_rev) else 0.0
    if abs(w_mom) + abs(w_rev) < 0.05:
        w_mom = 0.5
        w_rev = 0.0

    cur_mom = mom_sig.iloc[-1]
    cur_rev = rev_sig.iloc[-1]
    score = (w_mom * cur_mom.fillna(0.0)) + (w_rev * cur_rev.fillna(0.0))
    have_sig = cur_mom.notna() | cur_rev.notna()
    score = score.where(have_sig)
    score = score.replace([np.inf, -np.inf], np.nan)

    beta = pd.Series(0.0, index=universe, dtype=float)
    if n >= 60:
        win = min(120, n - 1)
        R = ret.tail(win)
        m = R.mean(axis=1)
        mc = m - m.mean()
        denom = float((mc ** 2).sum())
        if denom > 0:
            Rc = R.sub(R.mean(axis=0), axis=1)
            beta = Rc.mul(mc, axis=0).sum(axis=0) / denom
            beta = beta.reindex(universe)
            beta = beta.replace([np.inf, -np.inf], np.nan)
            beta_valid = beta.notna() & (R.notna().sum() >= max(20, win // 3))
            valid = valid & beta_valid

    cand = valid & score.notna() & beta.notna()
    if int(cand.sum()) < 10:
        return zeros

    b_rank = beta[cand].rank(pct=True)
    b_id = pd.Series(-1, index=universe, dtype=int)
    b_id.loc[cand] = np.minimum((b_rank * 5.0).astype(int), 4)
    counts = b_id[cand].value_counts()
    n_cand = int(cand.sum())
    target_side = min(50, n_cand // 2)
    if target_side < 2:
        return zeros

    quotas = {}
    for b in range(5):
        nb = int(counts.get(b, 0))
        q = int(round(target_side * nb / float(n_cand))) if n_cand > 0 else 0
        q = min(q, nb // 2)
        quotas[b] = max(0, q)

    remain = target_side - sum(quotas.values())
    if remain > 0:
        for b in sorted(range(5), key=lambda x: counts.get(x, 0), reverse=True):
            cap = int(counts.get(b, 0)) // 2 - quotas[b]
            if cap > 0:
                add = min(cap, remain)
                quotas[b] += add
                remain -= add
            if remain <= 0:
                break

    longs = []
    shorts = []
    for b in range(5):
        q = quotas.get(b, 0)
        if q <= 0:
            continue
        idx = cand & (b_id == b)
        if int(idx.sum()) < 2:
            continue
        s = score[idx].sort_values(kind='mergesort')
        take = min(q, len(s) // 2)
        if take < 1:
            continue
        shorts.extend(s.index[:take].tolist())
        longs.extend(s.index[-take:].tolist())

    n_side = len(longs)
    if n_side < 2 or len(shorts) != n_side:
        return zeros

    w = min(0.5 / float(n_side), 0.02)
    weights = zeros.copy()
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(float)
    weights = weights.clip(-0.02, 0.02)

    gross = float(weights.abs().sum())
    if gross > 1.0 + 1e-12:
        weights = weights * (1.0 / gross)

    net = float(weights.sum())
    if abs(net) > 0.01:
        weights = weights - net / max(1, len(universe))
        weights = weights.clip(-0.02, 0.02)

    return weights.astype(float)


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)

    px = history.get('adjusted_close')
    if px is None or px.shape[1] == 0:
        px = history.get('close')
    if px is None or px.shape[1] == 0:
        return zeros
    try:
        px = px.reindex(columns=universe)
    except Exception:
        return zeros
    if px.shape[0] < 130:
        return zeros

    last_date = px.index[-1]
    try:
        month_key = (last_date.year, last_date.month)
    except Exception:
        month_key = None

    if month_key is not None and _STATE.get('month') == month_key and _STATE.get('weights') is not None:
        cached_univ = _STATE.get('universe')
        if cached_univ is not None and cached_univ.equals(universe):
            w = _STATE['weights'].reindex(universe).fillna(0.0).astype(float)
            return w.clip(-0.02, 0.02)

    w = _compute_target(history, universe)
    if month_key is not None:
        _STATE['month'] = month_key
        _STATE['weights'] = w.copy()
        _STATE['universe'] = universe.copy()
    return w
