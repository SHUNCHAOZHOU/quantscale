import numpy as np
import pandas as pd

def generate_weights(history, universe):
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=universe, dtype=float)
    ac = history.get('adjusted_close')
    if ac is None or ac.empty:
        return zero
    ac = ac.reindex(columns=universe)
    if ac.empty:
        return zero
    last_date = ac.index.max()
    try:
        first = pd.Timestamp(year=last_date.year, month=last_date.month, day=1)
        cutoff = first - pd.Timedelta(days=1)
    except Exception:
        cutoff = last_date
    ac_cut = ac.loc[ac.index <= cutoff]
    if len(ac_cut) < 100:
        ac_cut = ac.loc[ac.index <= last_date]
        cutoff = last_date
    if len(ac_cut) < 60:
        return zero
    vol_cut = None
    close_cut = None
    vol = history.get('volume')
    close = history.get('close')
    if vol is not None and close is not None:
        try:
            vol_cut = vol.reindex(index=ac_cut.index, columns=universe)
            close_cut = close.reindex(index=ac_cut.index, columns=universe)
        except Exception:
            vol_cut = None
            close_cut = None
    n = len(ac_cut)
    records = {}
    for t in universe:
        raw = ac_cut[t].values.astype(float)
        finite_raw = np.isfinite(raw)
        if not finite_raw.any():
            continue
        valid_count = int(finite_raw.sum())
        if valid_count < 120:
            continue
        last_idx = int(np.where(finite_raw)[0][-1])
        if last_idx < n - 8:
            continue
        if last_idx < 126:
            continue
        p = pd.Series(np.where(finite_raw, raw, np.nan)).ffill().values
        if not np.isfinite(p[last_idx]) or p[last_idx] <= 1.0:
            continue
        i21 = last_idx - 21
        i126 = last_idx - 126
        i252 = last_idx - 252
        if not np.isfinite(p[i21]) or p[i21] <= 0:
            continue
        if i252 >= 0 and np.isfinite(p[i252]) and p[i252] > 0:
            mom12 = p[i21] / p[i252] - 1.0
        else:
            mom12 = np.nan
        if i126 >= 0 and np.isfinite(p[i126]) and p[i126] > 0:
            mom6 = p[i21] / p[i126] - 1.0
        else:
            mom6 = np.nan
        rev21 = p[last_idx] / p[i21] - 1.0
        start = max(0, last_idx - 60)
        seg = p[start:last_idx+1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 21:
            rets = np.diff(seg) / seg[:-1]
            rets = rets[np.isfinite(rets)]
            if len(rets) >= 10:
                vol60 = float(np.std(rets) * np.sqrt(252.0))
            else:
                vol60 = np.nan
        else:
            vol60 = np.nan
        start_high = max(0, last_idx - 252)
        high_seg = p[start_high:last_idx+1]
        high_seg = high_seg[np.isfinite(high_seg)]
        if len(high_seg) > 0:
            high_max = float(np.max(high_seg))
            high_prox = p[last_idx] / high_max if high_max > 0 else np.nan
        else:
            high_prox = np.nan
        med_dv = np.nan
        if vol_cut is not None and close_cut is not None:
            v = vol_cut[t].values.astype(float)
            c = close_cut[t].values.astype(float)
            dv = v * c
            dv_seg = dv[max(0, last_idx-60):last_idx+1]
            dv_seg = dv_seg[np.isfinite(dv_seg)]
            if len(dv_seg) >= 10:
                med_dv = float(np.median(dv_seg))
        records[t] = (mom12, mom6, rev21, vol60, high_prox, med_dv)
    if not records:
        return zero
    df = pd.DataFrame.from_dict(records, orient='index', columns=['mom12','mom6','rev21','vol60','high_prox','med_dv'])
    df = df.replace([np.inf, -np.inf], np.nan)
    df['mom12'] = df['mom12'].fillna(df['mom6'])
    df['mom6'] = df['mom6'].fillna(df['mom12'])
    base = df[['mom12','mom6','rev21','vol60']].notna().all(axis=1)
    liquid = base & (df['med_dv'] >= 5e6)
    if liquid.sum() < 100:
        liquid = base & (df['med_dv'] >= 1e6)
    if liquid.sum() < 60:
        liquid = base
    df = df[liquid]
    if len(df) < 10:
        return zero
    eligible = history.get('eligible')
    if eligible is not None:
        try:
            elig_last = eligible.reindex(columns=universe).loc[:last_date].iloc[-1]
            elig_last = elig_last.reindex(df.index).fillna(False).astype(bool)
            if int(elig_last.sum()) >= 10:
                df = df[elig_last]
        except Exception:
            pass
    if len(df) < 10:
        return zero
    df['high_prox'] = df['high_prox'].fillna(1.0)
    mom12_rank = df['mom12'].rank(pct=True)
    mom6_rank = df['mom6'].rank(pct=True)
    rev_rank = df['rev21'].rank(pct=True)
    vol_rank = df['vol60'].rank(pct=True)
    high_rank = df['high_prox'].rank(pct=True)
    score = (0.40 * mom12_rank + 0.20 * mom6_rank + 0.10 * high_rank - 0.20 * vol_rank - 0.10 * rev_rank)
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 10:
        return zero
    N_target = 60
    N = min(N_target, len(score) // 2)
    if N < 1:
        return zero
    sorted_score = score.sort_values(ascending=False)
    longs = sorted_score.index[:N]
    shorts = sorted_score.index[-N:]
    w = pd.Series(0.0, index=universe, dtype=float)
    if N >= 25:
        wl = 0.5 / N
        ws = -wl
    else:
        wl = min(0.5 / N, 0.02)
        ws = -wl
    w.loc[longs] = wl
    w.loc[shorts] = ws
    w = w.reindex(universe).fillna(0.0)
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    gross = float(w.abs().sum())
    if gross > 1.0:
        w = w * (1.0 / gross)
    w = w.clip(-0.02, 0.02)
    net = float(w.sum())
    if abs(net) > 0.01:
        nz = w[w != 0]
        if len(nz) > 0:
            corr = net / len(nz)
            w.loc[nz.index] = w.loc[nz.index] - corr
            w = w.clip(-0.02, 0.02)
            gross = float(w.abs().sum())
            if gross > 1.0:
                w = w * (1.0 / gross)
    return w.astype(float)