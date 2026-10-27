import math
import numpy as np
import pandas as pd


def _zeros(universe):
    return pd.Series(0.0, index=pd.Index(universe), dtype=float)


def _price_frame(history, universe):
    for name in ('adjusted_close', 'close'):
        df = history.get(name)
        if isinstance(df, pd.DataFrame) and df.shape[0] > 0 and df.shape[1] > 0:
            return df.reindex(columns=universe).astype(float)
    return None


def _eligible(history, universe, base_pos):
    elig_df = history.get('eligible')
    if isinstance(elig_df, pd.DataFrame) and elig_df.shape[0] > 0:
        try:
            e = elig_df.reindex(columns=universe).ffill()
            if base_pos >= e.shape[0]:
                row = e.iloc[-1]
            else:
                row = e.iloc[base_pos]
            return row.fillna(False).astype(bool)
        except Exception:
            pass
    return pd.Series(True, index=universe, dtype=bool)


def _soft_neutral_weights(signal, valid, universe):
    weights = _zeros(universe)
    if signal is None:
        return weights
    s = signal.reindex(universe).astype(float)
    v = valid.reindex(universe).fillna(False).astype(bool)
    s = s.where(v)
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.dropna()
    if len(s) < 4:
        return weights
    s = s.clip(-4.0, 4.0)
    s = s - s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return weights
    s = s / sd
    s = np.tanh(s / 1.5)
    s = s.where(s.abs() >= 0.10, 0.0)
    pos = s.clip(lower=0.0)
    neg = (-s).clip(lower=0.0)
    if pos.sum() <= 0.0 or neg.sum() <= 0.0:
        return weights
    long_w = pos / pos.sum() * 0.5
    short_w = -neg / neg.sum() * 0.5
    weights.loc[long_w.index] = long_w.values
    weights.loc[short_w.index] = short_w.values
    weights = weights.clip(-0.02, 0.02)
    lg = float(weights[weights > 0.0].sum())
    sg = float(-weights[weights < 0.0].sum())
    target = min(lg, sg, 0.5)
    if target <= 0.0:
        return _zeros(universe)
    if lg > 0.0:
        pos_mask = weights > 0.0
        weights.loc[pos_mask] = weights.loc[pos_mask] * (target / lg)
    if sg > 0.0:
        neg_mask = weights < 0.0
        weights.loc[neg_mask] = weights.loc[neg_mask] * (target / sg)
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    gross = float(weights.abs().sum())
    if gross > 1.0 + 1e-12:
        weights = weights / gross
    return weights.astype(float)


def generate_weights(history, universe):
    universe = pd.Index(list(universe))
    weights = _zeros(universe)
    if len(universe) == 0:
        return weights

    raw = _price_frame(history, universe)
    if raw is None:
        return weights
    raw = raw.ffill()
    if raw.shape[0] < 3:
        return weights

    last_pos = raw.shape[0] - 1
    base_pos = last_pos - (last_pos % 5)
    if base_pos < 2:
        base_pos = last_pos

    px_all = raw.iloc[:base_pos + 1]
    if px_all.shape[0] < 64:
        if px_all.shape[0] < 25:
            return weights
        L = min(20, px_all.shape[0] - 1)
        sig = px_all.iloc[-1] / px_all.iloc[-1 - L] - 1.0
        sig = sig.replace([np.inf, -np.inf], np.nan)
        elig = _eligible(history, universe, base_pos)
        valid = elig & sig.notna() & np.isfinite(sig) & (px_all.iloc[-1] > 0.0)
        return _soft_neutral_weights(sig, valid, universe)

    px = px_all.tail(320)
    last = px.iloc[-1]
    elig = _eligible(history, universe, base_pos)
    base_valid = last.notna() & np.isfinite(last) & (last > 0.0) & elig
    if not base_valid.any():
        return weights

    ret = px.pct_change()
    ret = ret.replace([np.inf, -np.inf], np.nan)
    vol = ret.rolling(63, min_periods=20).std(ddof=0)
    vol = vol.where(vol > 1e-8)

    lookbacks = [63, 126, 252]
    lb_weights = [0.35, 0.35, 0.30]
    comps = []
    for L, wt in zip(lookbacks, lb_weights):
        if px.shape[0] <= L + 1:
            continue
        mom = px / px.shift(L) - 1.0
        mom = mom.replace([np.inf, -np.inf], np.nan)
        comp = mom / (vol * math.sqrt(float(L)))
        comp = comp.replace([np.inf, -np.inf], np.nan)
        mu = comp.mean(axis=1, skipna=True)
        sd = comp.std(axis=1, ddof=0, skipna=True)
        sd = sd.where(sd > 1e-12)
        z = comp.sub(mu, axis=0).div(sd, axis=0)
        z = z.replace([np.inf, -np.inf], np.nan)
        z_s = z.tail(10).mean(axis=0, skipna=True)
        comps.append(z_s * wt)

    if comps:
        signal = pd.concat(comps, axis=1).sum(axis=1, min_count=1)
    else:
        L = min(20, px.shape[0] - 1)
        signal = px.iloc[-1] / px.iloc[-1 - L] - 1.0
        signal = signal.replace([np.inf, -np.inf], np.nan)

    signal = signal.reindex(universe)

    liquid_ok = pd.Series(True, index=universe)
    vol_df = history.get('volume')
    if isinstance(vol_df, pd.DataFrame) and vol_df.shape[0] > 0:
        try:
            vv = vol_df.reindex(index=raw.index, columns=universe).astype(float).ffill()
            start = max(0, base_pos - 19)
            dollar = raw.iloc[start:base_pos + 1] * vv.iloc[start:base_pos + 1]
            liq = dollar.median(axis=0, skipna=True)
            liq_ok = liq.notna() & np.isfinite(liq) & (liq > 0.0)
            if int(liq_ok.sum()) >= max(10, len(universe) // 2):
                q20 = liq[liq_ok].quantile(0.2)
                liquid_ok = liq_ok & (liq >= q20)
        except Exception:
            liquid_ok = pd.Series(True, index=universe)

    valid = (
        base_valid &
        signal.notna() &
        np.isfinite(signal) &
        liquid_ok.reindex(universe).fillna(False)
    )
    return _soft_neutral_weights(signal, valid, universe)
