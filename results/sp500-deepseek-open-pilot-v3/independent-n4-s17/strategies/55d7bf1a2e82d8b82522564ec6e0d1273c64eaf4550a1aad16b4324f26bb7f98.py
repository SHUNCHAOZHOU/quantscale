import numpy as np
import pandas as pd

def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero
    ac = history.get('adjusted_close')
    if ac is None:
        ac = history.get('close')
    if ac is None:
        return zero
    ac = ac.reindex(columns=universe)
    vol = history.get('volume')
    if vol is not None:
        vol = vol.reindex(columns=universe)
    n = len(ac)
    if n < 3:
        return zero
    p_last = ac.iloc[-1]
    ret = ac.pct_change()
    if n >= 253:
        mom12 = ac.iloc[-22] / ac.iloc[-253] - 1.0
    elif n >= 23:
        mom12 = ac.iloc[-22] / ac.iloc[0] - 1.0
    else:
        mom12 = pd.Series(np.nan, index=universe)
    if n >= 127:
        mom6 = ac.iloc[-22] / ac.iloc[-127] - 1.0
    elif n >= 23:
        mom6 = ac.iloc[-22] / ac.iloc[0] - 1.0
    else:
        mom6 = pd.Series(np.nan, index=universe)
    if n >= 22:
        rev1 = p_last / ac.iloc[-22] - 1.0
    else:
        rev1 = p_last / ac.iloc[0] - 1.0
    vol60 = ret.iloc[-60:].std() if n >= 20 else ret.std()
    if vol is not None:
        dv = (ac * vol).iloc[-20:].mean()
    else:
        dv = pd.Series(np.inf, index=universe)
    valid = (p_last.notna() & mom12.notna() & mom6.notna() & rev1.notna() &
             vol60.notna() & (vol60 > 0) & dv.notna() & (dv > 0))
    elig = history.get('eligible')
    if elig is not None:
        elig = elig.reindex(columns=universe)
        if len(elig) > 0:
            valid = valid & elig.iloc[-1].fillna(False).astype(bool)
    if valid.sum() < 2:
        return zero
    def rp(s):
        return s.where(valid).rank(pct=True)
    r_mom12 = rp(mom12)
    r_mom6 = rp(mom6)
    r_vol = rp(vol60)
    r_rev = rp(rev1)
    r_liq = rp(dv)
    score = 0.50 * r_mom12 + 0.30 * r_mom6 - 0.30 * r_vol - 0.10 * r_rev + 0.05 * r_liq
    score = score.where(valid)
    s = score.dropna()
    if len(s) < 2:
        return zero
    max_w = 0.02
    max_names = int(np.floor(0.5 / max_w))
    order = s.sort_values(ascending=False)
    n_side = min(max_names, len(order) // 2)
    if n_side < 1:
        return zero
    long_names = order.index[:n_side]
    short_names = order.index[-n_side:]
    w_side = min(max_w, 0.5 / n_side)
    weights = pd.Series(0.0, index=universe, dtype=float)
    weights.loc[long_names] = w_side
    weights.loc[short_names] = -w_side
    weights = weights.clip(-max_w, max_w)
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights