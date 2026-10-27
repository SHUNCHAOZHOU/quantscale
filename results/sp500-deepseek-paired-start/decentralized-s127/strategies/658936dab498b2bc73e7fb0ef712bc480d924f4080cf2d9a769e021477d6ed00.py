import numpy as np
import pandas as pd


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero
    close_df = history.get('adjusted_close')
    if close_df is None:
        close_df = history.get('close')
    if close_df is None or close_df.empty:
        return zero
    cols = [c for c in universe if c in close_df.columns]
    if len(cols) < 4:
        return zero
    n_total = len(close_df.index)
    if n_total < 70:
        return zero
    rebalance = 21
    p = n_total - 1
    apos = (p // rebalance) * rebalance
    if apos < 60:
        return zero
    anchor_date = close_df.index[apos]
    prices = close_df.loc[:anchor_date, cols].ffill()
    if prices.shape[0] < 60:
        return zero
    last = prices.iloc[-1]
    elig_df = history.get('eligible')
    if elig_df is not None and not elig_df.empty:
        e_sub = elig_df.loc[:anchor_date]
        if len(e_sub) == 0:
            return zero
        e_last = e_sub.iloc[-1].reindex(cols)
        elig = e_last.fillna(False).astype(bool)
    else:
        elig = pd.Series(True, index=cols)
    valid = elig & last.notna() & np.isfinite(last) & (last > 0.0)
    vol_df = history.get('volume')
    if vol_df is not None and not vol_df.empty:
        v = vol_df.reindex(index=prices.index, columns=cols).ffill()
        dollar = v * prices
        med_dv = dollar.tail(60).median(skipna=True)
        valid = valid & med_dv.notna() & (med_dv >= 1000000.0)
    rets = prices.pct_change()
    if len(prices) > 21:
        r21 = prices.iloc[-1] / prices.iloc[-22] - 1.0
    else:
        return zero
    vol60 = rets.tail(60).std()
    valid = valid & r21.notna() & vol60.notna() & (vol60 > 0.0)
    if int(valid.sum()) < 4:
        return zero
    f = pd.DataFrame({'r21': r21, 'vol': vol60}).loc[valid]
    rev_rank = (-f['r21']).rank(pct=True, method='average')
    lowvol_rank = (-f['vol']).rank(pct=True, method='average')
    score = 0.70 * rev_rank + 0.30 * lowvol_rank
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 4:
        return zero
    n_side = min(50, len(score) // 2)
    if n_side < 2:
        return zero
    ranked = score.sort_values(ascending=False)
    longs = ranked.index[:n_side]
    shorts = ranked.index[-n_side:]
    w = min(0.02, 0.5 / float(n_side))
    out = zero.copy()
    out.loc[longs] = w
    out.loc[shorts] = -w
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    gross = float(out.abs().sum())
    if gross > 1.0 and gross > 0.0:
        out = out * (1.0 / gross)
    net = float(out.sum())
    if abs(net) > 1e-12:
        nz = out != 0.0
        if nz.any():
            out.loc[nz] = out.loc[nz] - net / float(nz.sum())
            out = out.clip(-0.02, 0.02)
            gross = float(out.abs().sum())
            if gross > 1.0 and gross > 0.0:
                out = out * (1.0 / gross)
    return out.astype(float)