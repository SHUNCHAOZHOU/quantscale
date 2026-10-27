import numpy as np
import pandas as pd


def _get_panel(history, field):
    df = history.get(field)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return None


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    ac = _get_panel(history, 'adjusted_close')
    if ac is None:
        ac = _get_panel(history, 'close')
    if ac is None:
        return zero

    dates = ac.index
    n = len(dates)
    rebalance = 21
    min_obs = 150
    apos = ((n - 1) // rebalance) * rebalance
    if apos < min_obs:
        return zero
    anchor_date = dates[apos]

    cols = [c for c in universe if c in ac.columns]
    if len(cols) < 8:
        return zero

    raw = ac.loc[:anchor_date, cols]
    prices = raw.ffill()
    if prices.shape[0] < 60:
        return zero

    raw_last = raw.iloc[-1]
    last = prices.iloc[-1]

    elig_df = _get_panel(history, 'eligible')
    if elig_df is not None:
        sub = elig_df.loc[:anchor_date]
        if len(sub) > 0:
            e_raw = sub.iloc[-1].reindex(cols)
            e = e_raw.fillna(False).astype(bool)
        else:
            e = pd.Series(True, index=cols)
    else:
        e = pd.Series(True, index=cols)

    valid = (
        e
        & raw_last.notna()
        & np.isfinite(raw_last.astype(float))
        & (raw_last.astype(float) > 0.0)
        & last.notna()
        & np.isfinite(last.astype(float))
        & (last.astype(float) > 0.0)
    )

    vol_df = _get_panel(history, 'volume')
    close_df = _get_panel(history, 'close')
    if vol_df is not None and close_df is not None:
        v = vol_df.loc[:anchor_date, cols].reindex(index=prices.index, columns=cols)
        px = close_df.loc[:anchor_date, cols].reindex(index=prices.index, columns=cols)
        dollar = (v.astype(float) * px.astype(float)).tail(60)
        with np.errstate(all='ignore'):
            med_dv = dollar.median(axis=0)
        valid = valid & med_dv.notna() & (med_dv >= 1000000.0)

    if int(valid.sum()) < 8:
        return zero

    rets = prices.astype(float).pct_change()
    with np.errstate(all='ignore'):
        vol = rets.rolling(120, min_periods=60).std().iloc[-1]

    f = pd.DataFrame({'vol': vol}).reindex(cols)
    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.loc[valid].dropna(subset=['vol'])
    f = f[f['vol'] > 0.0]
    if len(f) < 8:
        return zero

    vol_rank = f['vol'].rank(pct=True, method='average')
    score = vol_rank.sort_values(ascending=False)

    n_side = min(50, len(score) // 2)
    if n_side < 1:
        return zero

    longs = score.index[:n_side]
    shorts = score.index[-n_side:]
    side = min(0.02, 0.5 / float(n_side))

    out = zero.copy()
    out.loc[longs] = side
    out.loc[shorts] = -side
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    gross = float(out.abs().sum())
    if gross > 1.0 and gross > 0.0:
        out = out * (1.0 / gross)

    net = float(out.sum())
    if abs(net) > 0.01:
        active = out[out != 0.0]
        if len(active) > 0:
            adj = net / float(len(active))
            out.loc[active.index] = out.loc[active.index] - adj
            out = out.clip(-0.02, 0.02)
            gross = float(out.abs().sum())
            if gross > 1.0 and gross > 0.0:
                out = out * (1.0 / gross)

    return out.astype(float)