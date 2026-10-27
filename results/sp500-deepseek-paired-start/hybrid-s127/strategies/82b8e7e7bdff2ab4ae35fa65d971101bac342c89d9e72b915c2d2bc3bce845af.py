import numpy as np
import pandas as pd


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zero = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return zero

    ac = history.get('adjusted_close')
    if not isinstance(ac, pd.DataFrame) or ac.empty:
        ac = history.get('close')
    if not isinstance(ac, pd.DataFrame) or ac.empty:
        return zero

    cols = [c for c in universe if c in ac.columns]
    if len(cols) < 8:
        return zero

    n = len(ac)
    if n < 160:
        return zero

    rebal = 21
    apos = ((n - 1) // rebal) * rebal
    if apos < 150:
        return zero

    raw = ac.iloc[:apos + 1][cols]
    px = raw.ffill()
    if len(px) < 130:
        return zero

    raw_last = raw.iloc[-1].to_numpy(dtype=float)
    last = px.iloc[-1].to_numpy(dtype=float)
    valid = pd.Series(
        np.isfinite(raw_last) & (raw_last > 0.0)
        & np.isfinite(last) & (last > 0.0),
        index=cols,
    )

    elig = history.get('eligible')
    if isinstance(elig, pd.DataFrame) and not elig.empty:
        sub = elig.loc[:px.index[-1]]
        if len(sub) > 0:
            e = sub.iloc[-1].reindex(cols).fillna(False).astype(bool)
            valid = valid & e

    vol_df = history.get('volume')
    close_df = history.get('close')
    if isinstance(vol_df, pd.DataFrame) and isinstance(close_df, pd.DataFrame):
        v = vol_df.reindex(index=px.index, columns=cols).astype(float)
        cp = close_df.reindex(index=px.index, columns=cols).astype(float)
        dollar = (v * cp).tail(60)
        med = dollar.median(axis=0)
        med_np = med.to_numpy(dtype=float)
        liq = pd.Series(
            np.isfinite(med_np) & (med_np >= 1000000.0),
            index=cols,
        )
        valid = valid & liq

    rets = px.pct_change()
    vola = rets.tail(120).std(axis=0)
    vola = vola.replace([np.inf, -np.inf], np.nan)
    vol_np = vola.to_numpy(dtype=float)
    valid = valid & pd.Series(
        np.isfinite(vol_np) & (vol_np > 0.0),
        index=cols,
    )

    if int(valid.sum()) < 8:
        return zero

    mkt_ret = rets.mean(axis=1)
    mkt_ret = mkt_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mkt_level = (1.0 + mkt_ret).cumprod()
    win = min(200, len(mkt_level))
    if win < 60:
        return zero
    ma = float(mkt_level.tail(win).mean())
    last_mkt = float(mkt_level.iloc[-1])
    risk_on = last_mkt > ma

    f = pd.DataFrame({'vol': vola}).loc[valid].dropna()
    f = f[f['vol'] > 0.0]
    if len(f) < 8:
        return zero

    score = f['vol'].rank(pct=True, method='average')
    if not risk_on:
        score = -score
    score = score.sort_values(ascending=False)

    n_side = int(min(60, len(score) // 2))
    if n_side < 1:
        return zero

    longs = score.index[:n_side]
    shorts = score.index[-n_side:]

    out = zero.copy()

    def _leg_weights(names):
        vv = f.loc[names, 'vol'].to_numpy(dtype=float)
        inv = 1.0 / np.maximum(vv, 1e-8)
        return pd.Series(0.5 * inv / inv.sum(), index=names)

    lw = _leg_weights(longs)
    sw = _leg_weights(shorts)
    out.loc[lw.index] = lw.to_numpy(dtype=float)
    out.loc[sw.index] = -sw.to_numpy(dtype=float)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = out.clip(-0.02, 0.02)

    gross = float(out.abs().sum())
    if gross > 1.0 and gross > 0.0:
        out = out * (1.0 / gross)

    for _ in range(6):
        net = float(out.sum())
        if abs(net) <= 1e-8:
            break
        active = out[out != 0.0]
        if len(active) == 0:
            break
        out.loc[active.index] = out.loc[active.index] - net / float(len(active))
        out = out.clip(-0.02, 0.02)
        gross = float(out.abs().sum())
        if gross > 1.0 and gross > 0.0:
            out = out * (1.0 / gross)

    return out.astype(float)