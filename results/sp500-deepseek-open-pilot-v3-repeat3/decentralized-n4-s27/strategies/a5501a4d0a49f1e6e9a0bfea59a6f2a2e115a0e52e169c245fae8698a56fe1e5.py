import numpy as np
import pandas as pd


def _zero_weights(universe):
    return pd.Series(0.0, index=pd.Index(universe), dtype=float)


def _window_sum(arr, k, min_count):
    if arr.shape[0] == 0:
        return np.full(arr.shape[1], np.nan)
    sub = arr[-k:]
    cnt = np.sum(np.isfinite(sub), axis=0)
    val = np.nansum(sub, axis=0)
    val[cnt < min_count] = np.nan
    return val


def generate_weights(history, universe):
    cols = list(universe)
    if len(cols) == 0:
        return _zero_weights(cols)

    px_df = history.get('adjusted_close')
    if px_df is None:
        px_df = history.get('close')
    if px_df is None or not isinstance(px_df, pd.DataFrame):
        return _zero_weights(cols)

    # Restrict to the supplied universe and recent history only.
    px_df = px_df.reindex(index=px_df.index, columns=cols).tail(300)
    if px_df.shape[0] < 3:
        return _zero_weights(cols)

    p = np.array(px_df, dtype=float, copy=True)
    n, m = p.shape

    # Daily simple returns and cross-sectional relative returns.
    with np.errstate(divide='ignore', invalid='ignore'):
        rets = p[1:] / p[:-1] - 1.0
    rets = np.where(np.isfinite(rets), rets, np.nan)

    with np.errstate(invalid='ignore'):
        market = np.nanmean(rets, axis=1)
    rel = rets - market[:, None]

    # Residual momentum: sum of relative returns over roughly six months,
    # skipping the most recent month to avoid short-term reversal.
    if rel.shape[0] > 21:
        if rel.shape[0] >= 147:
            mom_sub = rel[-147:-21]
        else:
            mom_sub = rel[:-21]
        min_mom = min(60, max(10, mom_sub.shape[0] // 2))
        mom = _window_sum(mom_sub, mom_sub.shape[0], min_mom)
    else:
        mom = np.full(m, np.nan)

    # Short-term relative reversal over the last five sessions.
    rev = _window_sum(rel, 5, 3)

    # Residual volatility over the last sixty sessions.
    vol_sub = rel[-60:]
    if vol_sub.shape[0] >= 20:
        vol_cnt = np.sum(np.isfinite(vol_sub), axis=0)
        vol = np.nanstd(vol_sub, axis=0)
        vol[vol_cnt < 20] = np.nan
    else:
        vol = np.full(m, np.nan)

    # Liquidity: median dollar volume over the last sixty sessions.
    v_df = history.get('volume')
    if v_df is not None and isinstance(v_df, pd.DataFrame):
        v_df = v_df.reindex(index=px_df.index, columns=cols)
        v = np.array(v_df, dtype=float, copy=True)
        with np.errstate(invalid='ignore'):
            dollar = p * v
        liq = np.nanmedian(dollar[-60:], axis=0)
    else:
        liq = np.ones(m, dtype=float)

    valid = (
        np.isfinite(p[-1])
        & np.isfinite(mom)
        & np.isfinite(rev)
        & np.isfinite(vol)
        & (vol > 1e-8)
        & np.isfinite(liq)
        & (liq > 0)
    )
    if int(np.sum(valid)) < 2:
        return _zero_weights(cols)

    idx = pd.Index(cols)
    mom_s = pd.Series(mom, index=idx)[valid]
    rev_s = pd.Series(rev, index=idx)[valid]
    vol_s = pd.Series(vol, index=idx)[valid]
    liq_s = pd.Series(liq, index=idx)[valid]

    # Risk-scaled ranks. High momentum is long; high recent reversal is short;
    # high volatility and low liquidity receive small penalties.
    rm = (mom_s / vol_s).rank(pct=True, method='average')
    rr = (rev_s / vol_s).rank(pct=True, method='average')
    rv = vol_s.rank(pct=True, method='average')
    rl = liq_s.rank(pct=True, method='average')

    score = 0.60 * rm - 0.30 * rr - 0.10 * rv + 0.05 * rl
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if score.shape[0] < 2:
        return _zero_weights(cols)

    n_side = min(50, score.shape[0] // 2)
    if n_side < 1:
        return _zero_weights(cols)

    longs = score.nlargest(n_side).index
    shorts = score.nsmallest(n_side).index

    # Equal-weight dollar-neutral book.  Cap at 2% per name and gross at 1.0.
    weight = min(0.02, 1.0 / (2.0 * n_side))

    w = _zero_weights(cols)
    w.loc[longs] = weight
    w.loc[shorts] = -weight
    w = w.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return w
