import numpy as np
import pandas as pd


def _centered_rank(s):
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return s
    return s.rank(method='average', pct=True) - 0.5


def generate_weights(history, universe):
    tickers = pd.Index(list(universe))
    if len(tickers) == 0:
        return pd.Series(dtype=float)

    zeros = pd.Series(0.0, index=tickers)
    px_all = history.get('adjusted_close')
    if px_all is None:
        return zeros

    idx = px_all.index
    if len(idx) < 2:
        return zeros

    current = idx[-1]
    month_start = pd.Timestamp(year=current.year, month=current.month, day=1)
    prior_pos = idx.searchsorted(month_start, side='left') - 1
    if prior_pos < 0:
        return zeros

    ref_pos = int(prior_pos)
    if ref_pos < 273:
        return zeros

    start = ref_pos - 252
    window = px_all.iloc[start:ref_pos + 1].reindex(columns=tickers).astype(float)
    if window.shape[0] < 253:
        return zeros

    p_ref = window.iloc[-1]
    p_skip = window.iloc[-22]
    p12 = window.iloc[0]
    p6 = window.iloc[126]

    mom12 = p_skip / p12 - 1.0
    mom6 = p_skip / p6 - 1.0

    rets = window.pct_change()
    vol = rets.iloc[-126:].std()
    vol = vol.replace([np.inf, -np.inf], np.nan)

    valid = (
        p_ref.notna() & (p_ref > 0)
        & p_skip.notna() & (p_skip > 0)
        & p12.notna() & (p12 > 0)
        & p6.notna() & (p6 > 0)
        & np.isfinite(mom12) & np.isfinite(mom6)
        & vol.notna() & (vol > 0)
    )

    ref_date = idx[ref_pos]
    elig_all = history.get('eligible')
    if elig_all is not None:
        if ref_date in elig_all.index:
            e_row = elig_all.loc[ref_date].reindex(tickers)
        else:
            e_row = pd.Series(False, index=tickers)
        valid = valid & e_row.fillna(False).astype(bool)

    vol_all = history.get('volume')
    if vol_all is not None:
        vstart = max(0, ref_pos - 59)
        v_win = vol_all.iloc[vstart:ref_pos + 1].reindex(columns=tickers).astype(float)
        medv = v_win.median()
        valid = valid & (medv.fillna(0) > 100000.0)

    valid_t = valid[valid].index
    if len(valid_t) < 10:
        return zeros

    r12 = _centered_rank(mom12.reindex(valid_t))
    r6 = _centered_rank(mom6.reindex(valid_t))
    rv = _centered_rank(vol.reindex(valid_t))

    score = 0.5 * r12 + 0.5 * r6 - 0.25 * rv
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 10:
        return zeros

    n_side = min(25, len(score) // 2)
    if n_side < 5:
        return zeros

    order = score.sort_values(ascending=False)
    longs = order.index[:n_side]
    shorts = order.index[-n_side:]

    w = min(0.02, 0.5 / float(n_side))
    weights = zeros.copy()
    weights.loc[longs] = w
    weights.loc[shorts] = -w

    gross = float(weights.abs().sum())
    if gross > 1.0:
        weights = weights * (1.0 / gross)

    return weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)