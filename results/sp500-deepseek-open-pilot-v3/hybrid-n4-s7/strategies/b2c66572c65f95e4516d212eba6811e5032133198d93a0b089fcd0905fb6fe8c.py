import numpy as np
import pandas as pd
import math


def _get_field(history, name):
    df = history.get(name)
    if df is None or not isinstance(df, pd.DataFrame):
        return None
    return df


def generate_weights(history, universe):
    universe = pd.Index(universe)
    weights = pd.Series(0.0, index=universe, dtype=float)
    if len(universe) == 0:
        return weights

    prices = _get_field(history, 'adjusted_close')
    if prices is None:
        prices = _get_field(history, 'close')
    if prices is None or prices.empty:
        return weights

    cols = [t for t in universe if t in prices.columns]
    if len(cols) < 2:
        return weights

    px = prices.loc[:, cols].tail(140).astype(float)
    if len(px) < 3:
        return weights

    elig = _get_field(history, 'eligible')
    elig_last = None
    if elig is not None and not elig.empty:
        ecols = [t for t in cols if t in elig.columns]
        if ecols:
            e = elig.loc[:, ecols].tail(1)
            elig_last = e.iloc[-1].reindex(cols)

    sig_px = px.tail(30)
    last = sig_px.iloc[-1]

    if len(sig_px) >= 6:
        ret5 = last / sig_px.iloc[-6] - 1.0
    else:
        ret5 = last / sig_px.iloc[0] - 1.0

    ret20 = None
    if len(sig_px) >= 21:
        ret20 = last / sig_px.iloc[-21] - 1.0

    mask = last.notna() & (last > 0) & ret5.notna()
    mask &= np.isfinite(ret5)
    if elig_last is not None:
        em = elig_last.fillna(True).astype(bool)
        mask &= em

    candidates = [t for t in cols if bool(mask.get(t, False))]
    if len(candidates) < 2:
        return weights

    r5 = ret5.reindex(candidates).astype(float)
    rank5 = r5.rank(method='average', pct=True)

    if ret20 is not None:
        r20 = ret20.reindex(candidates).astype(float)
        rank20 = r20.rank(method='average', pct=True).fillna(0.5)
        score = -(0.75 * rank5 + 0.25 * rank20)
    else:
        score = -rank5

    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < 2:
        return weights

    rets = px.pct_change()
    valid_counts = rets.notna().sum(axis=1)
    disp = rets.std(axis=1, ddof=0)
    breadth = (rets > 0).sum(axis=1) / valid_counts.replace(0, np.nan)
    mkt = rets.mean(axis=1)

    cur_disp = disp.iloc[-1]
    cur_breadth = breadth.iloc[-1]

    if not np.isfinite(cur_disp):
        cur_disp = disp.dropna().iloc[-1] if disp.notna().any() else np.nan
    if not np.isfinite(cur_breadth):
        cur_breadth = breadth.dropna().iloc[-1] if breadth.notna().any() else 0.5

    if np.isfinite(cur_disp):
        hist_disp = disp.iloc[:-1].dropna().tail(120)
        if len(hist_disp) >= 20:
            disp_pct = float((hist_disp <= cur_disp).mean())
        else:
            disp_pct = 0.5
    else:
        disp_pct = 0.5

    mkt_vol_series = mkt.rolling(20, min_periods=10).std(ddof=0)
    cur_mvol = mkt_vol_series.iloc[-1]
    if np.isfinite(cur_mvol):
        hist_mvol = mkt_vol_series.iloc[:-1].dropna().tail(120)
        if len(hist_mvol) >= 20:
            mvol_pct = float((hist_mvol <= cur_mvol).mean())
        else:
            mvol_pct = 0.5
    else:
        mvol_pct = 0.5

    if not np.isfinite(cur_breadth):
        cur_breadth = 0.5

    risk_off = (cur_breadth < 0.25) or (mvol_pct >= 0.90 and cur_breadth < 0.40)
    high_disp = disp_pct >= 0.65
    moderate_breadth = (0.35 <= cur_breadth <= 0.70)

    if risk_off:
        gross_target = 0.25
    elif high_disp or moderate_breadth:
        gross_target = 1.0
    elif disp_pct >= 0.45 and (0.25 <= cur_breadth <= 0.80):
        gross_target = 0.60
    else:
        gross_target = 0.35

    valid_n = len(score)
    max_side = valid_n // 2
    if max_side < 1:
        return weights

    desired = int(math.ceil(0.5 * gross_target / 0.02)) if gross_target > 0 else 0
    desired = max(1, desired)
    n_side = min(25, desired, max_side)
    if n_side < 1:
        return weights

    gross = min(gross_target, 0.04 * n_side)
    if gross <= 1e-12:
        return weights

    w = gross / (2.0 * n_side)
    if w > 0.02 + 1e-15:
        w = 0.02
        gross = 2.0 * n_side * w

    sorted_scores = score.sort_values(kind='mergesort')
    longs = list(sorted_scores.index[:n_side])
    shorts = list(sorted_scores.index[-n_side:])

    if set(longs) & set(shorts):
        n_side = min(n_side, valid_n // 2)
        if n_side < 1:
            return weights
        w = gross / (2.0 * n_side)
        longs = list(sorted_scores.index[:n_side])
        shorts = list(sorted_scores.index[-n_side:])

    for t in longs:
        weights.loc[t] = w
    for t in shorts:
        weights.loc[t] = -w

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    weights = weights.clip(-0.02, 0.02)

    gross_actual = weights.abs().sum()
    if gross_actual > 1.0 + 1e-12:
        weights = weights * (1.0 / gross_actual)

    return weights
