import numpy as np
import pandas as pd


def _last_month_end_position(index):
    if index is None or len(index) == 0:
        return -1
    try:
        ym = index.year.to_numpy() * 12 + index.month.to_numpy()
    except Exception:
        return len(index) - 1
    cur = ym[-1]
    prev = np.where(ym < cur)[0]
    if prev.size > 0:
        return int(prev[-1])
    return len(index) - 1


def generate_weights(history, universe):
    if universe is None:
        return pd.Series(dtype=float)
    universe = pd.Index(universe)
    if len(universe) == 0:
        return pd.Series(dtype=float)

    ac = history.get('adjusted_close')
    if ac is None or ac.empty:
        return pd.Series(0.0, index=universe)

    n = len(ac)
    if n < 25:
        return pd.Series(0.0, index=universe)

    cut = _last_month_end_position(ac.index)
    if cut < 0:
        cut = n - 1
    if cut + 1 < 23:
        cut = n - 1
    if cut + 1 < 23:
        return pd.Series(0.0, index=universe)

    px = ac.iloc[:cut + 1]
    last_px = px.iloc[-1]

    lag = min(21, len(px) - 1)
    if lag < 5:
        return pd.Series(0.0, index=universe)
    ret = last_px / px.iloc[-1 - lag] - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)

    daily_ret = px.pct_change()
    vol_tail = daily_ret.tail(60)
    vol = vol_tail.std(ddof=0)
    count = vol_tail.count()
    vol = vol.where(count >= 30)
    vol = vol.replace(0.0, np.nan).replace([np.inf, -np.inf], np.nan)

    close_hist = history.get('close')
    if close_hist is not None and not close_hist.empty:
        close_cut = close_hist.iloc[:cut + 1].iloc[-1]
    else:
        close_cut = last_px
    close_cut = close_cut.replace([np.inf, -np.inf], np.nan)

    volume = history.get('volume')
    if volume is not None and not volume.empty:
        avg_vol = volume.iloc[:cut + 1].tail(60).mean()
    else:
        avg_vol = pd.Series(np.nan, index=ac.columns)
    dollar_vol = avg_vol * close_cut

    if 'eligible' in history and history['eligible'] is not None and not history['eligible'].empty:
        e = history['eligible']
        elig = e.iloc[cut] if cut < len(e) else e.iloc[-1]
    else:
        elig = pd.Series(True, index=ac.columns)

    df = pd.DataFrame({
        'ret': ret,
        'vol': vol,
        'px': close_cut,
        'dollar_vol': dollar_vol,
        'eligible': elig,
    }).reindex(universe)

    valid = df['ret'].notna() & np.isfinite(df['ret'])
    valid &= df['vol'].notna() & np.isfinite(df['vol']) & (df['vol'] > 0)
    valid &= df['px'].notna() & np.isfinite(df['px']) & (df['px'] > 2.0)
    valid &= df['dollar_vol'].notna() & np.isfinite(df['dollar_vol']) & (df['dollar_vol'] > 1000000.0)
    valid &= df['eligible'].fillna(False).astype(bool)

    if valid.sum() < 30:
        valid = df['ret'].notna() & np.isfinite(df['ret'])
        valid &= df['vol'].notna() & np.isfinite(df['vol']) & (df['vol'] > 0)
        valid &= df['px'].notna() & np.isfinite(df['px']) & (df['px'] > 1.0)
    if valid.sum() < 20:
        valid = df['ret'].notna() & np.isfinite(df['ret'])
    if valid.sum() < 2:
        return pd.Series(0.0, index=universe)

    score = -df['ret'] / df['vol']
    score = score.replace([np.inf, -np.inf], np.nan)
    score = score.where(valid)
    s = score[valid]
    s = s[np.isfinite(s)]
    if len(s) >= 20:
        lo = s.quantile(0.01)
        hi = s.quantile(0.99)
        score = score.clip(lo, hi)
    score = score.where(valid)
    score = score.replace([np.inf, -np.inf], np.nan)

    ranked = score.dropna().sort_values(ascending=False)
    n_side = min(50, len(ranked) // 2)
    if n_side < 1:
        return pd.Series(0.0, index=universe)

    w_side = min(0.02, 0.5 / n_side)
    longs = ranked.iloc[:n_side].index
    shorts = ranked.iloc[-n_side:].index

    weights = pd.Series(0.0, index=universe, dtype=float)
    weights.loc[longs] += w_side
    weights.loc[shorts] -= w_side
    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights.astype(float)