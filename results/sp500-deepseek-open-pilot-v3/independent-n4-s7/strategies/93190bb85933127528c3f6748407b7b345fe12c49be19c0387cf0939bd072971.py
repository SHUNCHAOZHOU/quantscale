import numpy as np
import pandas as pd


def _last_week_cut(index):
    n = len(index)
    if n == 0:
        return -1
    try:
        dow = pd.Index(index).dayofweek.to_numpy()
        fridays = np.where(dow == 4)[0]
        if len(fridays) > 0:
            return int(fridays[-1])
    except Exception:
        pass
    return n - 1


def generate_weights(history, universe):
    if universe is None or len(universe) == 0:
        return pd.Series(dtype=float)
    universe = pd.Index(universe)

    ac = history.get('adjusted_close')
    if ac is None or ac.empty:
        return pd.Series(0.0, index=universe)

    n = len(ac)
    if n < 6:
        return pd.Series(0.0, index=universe)

    cut = _last_week_cut(ac.index)
    if cut < 0:
        return pd.Series(0.0, index=universe)

    # Use the Friday cutoff if enough history; otherwise fall back to the latest row.
    if cut + 1 < 22:
        cut = n - 1
    if cut + 1 < 6:
        return pd.Series(0.0, index=universe)

    px_hist = ac.iloc[:cut + 1]
    horizon = 5
    if len(px_hist) <= horizon:
        return pd.Series(0.0, index=universe)

    px_last = px_hist.iloc[-1]
    px_lag = px_hist.iloc[-1 - horizon]
    ret = px_last / px_lag - 1.0
    ret = ret.replace([np.inf, -np.inf], np.nan)

    vol_window = px_hist.iloc[-21:] if len(px_hist) >= 21 else px_hist
    daily_ret = vol_window.pct_change()
    vol = daily_ret.std(ddof=0)
    vol = vol.replace(0.0, np.nan).replace([np.inf, -np.inf], np.nan)

    volume = history.get('volume')
    if volume is not None and not volume.empty:
        vol_hist = volume.iloc[:cut + 1]
        avg_vol = vol_hist.iloc[-20:].mean()
    else:
        avg_vol = pd.Series(np.nan, index=ac.columns)

    close = history.get('close')
    if close is not None and not close.empty:
        close_hist = close.iloc[:cut + 1]
        px_filter = close_hist.iloc[-1]
    else:
        px_filter = px_last

    if 'eligible' in history and history['eligible'] is not None and not history['eligible'].empty:
        e = history['eligible']
        elig = e.iloc[cut] if cut < len(e) else e.iloc[-1]
    else:
        elig = pd.Series(True, index=ac.columns)

    df = pd.DataFrame({
        'ret': ret,
        'vol': vol,
        'avg_vol': avg_vol,
        'px': px_filter,
        'eligible': elig,
    })
    df = df.reindex(universe)

    dollar_vol = df['avg_vol'] * df['px']

    valid = df['ret'].notna() & np.isfinite(df['ret'])
    valid &= df['vol'].notna() & np.isfinite(df['vol']) & (df['vol'] > 0.0)
    valid &= df['px'].notna() & (df['px'] > 2.0)
    valid &= dollar_vol.notna() & (dollar_vol > 1_000_000.0)
    valid &= df['eligible'].fillna(False).astype(bool)

    # Relax filters if the liquid eligible set becomes too small.
    if valid.sum() < 20:
        valid = df['ret'].notna() & np.isfinite(df['ret'])
        valid &= df['vol'].notna() & np.isfinite(df['vol']) & (df['vol'] > 0.0)
        valid &= df['px'].notna() & (df['px'] > 1.0)

    if valid.sum() < 10:
        valid = df['ret'].notna() & np.isfinite(df['ret'])

    if valid.sum() < 2:
        return pd.Series(0.0, index=universe)

    # Negative return scaled by volatility: high score means a recent loser with lower noise.
    score = -df['ret'] / df['vol']
    score = score.replace([np.inf, -np.inf], np.nan)

    s_valid = score[valid]
    if s_valid.notna().sum() >= 20:
        lo = s_valid.quantile(0.01)
        hi = s_valid.quantile(0.99)
        score = score.clip(lo, hi)

    valid &= score.notna() & np.isfinite(score)
    ranked = score[valid].sort_values(ascending=False)

    n_side = min(50, len(ranked) // 2)
    if n_side < 1:
        return pd.Series(0.0, index=universe)

    w_side = min(0.02, 0.5 / n_side)
    longs = ranked.iloc[:n_side].index
    shorts = ranked.iloc[-n_side:].index

    weights = pd.Series(0.0, index=universe, dtype=float)
    weights.loc[longs] = w_side
    weights.loc[shorts] = -w_side

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return weights.astype(float)