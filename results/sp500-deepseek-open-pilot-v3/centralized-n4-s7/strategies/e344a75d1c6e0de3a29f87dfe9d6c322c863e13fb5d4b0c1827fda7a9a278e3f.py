import pandas as pd
import numpy as np


def generate_weights(history, universe):
    universe = pd.Index(universe)
    zeros = pd.Series(0.0, index=universe, dtype=float)

    try:
        ac = history['adjusted_close']
    except Exception:
        return zeros
    if ac is None or ac.shape[0] == 0 or ac.shape[1] == 0:
        return zeros

    dates = ac.index
    if len(dates) < 260:
        return zeros

    current_date = dates[-1]

    # Freeze signals at the most recent month-end to control turnover.
    s = pd.Series(dates, index=dates)
    me = s.groupby([dates.year, dates.month]).max()
    cand = me[me <= current_date]
    rebal = current_date if len(cand) == 0 else cand.iloc[-1]
    idx = ac.index.get_loc(rebal)
    if idx < 252:
        return zeros

    close_hist = ac.iloc[:idx + 1]
    window = close_hist.iloc[-252:]
    if window.shape[0] < 252:
        return zeros

    # 12-1 momentum: return over ~12 months ending ~1 month ago.
    p0 = window.iloc[0]
    p1 = window.iloc[-21]
    mom = (p1 / p0) - 1.0

    # Liquidity / tradability filters.
    vol_hist = history['volume'].iloc[:idx + 1].iloc[-20:]
    close_20 = close_hist.iloc[-20:]
    dollar = (close_20 * vol_hist).mean(axis=0)
    price = close_hist.iloc[-1]

    if 'eligible' in history:
        try:
            elig = history['eligible'].iloc[idx]
        except Exception:
            elig = pd.Series(True, index=ac.columns)
    else:
        elig = pd.Series(True, index=ac.columns)

    mom = mom.reindex(universe)
    d = dollar.reindex(universe)
    p = price.reindex(universe)
    e = elig.reindex(universe).fillna(False).astype(bool)

    mask = (
        mom.notna()
        & d.notna()
        & p.notna()
        & np.isfinite(mom)
        & (p > 1.0)
        & (d > 1_000_000.0)
        & e
    )
    cands = universe[mask]
    if len(cands) < 10:
        return zeros

    n_side = min(25, len(cands) // 2)
    if n_side < 5:
        return zeros

    # Volatility-management overlay: scale gross toward target market vol.
    rets_window = window.iloc[-61:].pct_change().iloc[1:]
    mkt = rets_window.mean(axis=1)
    mkt_vol = mkt.std(ddof=1) * np.sqrt(252.0)
    if not np.isfinite(mkt_vol) or mkt_vol <= 0:
        mkt_vol = 0.15
    scale = float(np.clip(0.15 / mkt_vol, 0.4, 1.0))

    w = min(0.02, scale * 0.5 / n_side)

    vals = mom.loc[cands].values
    order = np.argsort(vals)
    sorted_c = cands[order]

    shorts = sorted_c[:n_side]      # weakest momentum
    longs = sorted_c[-n_side:]      # strongest momentum

    weights = pd.Series(0.0, index=universe, dtype=float)
    weights.loc[longs] = w
    weights.loc[shorts] = -w

    weights = weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    weights = weights.clip(-0.02, 0.02)

    gross = weights.abs().sum()
    if gross > 1.0:
        weights = weights / gross

    return weights