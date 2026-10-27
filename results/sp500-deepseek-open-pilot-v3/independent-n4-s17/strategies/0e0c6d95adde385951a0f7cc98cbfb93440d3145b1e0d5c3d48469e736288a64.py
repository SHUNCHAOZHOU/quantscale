import numpy as np
import pandas as pd


def generate_weights(history, universe):
    weights = pd.Series(0.0, index=universe, dtype=float)
    tickers = list(universe)
    if len(tickers) == 0:
        return weights

    close = history.get('adjusted_close')
    if close is None or close.shape[0] < 130:
        return weights

    cols = [t for t in tickers if t in close.columns]
    if len(cols) < 30:
        return weights

    C = close[cols].astype(float)
    if C.shape[0] < 130:
        return weights

    last = C.iloc[-1]

    # Medium-term momentum: 6-month return, skipping the most recent month.
    # This reduces turnover and avoids contamination from short-term reversal.
    if C.shape[0] >= 127:
        p_start = C.iloc[-127]
    else:
        p_start = C.iloc[0]
    if C.shape[0] >= 22:
        p_end = C.iloc[-22]
    else:
        p_end = last
    mom = p_end / p_start - 1.0

    # Realized volatility over the last 60 trading days.
    rets = C.pct_change()
    tail = rets.tail(60)
    vol = tail.std()
    valid_vol_days = tail.notna().sum()

    # Liquidity screen using trailing 20-day average dollar volume.
    volume = history.get('volume')
    if volume is not None:
        V = volume.reindex(index=C.index, columns=cols).astype(float)
        adv = V.tail(20).mean()
        dollar_vol = adv * last
    else:
        dollar_vol = pd.Series(np.inf, index=cols, dtype=float)

    # Eligibility screen if supplied.
    eligible = history.get('eligible')
    if eligible is not None:
        try:
            e = eligible.reindex(index=C.index, columns=cols).iloc[-1]
            elig = e.fillna(False).astype(bool)
        except Exception:
            elig = pd.Series(True, index=cols)
    else:
        elig = pd.Series(True, index=cols)

    mask = last.notna() & mom.notna() & vol.notna() & dollar_vol.notna() & elig
    mask &= valid_vol_days >= 40
    mask &= last > 5.0
    mask &= dollar_vol > 5e6
    mask &= (vol > 0.005) & (vol < 0.08)

    if int(mask.sum()) < 30:
        return weights

    sub = pd.DataFrame({'mom': mom[mask], 'vol': vol[mask]})
    r_mom = sub['mom'].rank(pct=True)
    r_lowvol = (-sub['vol']).rank(pct=True)
    score = 0.6 * r_mom + 0.4 * r_lowvol

    n_candidates = len(score)
    k = min(40, max(10, int(n_candidates * 0.10)))
    if k * 2 > n_candidates:
        k = n_candidates // 2
    if k < 10:
        return weights

    ordered = score.sort_values()
    longs = ordered.index[-k:].tolist()
    shorts = ordered.index[:k].tolist()

    w = min(0.02, 0.5 / k)
    weights.loc[longs] = w
    weights.loc[shorts] = -w
    return weights
